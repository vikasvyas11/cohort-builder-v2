"""Address and phone columns for the upload samples: shaped like the voter file, drawn from aggregates only,
and good enough to link on end to end through the Upload flow."""

import time

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from modules.voter_contact import add_address_and_phone, damage_address_and_phone, load_extras
from modules.voter_data import generate_voter_registry
from tools.make_upload_samples import main as make_samples


def test_new_columns_have_the_files_shape():
    registry = generate_voter_registry(1000)
    out = add_address_and_phone(registry)
    assert {"res_street_address", "full_phone_number"} <= set(out.columns) - set(registry.columns)
    phones = out["full_phone_number"].dropna()
    assert phones.str.fullmatch(r"\d{10}").all()
    assert 0.3 < out["full_phone_number"].notna().mean() < 0.55            # about 43% in the source file
    streets = out["res_street_address"].str.replace(r"^\d+ ", "", regex=True).str.replace(r" (?:APT|UNIT|LOT) \d+$", "", regex=True)
    assert set(streets) <= set(load_extras()["street_names"])               # only streets shared by hundreds of addresses
    assert "REMOVED" not in set(streets)
    assert out.equals(add_address_and_phone(registry))                       # seeded


def test_registrations_of_one_voter_share_a_phone_number():
    out = add_address_and_phone(generate_voter_registry(1000))
    groups = out.dropna(subset=["full_phone_number"]).groupby("cluster")["full_phone_number"].nunique()
    assert (groups == 1).all()


def test_damage_changes_some_values_and_blanks_others():
    out = add_address_and_phone(generate_voter_registry(2000))
    damaged = damage_address_and_phone(out)
    assert damaged["res_street_address"].isna().sum() > out["res_street_address"].isna().sum()
    assert (damaged["full_phone_number"].notna() & (damaged["full_phone_number"] != out["full_phone_number"])).sum() > 50


def test_the_sample_files_link_through_the_upload_flow(tmp_path):
    make_samples(tmp_path, rows=2000)
    a, b = tmp_path / "voters_2k_A.csv", tmp_path / "voters_2k_B.csv"
    assert pd.read_csv(b, dtype=str).shape[0] == 1000 and "source_dataset" not in pd.read_csv(b, nrows=1).columns
    client = TestClient(app)
    created = client.post("/api/sessions/upload", data={"mode": "uploaded", "id_col_a": "ncid", "id_col_b": "ncid"},
                          files={"file_a": ("a.csv", a.read_bytes(), "text/csv"), "file_b": ("b.csv", b.read_bytes(), "text/csv")})
    assert created.status_code == 200, created.text
    over = created.json()
    assert over["has_ground_truth"] and over["rows_b"] == 1000
    assert {"res_street_address", "full_phone_number"} <= {c["column"] for c in over["column_info"]}
    config = {"fields": ["first_name", "last_name", "dob", "res_street_address", "full_phone_number", "zip_code"],
              "blocking_toggles": {"dob": True, "last_name": True}, "operation_mode": "link_dedupe",
              "linkage_type": "probabilistic", "comp_types": over["defaults"]["comp_types"]}
    job = client.post(f"/api/sessions/{over['session_id']}/runs/run1", json=config).json()["job_id"]
    for _ in range(300):
        status = client.get(f"/api/jobs/{job}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert status["status"] == "done", status
    cm = client.get(f"/api/sessions/{over['session_id']}/runs/run1").json()["accuracy"]["cm"]
    assert cm["recall"] > 0.8 and cm["precision"] > 0.8
