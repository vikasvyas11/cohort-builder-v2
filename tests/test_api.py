"""Click through the HTTP API end to end on the small demo data: every step of every flow."""

import io
import time

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

PEOPLE_CONFIG = {
    "fields": ["first_name", "surname", "dob", "email", "postcode"],
    "blocking_toggles": {"first_name": True, "surname": True, "dob": True, "email": False, "postcode": False},
    "operation_mode": "dedupe", "linkage_type": "probabilistic",
}


def wait(job_id: str, timeout: float = 240) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            return status
        time.sleep(0.3)
    raise AssertionError("job did not finish")


def run(sid: str, slot: str, config: dict) -> dict:
    response = client.post(f"/api/sessions/{sid}/runs/{slot}", json=config)
    assert response.status_code == 200, response.text
    status = wait(response.json()["job_id"])
    assert status["status"] == "done", status
    return client.get(f"/api/sessions/{sid}/runs/{slot}").json()


@pytest.fixture(scope="module")
def people():
    response = client.post("/api/sessions", json={"source": "people"})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="module")
def run1(people):
    return people["session_id"], run(people["session_id"], "run1", PEOPLE_CONFIG)


def test_health_and_config():
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/config").json()["assistant_url"].startswith("https://huggingface.co/spaces/")


def test_overview_has_defaults_and_profile(people):
    assert people["rows_a"] > 500 and people["has_ground_truth"]
    defaults = people["defaults"]
    assert "first_name" in defaults["fields"] and defaults["blocking_toggles"]["gender"] is False
    assert people["demographics"], "gender/city profile charts expected"


def test_estimate_gives_a_verdict(people):
    result = client.post(f"/api/sessions/{people['session_id']}/estimate", json=PEOPLE_CONFIG).json()
    assert result["pairs"] > 0 and result["level"] in ("ok", "warn")


def test_run_one_has_every_tab(run1):
    sid, summary = run1
    assert summary["cards"]["edges"] > 0 and summary["accuracy"]["available"]
    assert summary["accuracy"]["cm"]["tp"] > 0 and summary["accuracy"]["curve_figures"]
    assert summary["edges"]["charts"] and summary["clusters"]["size_figure"]
    assert client.get(f"/api/sessions/{sid}/runs/run1/demographics").status_code == 200
    raw = client.get(f"/api/sessions/{sid}/runs/run1/raw").json()
    assert raw["predict"] and raw["cluster"]
    assert "<html" in client.get(f"/api/sessions/{sid}/runs/run1/studio").text.lower()


def test_explorer_follows_the_toggles(run1):
    sid, summary = run1
    all_on = client.post(f"/api/sessions/{sid}/runs/run1/explorer", json={}).json()
    rules = [r["rule"] for r in all_on["rules"] if r["on"]]
    one_off = client.post(f"/api/sessions/{sid}/runs/run1/explorer",
                          json={"toggles": {rules[0]: False}}).json()
    assert one_off["stats"]["candidate_pairs"] <= all_on["stats"]["candidate_pairs"]
    assert all_on["waterfall"]["fields"] and all_on["table"]
    none_on = client.post(f"/api/sessions/{sid}/runs/run1/explorer", json={"toggles": {r: False for r in rules}}).json()
    assert none_on["stats"]["candidate_pairs"] == 0
    cluster = client.post(f"/api/sessions/{sid}/runs/run1/recluster", json={"toggles": {}, "threshold": 0.8})
    assert cluster.json()["new_clusters"] > 0


def test_live_waterfall_reacts_to_new_and_combined_rules(run1):
    sid, _ = run1
    base = {"first_name": True, "surname": True, "dob": True, "email": False, "postcode": False}
    same = client.post(f"/api/sessions/{sid}/waterfall", json={"toggles": base}).json()
    assert same["totals"]["change"] == 0
    more = client.post(f"/api/sessions/{sid}/waterfall", json={"toggles": {**base, "postcode": True}}).json()
    assert more["totals"]["after"] > same["totals"]["after"] and "postcode" in more["fields"]   # a rule Run 1 never used
    combo = client.post(f"/api/sessions/{sid}/waterfall",
                        json={"toggles": {"first_name": False, "surname": False, "dob": False, "first_name+surname": True}}).json()
    assert "first_name+surname" in combo["fields"] and combo["totals"]["after"] > 0
    assert combo["totals"]["after"] < same["totals"]["after"]
    fewer = client.post(f"/api/sessions/{sid}/waterfall", json={"toggles": {**base, "dob": False}}).json()
    assert fewer["totals"]["after"] <= same["totals"]["after"] and fewer["left"] and fewer["right"]


def test_run_two_compare_and_exports(run1):
    sid, _ = run1
    second = run(sid, "run2", {**PEOPLE_CONFIG, "blocking_toggles": {**PEOPLE_CONFIG["blocking_toggles"], "postcode": True}})
    assert second["cards"]["edges"] > 0 and second["accuracy"]["available"]       # same tabs as Run 1
    compared = client.get(f"/api/sessions/{sid}/compare").json()
    assert compared["changes"] and compared["accuracy"]
    cohort = client.get(f"/api/sessions/{sid}/runs/run2/cohort.csv")
    assert cohort.headers["content-type"].startswith("text/csv") and b"cluster_id" in cohort.content[:2000]
    report = client.get(f"/api/sessions/{sid}/runs/run1/report.html")
    assert b"Linkage run report" in report.content
    model = client.get(f"/api/sessions/{sid}/runs/run1/model.json")
    assert model.json()["comparisons"]


def test_advanced_flow_round_trips_a_saved_model(run1):
    sid, _ = run1
    model = client.get(f"/api/sessions/{sid}/runs/run1/model.json").content
    fresh = client.post("/api/sessions", json={"source": "people"}).json()["session_id"]
    uploaded = client.post(f"/api/sessions/{fresh}/model", files={"file": ("m.json", model, "application/json")})
    assert uploaded.status_code == 200 and uploaded.json()["detected_linkage_type"] == "probabilistic"
    summary = run(fresh, "run1", {"from_model": True, "operation_mode": "dedupe", "linkage_type": "probabilistic"})
    assert summary["cards"]["edges"] > 0


def test_upload_flow_dedupe_and_generated_dataset_b():
    csv = "id,first_name,surname,city\n" + "\n".join(
        f"{i},{n},{s},{c}" for i, (n, s, c) in enumerate(
            [("anna", "smith", "leeds"), ("anna", "smith", "leeds"), ("bob", "jones", "york"), ("carl", "brown", "bath"),
             ("dina", "king", "hull"), ("evan", "wood", "ely")] * 12))
    created = client.post("/api/sessions/upload", data={"mode": "sample", "id_col_a": ""},
                          files={"file_a": ("people.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert created.status_code == 200, created.text
    over = created.json()
    assert over["source"] == "upload" and over["rows_b"] is None
    made = client.post(f"/api/sessions/{over['session_id']}/dataset-b",
                       json={"sample_frac": 0.5, "rates": {"first_name": 0.2}, "letters": 1, "year_shift": 1})
    assert made.status_code == 200 and made.json()["rows_b"] > 0 and made.json()["has_ground_truth"]


def test_errors_are_reported_not_raised(people):
    sid = people["session_id"]
    bad = client.post(f"/api/sessions/{sid}/runs/run1", json={**PEOPLE_CONFIG, "fields": ["nope"]})
    status = wait(bad.json()["job_id"])
    assert status["status"] == "error" and "None of the selected fields" in status["error"]
    assert client.get("/api/sessions/does-not-exist").status_code == 404
    assert client.post(f"/api/sessions/{sid}/runs/run3", json={}).status_code == 404
    assert client.post("/api/sessions", json={"source": "voters", "n_rows": 7}).status_code == 400
    assert client.post(f"/api/sessions/{sid}/model", files={"file": ("m.json", b"not json")}).status_code == 400
