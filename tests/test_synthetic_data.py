import pandas as pd

from modules import synthetic_data as sd

def test_people_generation_is_deterministic():
    pd.testing.assert_frame_equal(sd.generate_people(300, seed=3), sd.generate_people(300, seed=3))
    assert not sd.generate_people(300, seed=3).equals(sd.generate_people(300, seed=4))


def test_people_size_columns_and_duplicates():
    df = sd.generate_people(1000)
    assert len(df) == 1000
    assert {"unique_id", "first_name", "surname", "dob", "city", "email", "gender", "postcode", "cluster"} <= set(df)
    assert df["unique_id"].is_unique
    assert df["cluster"].nunique() < len(df)             # some people appear more than once


def test_people_values_are_plausible():
    df = sd.generate_people(500)
    assert df["gender"].isin(["F", "M"]).all()
    dobs = pd.to_datetime(df["dob"].dropna(), format="%Y-%m-%d", errors="coerce")
    assert dobs.notna().all()                             # transposed dates stay valid
    assert dobs.between("1938-01-01", "2006-12-31").all()
    assert df["email"].dropna().str.endswith(tuple(sd.EMAIL_DOMAINS)).all()    # reserved example domains only
    assert df["city"].isin(sd.UK_CITIES).all()


def test_demo_datasets_link_back_to_each_other():
    base, a, b = sd.build_demo_datasets()
    assert (a["source_dataset"] == "A").all() and (b["source_dataset"] == "B").all()
    assert len(b) == 500
    assert b["unique_id"].str.endswith("_B").all()
    assert set(b["cluster"]) <= set(a["cluster"])         # every B record has a true A match
    assert not set(a["unique_id"]) & set(b["unique_id"])


def test_demo_dataset_b_is_actually_damaged():
    _, a, b = sd.build_demo_datasets()
    originals = a.set_index("unique_id").loc[b["unique_id"].str.replace("_B", "")]
    changed = (originals["first_name"].values != b["first_name"].values).mean()
    assert 0.03 < changed < 0.25
