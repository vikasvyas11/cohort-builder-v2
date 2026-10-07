import random

import pandas as pd
import pytest

from modules import corruption as co


@pytest.fixture
def people():
    return pd.DataFrame({
        "unique_id": [str(i) for i in range(400)],
        "first_name": ["Alexander"] * 400,
        "dob": ["1980-06-15"] * 400,
        "birth_year": ["1980"] * 400,
        "gender_code": ["M", "F", "U", "M"] * 100,
        "email": ["a.b@example.com"] * 400,
        "source_dataset": "A",
    })


def test_text_corruption_changes_exactly_n_letters():
    out = co.corrupt_text("Alexander", random.Random(1), n_edits=2)
    assert len(out) == 9 and sum(a != b for a, b in zip(out, "Alexander")) == 2


def test_text_corruption_preserves_case_class_and_handles_no_letters():
    assert co.corrupt_text("12345", random.Random(0)) == "12345"
    assert co.corrupt_text("ABC", random.Random(0), 3).isupper()


def test_year_shift_magnitude():
    assert co.corrupt_year("1980", random.Random(0), 3) in {"1977", "1983"}
    assert co.corrupt_year("n/a", random.Random(0)) == "n/a"


def test_date_corruption_stays_iso_or_blank():
    rng = random.Random(3)
    seen = {co.corrupt_date("2001-03-04", rng) for _ in range(200)}
    assert None in seen
    assert all(v is None or pd.to_datetime(v, format="%Y-%m-%d") for v in seen)
    assert co.corrupt_date("not a date", rng) == "not a date"


def test_classify_field_kinds(people):
    assert co.classify_field(people["birth_year"]) == "year"
    assert co.classify_field(people["dob"], "dob") == "date"
    assert co.classify_field(people["gender_code"]) == "category"
    assert co.classify_field(people["email"], "email") == "email"
    # a name column in a tiny cohort must not be mistaken for a code
    assert co.classify_field(pd.Series(["Ann", "Bob", "Ann"]), "first_name") == "text"


def test_rate_zero_changes_nothing_and_rate_one_changes_everything(people):
    types = {"first_name": "first_name"}
    none = co.make_noisy_copy(people, {"first_name": 0.0}, sample_frac=1.0, field_types=types)
    assert (none["first_name"] == "Alexander").all()
    every = co.make_noisy_copy(people, {"first_name": 1.0}, sample_frac=1.0, field_types=types)
    assert (every["first_name"] != "Alexander").all()


def test_intermediate_rate_is_roughly_respected(people):
    out = co.make_noisy_copy(people, {"first_name": 0.25}, sample_frac=1.0, seed=5,
                             field_types={"first_name": "first_name"})
    assert 0.15 < (out["first_name"] != "Alexander").mean() < 0.35


def test_categories_never_leave_the_observed_code_set(people):
    out = co.make_noisy_copy(people, {"gender_code": 1.0}, sample_frac=1.0)
    assert set(out["gender_code"]) <= {"M", "F", "U"}
    original = people.set_index("unique_id").loc[out["unique_id"].str.replace("_B", ""), "gender_code"]
    assert (out["gender_code"].values != original.values).all()


def test_bookkeeping_columns_and_ground_truth(people):
    out = co.make_noisy_copy(people, {"unique_id": 1.0, "first_name": 0.5}, sample_frac=0.5, seed=1,
                             field_types={"first_name": "first_name"})
    assert len(out) == 200
    assert (out["source_dataset"] == "B").all()
    assert out["unique_id"].str.endswith("_B").all()
    # the B record points back at the A record it came from
    assert (out["cluster"] == out["unique_id"].str.replace("_B", "")).all()


def test_existing_cluster_column_is_kept(people):
    people["cluster"] = "E" + people["unique_id"]
    out = co.make_noisy_copy(people, {}, sample_frac=0.5)
    assert out["cluster"].str.startswith("E").all()


def test_same_seed_gives_identical_output(people):
    kwargs = dict(seed=9, field_types={"first_name": "first_name", "dob": "dob"})
    one = co.make_noisy_copy(people, {"first_name": 0.5, "dob": 0.5}, **kwargs)
    two = co.make_noisy_copy(people, {"first_name": 0.5, "dob": 0.5}, **kwargs)
    pd.testing.assert_frame_equal(one, two)


def test_missing_rates_blank_values(people):
    out = co.make_noisy_copy(people, {}, sample_frac=1.0, missing_rates={"email": 1.0})
    assert out["email"].isna().all()


def test_constant_column_without_a_type_is_treated_as_a_code_and_left_alone(people):
    """A column with one distinct value has no alternative to swap to."""
    out = co.make_noisy_copy(people, {"first_name": 1.0}, sample_frac=1.0)
    assert (out["first_name"] == "Alexander").all()


def test_works_without_unique_id_column():
    df = pd.DataFrame({"first_name": ["Anna", "Bea", "Cy", "Di"]})
    out = co.make_noisy_copy(df, {"first_name": 1.0}, sample_frac=1.0)
    assert out["cluster"].notna().all()


def test_invalid_sample_fraction_rejected(people):
    with pytest.raises(ValueError):
        co.make_noisy_copy(people, {}, sample_frac=0)


def test_email_and_location_variants_stay_plausible():
    rng = random.Random(4)
    emails = {co.corrupt_email("ann.lee12@example.com", rng) for _ in range(60)}
    assert len(emails) >= 3 and all("@" in e for e in emails)
    assert co.corrupt_location("New Haven", rng) == "N. Haven"
    assert co.corrupt_location("Bristol", rng) == "Bris."
