import numpy as np
import pandas as pd

from modules import eda_engine as eda


def test_field_names_are_normalised():
    df = pd.DataFrame(columns=["First Name", "DOB (YYYY)", "e-mail  ", "ok_name"])
    out, renames = eda.clean_field_names(df)
    assert list(out.columns) == ["first_name", "dob_yyyy", "e_mail", "ok_name"]
    assert "ok_name" not in renames


def test_null_removal_passes():
    df = pd.DataFrame({
        "all_null": [None] * 5,
        "a": ["x", None, "x", "x", "x"], "b": ["y", None, None, "y", "y"], "c": ["z", None, None, None, "z"],
    })
    out, log = eda.remove_null_elements(df)
    assert "all_null" in log["null_columns_dropped"]
    assert log["all_empty_rows"] == 1                   # row 1 is empty
    assert log["one_value_rows"] == 1                    # row 2 has one value
    assert log["two_value_rows"] == 1                    # row 3 has two values
    assert len(out) == 2                           # only rows with >= 3 populated values survive


def test_text_cleaning_trims_and_nulls_markers():
    out = eda.clean_text_fields(pd.DataFrame({"a": ["  hi   there ", "N/A", "None", ""]}))
    assert out["a"].iloc[0] == "hi there"
    assert out["a"].iloc[1:].isna().all()


def test_duplicate_removal():
    out, n = eda.remove_duplicate_records(pd.DataFrame({"a": [1, 1, 2]}))
    assert n == 1 and len(out) == 2


def test_dates_are_standardised_but_bare_years_are_left_alone():
    df = pd.DataFrame({"dob": ["04/02/1975", "1980-01-31", "12 Mar 1990"], "birth_year": ["1975", "1980", "1990"]})
    out, log = eda.standardise_date_formats(df)
    assert "dob" in log and "birth_year" not in log
    assert out["birth_year"].tolist() == ["1975", "1980", "1990"]
    assert out["dob"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()


def test_mixed_date_formats_in_one_column_are_all_kept():
    """pd.to_datetime alone would infer one format from row 0 and NaT the rest."""
    df = pd.DataFrame({"dob": ["04/02/1975", "1980-01-31", "12 Mar 1990", "25/12/1990"]})
    out, _ = eda.standardise_date_formats(df)
    assert out["dob"].notna().all()
    assert out["dob"].iloc[1] == "1980-01-31" and out["dob"].iloc[2] == "1990-03-12"


def test_ambiguous_dates_are_day_first_only_when_the_column_proves_it():
    month_first = eda.parse_mixed_dates(pd.Series(["04/02/1975", "03/04/1985"]))
    assert month_first.iloc[0].month == 4
    day_first = eda.parse_mixed_dates(pd.Series(["04/02/1975", "25/12/1985"]))
    assert day_first.iloc[0].month == 2 and day_first.iloc[1].day == 25


def test_field_type_inference():
    types = eda.infer_field_types(["unique_id", "first_name", "last_name", "full_name", "dob", "email",
                                   "zip_code", "gender", "city", "notes"])
    assert types == {"unique_id": "id", "first_name": "first_name", "last_name": "surname",
                     "full_name": "full_name", "dob": "dob", "email": "email", "zip_code": "postcode",
                     "gender": "gender", "city": "location", "notes": "text"}


def test_run_full_eda_end_to_end():
    raw = pd.DataFrame({
        "First Name": ["Ann", "Ann", "Bob", None, "Cy"],
        "DOB": ["01/02/1990", "01/02/1990", "03/04/1985", None, "05/06/1970"],
        "City": ["Leeds", "Leeds", "York", None, "Bath"], "Empty": [np.nan] * 5,
    })
    clean, types, summary, log = eda.run_full_eda(raw)
    assert "empty" not in clean.columns and summary["cols_removed"] == 1
    assert summary["original_rows"] == 5 and summary["final_rows"] == len(clean) == 3   # 1 null row, 1 duplicate
    assert types["first_name"] == "first_name" and types["dob"] == "dob"
    assert clean["dob"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()
    assert log["duplicates_removed"] == 1


def test_high_correlation_pairs_threshold():
    df = pd.DataFrame({"a": list("abcdefghij"), "b": list("abcdefghij"), "c": list("abcdefghik"),
                       "unique_id": range(10)})
    pairs = eda.find_high_correlation_pairs(df, [])
    assert [(p[0], p[1]) for p in pairs] == [("a", "b")]                     # a~c is 0.9 agreement < 0.95
    assert len(eda.find_high_correlation_pairs(df, [], threshold=0.85)) == 3


def test_suggestions():
    types = {"first_name": "first_name", "email": "email", "gender": "gender", "uid": "id"}
    assert eda.suggest_comparison_types(types) == {
        "first_name": "JaroWinklerAtThresholds", "email": "EmailComparison", "gender": "ExactMatch"}
    blocking = eda.suggest_blocking_rules(types)
    assert blocking["first_name"] and not blocking["gender"]


def test_min_populated_is_adjustable():
    df = pd.DataFrame({"a": ["x", "x", None], "b": ["y", None, None], "c": ["z", None, None]})
    assert len(eda.remove_null_elements(df)[0]) == 1                    # default: need 3 values
    assert len(eda.remove_null_elements(df, min_populated=2)[0]) == 1   # row 1 has only one value
    assert len(eda.remove_null_elements(df, min_populated=1)[0]) == 2


def test_selective_field_detection():
    assert eda.is_selective_field(pd.Series([f"v{i}" for i in range(100)]))
    assert not eda.is_selective_field(pd.Series(["M", "F"] * 50))
    assert not eda.is_selective_field(pd.Series([None, None]))
