"""The voter demo is synthetic: statistically like the NC file, containing no real record."""

import pandas as pd

from modules import voter_data as vd

PII_COLUMNS = {"res_street_address", "mail_addr1", "full_phone_number", "ssn", "drivers_lic", "phone"}


def test_profile_holds_only_aggregates_and_no_rare_names():
    profile = vd.load_profile()
    assert profile["n_rows"] > 1_000_000
    names = [*profile["last_names"].values(), *(c for g in profile["first_names"].values() for c in g.values())]
    assert min(names) >= profile["min_name_count"] >= 500              # no name that could single someone out
    assert all(isinstance(v, (int, dict)) or k == "source" for k, v in profile.items())


def test_generation_is_deterministic_and_sized():
    one, two = vd.generate_voter_registry(800, seed=3), vd.generate_voter_registry(800, seed=3)
    pd.testing.assert_frame_equal(one, two)
    assert len(one) == 800


def test_no_address_phone_or_id_document_columns():
    assert not PII_COLUMNS & set(vd.generate_voter_registry(100).columns)


def test_every_name_comes_from_the_high_count_profile_lists():
    profile, registry = vd.load_profile(), vd.generate_voter_registry(1500)
    firsts = {n for g in profile["first_names"].values() for n in g}
    assert set(registry["last_name"]) <= set(profile["last_names"])
    # only second registrations may carry a one-letter typo (about 1% of rows)
    assert registry["first_name"].isin(firsts).mean() > 0.98


def test_demographics_follow_the_profile():
    registry = vd.generate_voter_registry(6000)
    profile = vd.load_profile()
    expected_white = profile["race_code"]["W"] / sum(profile["race_code"].values())
    assert abs((registry["race_code"] == "W").mean() - expected_white) < 0.03
    assert set(registry["gender_code"]) <= {"F", "M", "U"}
    assert (registry["age_at_year_end"].astype(int) == 2026 - registry["birth_year"].astype(int)).all()
    assert registry["birth_year"].astype(int).max() <= 2026 - 18             # voting age


def test_dates_are_valid_and_consistent():
    registry = vd.generate_voter_registry(1000)
    dob = pd.to_datetime(registry["dob"], format="%Y-%m-%d", errors="coerce")
    assert dob.notna().all() and (dob.dt.year.astype(str) == registry["birth_year"]).all()
    registered = pd.to_datetime(registry["registr_dt"], format="%m/%d/%Y", errors="coerce")
    assert registered.notna().all() and (registered.dt.year >= dob.dt.year + 18).all()


def test_duplicates_share_a_cluster_but_not_an_id():
    registry = vd.generate_voter_registry(2000)
    assert registry["ncid"].is_unique and registry["voter_reg_num"].is_unique
    assert 0.03 < registry["cluster"].duplicated().mean() < 0.07


def test_dataset_b_links_back_to_a_with_damage():
    a, b = vd.build_voter_datasets(2000)
    assert len(b) == 1000 and (b["source_dataset"] == "B").all()
    assert set(b["cluster"]) <= set(a["cluster"])
    assert b["ncid"].str.endswith("_B").all() and not set(a["ncid"]) & set(b["ncid"])
    sample = a.drop_duplicates("cluster").set_index("cluster").loc[b["cluster"].values]
    assert 0.02 < (sample["first_name"].values != b["first_name"].values).mean() < 0.3


def test_loader_cleans_and_labels_both_datasets():
    from modules.data_builder import load_voter_datasets, voter_linkage_defaults
    a, b, field_types, log = load_voter_datasets(1000)
    assert (a["source_dataset"] == "A").all() and (b["source_dataset"] == "B").all()
    assert a["unique_id"].is_unique and b["unique_id"].is_unique
    assert field_types["dob"] == "dob" and field_types["birth_year"] == "text" and field_types["birth_state"] == "text"
    fields, toggles = voter_linkage_defaults(a.columns)
    assert "ncid" not in fields and toggles["first_name"] and not toggles["gender_code"]
    assert log["summary"]["final_rows"] == 1000
