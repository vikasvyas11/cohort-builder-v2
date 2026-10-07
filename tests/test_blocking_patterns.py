"""The live blocking waterfall: pair counts per rule, worked out by hand on a tiny table."""

import pandas as pd

from modules.splink_runner import blocking_rule_patterns, compute_blocking_waterfall, determine_cascade_order

# Records 0-3. Pairs that agree: first (0,1) (0,2) (1,2); last (0,1) (2,3); first+last (0,1).
DATA = pd.DataFrame({
    "unique_id": [0, 1, 2, 3],
    "first": ["ann", "ann", "ann", "bob"],
    "last": ["x", "x", "y", "y"],
    "dob": [None, None, "d", "d"],
})


def _waterfall(rules, toggles, **kw):
    patterns, skipped = blocking_rule_patterns(DATA, None, "dedupe", rules, **kw)
    order = determine_cascade_order(patterns, rules)
    return compute_blocking_waterfall(patterns, order, toggles), skipped


def test_single_rules_cover_the_hand_counted_pairs():
    data, skipped = _waterfall(["first", "last"], {"first": True, "last": True})
    assert not skipped
    assert data["grand_total"] == 4                     # (0,1) (0,2) (1,2) (2,3)
    assert data["active_total"] == 4
    assert sum(data["active_only_count"].values()) == 4  # no pair is counted twice


def test_switching_a_rule_off_hands_pairs_to_the_other_rule():
    data, _ = _waterfall(["first", "last"], {"first": False, "last": True})
    assert data["grand_total"] == 4
    assert data["active_total"] == 2                    # only (0,1) and (2,3) agree on last
    assert data["active_only_count"] == {"last": 2}


def test_a_new_combined_rule_is_counted_even_though_no_run_used_it():
    data, _ = _waterfall(["first", "last", "first+last"], {"first": False, "last": False, "first+last": True})
    assert data["active_total"] == 1                    # (0,1) is the only pair agreeing on both
    assert data["all_active_count"].get("first+last", 0) == 0   # with its parts on, the combined rule adds nothing
    assert data["grand_total"] == 4


def test_missing_values_never_agree():
    data, _ = _waterfall(["dob"], {"dob": True})
    assert data["grand_total"] == 1                     # (2,3); two missing dobs do not match


def test_link_mode_counts_cross_dataset_pairs_only():
    b = pd.DataFrame({"unique_id": [10, 11], "first": ["ann", "zed"], "last": ["x", "q"], "dob": [None, None]})
    patterns, _ = blocking_rule_patterns(DATA, b, "link", ["first", "last"])
    assert int(patterns["n_pairs"].sum()) == 3          # ann(0,1,2)-ann(10) and x(0,1)-x(10), union = 3 pairs


def test_unknown_columns_and_oversized_rules_are_reported_not_fatal():
    patterns, skipped = blocking_rule_patterns(DATA, None, "dedupe", ["first", "nope", "last"], max_pairs=3)
    assert ("nope", None) in skipped
    assert ("first", 3) in skipped                      # the more selective rule is admitted first
    assert not patterns.empty
