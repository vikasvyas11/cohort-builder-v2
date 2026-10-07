"""Metrics are checked against small hand-built cases whose answers can be worked out on paper."""

import math

import numpy as np
import pandas as pd
import pytest

from modules import metrics_engine as me


def _nodes(ids, clusters, source="A"):
    return pd.DataFrame({"unique_id": ids, "cluster": clusters, "source_dataset": source})


def _edges(pairs, probs, source=("A", "A")):
    probs = np.array(probs, dtype=float)
    return pd.DataFrame({
        "unique_id_l": [p[0] for p in pairs], "unique_id_r": [p[1] for p in pairs],
        "source_dataset_l": source[0], "source_dataset_r": source[1],
        "match_probability": probs, "match_weight": np.log2(probs / (1 - probs + 1e-9)) if len(probs) else probs,
    })


# -- ground truth ------------------------------------------------------------

def test_ground_truth_pairs_dedupe_counts_within_cluster_pairs():
    # clusters: x={1,2,3} -> 3 pairs, y={4,5} -> 1 pair, z={6} -> 0 pairs
    a = _nodes(["1", "2", "3", "4", "5", "6"], ["x", "x", "x", "y", "y", "z"])
    assert len(me.ground_truth_pairs(a, None, "dedupe")) == 4


def test_ground_truth_pairs_link_only_counts_cross_dataset_pairs():
    a = _nodes(["1", "2"], ["x", "x"], "A")          # two A records of the same entity
    b = _nodes(["1_B"], ["x"], "B")
    # A-B pairs only: (1,1_B) and (2,1_B); the A-A pair is not a link candidate
    assert len(me.ground_truth_pairs(a, b, "link_dedupe")) == 2


def test_pair_key_is_order_independent():
    fwd = _edges([("1", "2")], [0.9])
    rev = _edges([("2", "1")], [0.9])
    assert me._edge_pair_keys(fwd).iloc[0] == me._edge_pair_keys(rev).iloc[0]


def test_integer_ids_do_not_break_keys():
    a = pd.DataFrame({"unique_id": [1, 2, 3], "cluster": [7, 7, 8], "source_dataset": "A"})
    e = pd.DataFrame({"unique_id_l": ["1"], "unique_id_r": ["2"], "source_dataset_l": "A",
                      "source_dataset_r": "A", "match_probability": [0.9]})
    cm = me.compute_confusion_matrix(e, a, None, "dedupe")
    assert (cm["tp"], cm["fp"], cm["fn"]) == (1, 0, 0)


# -- confusion matrix --------------------------------------------------------

def test_confusion_matrix_hand_computed():
    a = _nodes(["1", "2", "3", "4", "5"], ["x", "x", "x", "y", "y"])      # true pairs: 12,13,23,45
    e = _edges([("1", "2"), ("1", "3"), ("1", "4"), ("4", "5")], [0.9, 0.9, 0.9, 0.9])
    cm = me.compute_confusion_matrix(e, a, None, "dedupe")
    assert (cm["tp"], cm["fp"], cm["fn"]) == (3, 1, 1)
    assert cm["n_gt_edges"] == 4
    assert cm["precision"] == pytest.approx(0.75)
    assert cm["recall"] == pytest.approx(0.75)
    assert cm["f1"] == pytest.approx(0.75)
    assert cm["fstar"] == pytest.approx(3 / 5)          # tp / (tp + fp + fn)
    assert cm["fdr"] == pytest.approx(0.25) and cm["fnr"] == pytest.approx(0.25)


def test_confusion_matrix_counts_pairs_blocking_never_generated_as_misses():
    a = _nodes(["1", "2", "3"], ["x", "x", "x"])                          # 3 true pairs
    cm = me.compute_confusion_matrix(_edges([("1", "2")], [0.9]), a, None, "dedupe")
    assert cm["fn"] == 2 and cm["recall"] == pytest.approx(1 / 3, abs=1e-4)


def test_confusion_matrix_at_an_operating_threshold():
    a = _nodes(["1", "2", "3", "4"], ["x", "x", "y", "y"])               # true pairs: 12, 34
    e = _edges([("1", "2"), ("1", "3"), ("3", "4")], [0.9, 0.6, 0.3])    # TP, FP, TP
    loose = me.compute_confusion_matrix(e, a, None, "dedupe")
    strict = me.compute_confusion_matrix(e, a, None, "dedupe", min_match_probability=0.8)
    assert (loose["tp"], loose["fp"], loose["fn"]) == (2, 1, 0)
    assert (strict["tp"], strict["fp"], strict["fn"]) == (1, 0, 1)
    assert strict["n_pred_edges"] == 1 and strict["min_match_probability"] == 0.8
    mid = me.compute_confusion_matrix(e, a, None, "dedupe", min_match_probability=0.6)   # >= is inclusive
    assert (mid["tp"], mid["fp"]) == (1, 1)


def test_confusion_matrix_without_ground_truth_is_flagged_unavailable():
    a = pd.DataFrame({"unique_id": ["1", "2"], "source_dataset": "A"})
    cm = me.compute_confusion_matrix(_edges([("1", "2")], [0.9]), a, None, "dedupe")
    assert cm["unavailable"] and cm["precision"] is None


def test_confusion_matrix_empty_predictions():
    a = _nodes(["1", "2"], ["x", "x"])
    cm = me.compute_confusion_matrix(_edges([], []), a, None, "dedupe")
    assert (cm["tp"], cm["fp"], cm["fn"]) == (0, 0, 1)


def test_dedupe_works_when_edges_carry_no_source_columns():
    """Splink's dedupe output has no source_dataset_l/r on some code paths."""
    a = _nodes(["1", "2"], ["x", "x"])
    e = _edges([("1", "2")], [0.9]).drop(columns=["source_dataset_l", "source_dataset_r"])
    assert me.compute_confusion_matrix(e, a, None, "dedupe")["tp"] == 1


# -- threshold curve ---------------------------------------------------------

def test_threshold_curve_hand_computed():
    a = _nodes(["1", "2", "3", "4"], ["x", "x", "y", "y"])               # true pairs: 12, 34
    e = _edges([("1", "2"), ("1", "3"), ("3", "4")], [0.9, 0.6, 0.3])    # TP, FP, TP
    curve = me.compute_threshold_curve(e, a, None, "dedupe")
    assert list(curve["match_probability"]) == [0.9, 0.6, 0.3]
    assert list(curve["tp"]) == [1, 1, 2]
    assert list(curve["fp"]) == [0, 1, 1]
    assert list(curve["fn"]) == [1, 1, 0]
    assert curve["precision"].tolist() == pytest.approx([1.0, 0.5, 2 / 3])
    assert curve["recall"].tolist() == pytest.approx([0.5, 0.5, 1.0])
    assert curve["fstar"].tolist() == pytest.approx([1 / 2, 1 / 3, 2 / 3])


def test_threshold_curve_groups_tied_probabilities_into_one_threshold():
    a = _nodes(["1", "2", "3"], ["x", "x", "y"])
    e = _edges([("1", "2"), ("1", "3")], [0.8, 0.8])
    assert len(me.compute_threshold_curve(e, a, None, "dedupe")) == 1


def test_threshold_curve_empty_without_ground_truth_or_edges():
    a = pd.DataFrame({"unique_id": ["1", "2"], "source_dataset": "A"})
    assert me.compute_threshold_curve(_edges([("1", "2")], [0.9]), a, None, "dedupe").empty
    assert me.compute_threshold_curve(_edges([], []), _nodes(["1"], ["x"]), None, "dedupe").empty


def test_summary_average_precision_hand_computed():
    a = _nodes(["1", "2", "3", "4"], ["x", "x", "y", "y"])
    e = _edges([("1", "2"), ("1", "3"), ("3", "4")], [0.9, 0.6, 0.3])
    summary = me.summarise_threshold_curve(me.compute_threshold_curve(e, a, None, "dedupe"))
    # AP = sum (R_k - R_{k-1}) * P_k = 0.5*1.0 + 0*0.5 + 0.5*(2/3)
    assert summary["average_precision"] == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert summary["best_f1_threshold"] == pytest.approx(0.3)            # f1: .667, .5, .8
    assert summary["best_f1"] == pytest.approx(0.8)


def test_summary_of_nothing_is_empty():
    assert me.summarise_threshold_curve(None) == {}
    assert me.summarise_threshold_curve(pd.DataFrame()) == {}


# -- intra / inter run metrics -----------------------------------------------

def _cluster_table(assignments, source="A"):
    return pd.DataFrame({"unique_id": list(assignments), "cluster_id": list(assignments.values()),
                         "source_dataset": source})


def test_intra_metrics_basic_counts():
    clusters = _cluster_table({"1": "c1", "2": "c1", "3": "c2", "4": "c3"})
    edges = _edges([("1", "2")], [0.9])
    m = me.compute_intra_metrics(edges, clusters)
    assert m["n_edges"] == 1 and m["n_clusters"] == 3 and m["n_unique_ids"] == 2
    assert m["linkage_rate"] == pytest.approx(50.0)
    split = m["singleton_stats"].set_index("cluster_type")
    assert split.loc["Single-record clusters", "n_clusters"] == 2
    assert m["reduction_ratio"] == pytest.approx(100 * (1 - 1 / 6), abs=0.01)


def test_intra_metrics_on_empty_edges_does_not_crash():
    m = me.compute_intra_metrics(_edges([], []), _cluster_table({"1": "c1", "2": "c2"}))
    assert m["n_edges"] == 0 and m["linkage_rate"] == 0.0


def test_gamma_means_ignore_null_level():
    e = _edges([("1", "2"), ("1", "3")], [0.9, 0.9])
    e["gamma_name"] = [2, -1]                                            # -1 = missing value
    metrics = me.compute_intra_metrics(e, _cluster_table({"1": "a", "2": "a", "3": "b"}))
    assert metrics["gamma_means"]["gamma_name"].iloc[0] == 2


def test_venn_counts_by_source():
    clusters = pd.DataFrame({"unique_id": ["1", "2", "3", "4_B", "5_B"], "cluster_id": ["a", "a", "b", "a", "c"],
                             "source_dataset": ["A", "A", "A", "B", "B"]})
    venn = me.compute_intra_metrics(_edges([], []), clusters)["venn"]
    assert venn == {"a_only": 1, "b_only": 1, "both_ab": 1}


def test_inter_metrics_edge_diff_and_cluster_agreement():
    e1 = _edges([("1", "2"), ("3", "4")], [0.9, 0.9])
    e2 = _edges([("2", "1"), ("5", "6")], [0.9, 0.9])                     # (1,2) reversed = shared
    c1 = _cluster_table({"1": "a", "2": "a", "3": "b", "4": "b"})
    c2 = _cluster_table({"1": "x", "2": "x", "3": "y", "4": "z"})
    inter = me.compute_inter_metrics(e1, e2, c1, c2)
    diff = inter["edge_diff"].set_index("category")["n"]
    assert (diff["shared"], diff["added"], diff["removed"]) == (1, 1, 1)
    assert inter["n_exact_matching_clusters"] == 1                       # {1,2}
    assert inter["n_partial_matching_clusters"] == 2                     # b overlaps y and z


def test_inter_metrics_skip_gamma_when_a_run_has_none():
    e1, e2 = _edges([("1", "2")], [0.9]), _edges([("1", "2")], [0.9])
    e1["gamma_a"] = 1
    c = _cluster_table({"1": "a", "2": "a"})
    assert me.compute_inter_metrics(e1, e2, c, c)["gamma_comparison"].empty


def test_f_star_is_monotone_transform_of_f1():
    for tp, fp, fn in [(5, 1, 2), (10, 10, 0), (1, 0, 9)]:
        f1 = 2 * tp / (2 * tp + fp + fn)
        assert tp / (tp + fp + fn) == pytest.approx(f1 / (2 - f1))
        assert not math.isnan(f1)


def test_ground_truth_size_guard_stops_quadratic_blowup(monkeypatch):
    monkeypatch.setattr(me, "MAX_GROUND_TRUTH_PAIRS", 1000)
    big = _nodes([str(i) for i in range(500)], ["same"] * 500)           # 124,750 true pairs
    with pytest.raises(me.GroundTruthTooLarge):
        me.ground_truth_pairs(big, None, "dedupe")
    cm = me.compute_confusion_matrix(_edges([("1", "2")], [0.9]), big, None, "dedupe")
    assert cm["unavailable"] and "ground-truth" in cm["unavailable_reason"]
    assert me.compute_threshold_curve(_edges([("1", "2")], [0.9]), big, None, "dedupe").empty


def test_ground_truth_size_guard_in_link_mode(monkeypatch):
    monkeypatch.setattr(me, "MAX_GROUND_TRUTH_PAIRS", 100)
    a = _nodes([str(i) for i in range(50)], ["x"] * 50, "A")
    b = _nodes([f"{i}_B" for i in range(50)], ["x"] * 50, "B")           # 2,500 A-B pairs
    with pytest.raises(me.GroundTruthTooLarge):
        me.ground_truth_pairs(a, b, "link_dedupe")


def test_ground_truth_guard_leaves_normal_data_alone():
    a = _nodes([str(i) for i in range(1000)], [str(i // 4) for i in range(1000)])   # 250 clusters of 4
    assert len(me.ground_truth_pairs(a, None, "dedupe")) == 250 * 6


def test_venn_circles_really_overlap():
    from modules.report import venn_figure
    circles = [sh for sh in venn_figure(5, 3, 2).layout.shapes if sh.type == "circle"]
    left, right = sorted(circles, key=lambda c: c.x0)
    assert right.x0 < left.x1                        # the circles intersect, not just touch
    assert left.opacity >= 0.5 and left.fillcolor != right.fillcolor
