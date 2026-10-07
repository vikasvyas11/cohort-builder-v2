"""Linkage-quality metrics.

Everything in this module is computed with pandas / numpy from the two tables
Splink produces for a run:

* ``df_predict`` – one row per scored candidate pair (``unique_id_l`` /
  ``unique_id_r``, ``match_probability``, ``match_weight``, ``gamma_*`` ...)
* ``df_cluster`` – one row per input record (``unique_id``, ``cluster_id``,
  ``source_dataset`` ...)

Design notes
------------
*Pair identity.*  A pair is identified by a **canonical key**: the two record
keys (``"<source_dataset>|<unique_id>"``) sorted and joined.  Sorting makes the
key independent of which side Splink happened to put a record on, so edge sets
from different runs - and ground-truth pair sets built from a ``cluster``
column - can be compared with a plain ``isin`` / ``merge``.

*Ground truth.*  When the input data carries a ``cluster`` column, two records
are a true match if they share its value.  Recall is always measured against
the **full** set of true pairs, so pairs that blocking never generated count as
misses (they cannot be found by any threshold).

*F\\*.*  ``tp / (tp + fp + fn)`` is the metric proposed by Hand, Christen &
Kirielle (2021), "F*: an interpretable transformation of the F-measure",
Machine Learning 110, 451-456.  It is a monotone transform of F1.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from modules.demographics import compute_demographic_breakdowns

logger = logging.getLogger(__name__)

_KEY_SEP = "\x1f"          # unit separator: cannot appear in ids or dataset names
_GROUND_TRUTH_COLUMN = "cluster"

# Ground-truth pairs grow with the square of each cluster's size, so one huge
# group (e.g. a `cluster` column that is the same everywhere) could ask for
# billions of pairs. Beyond this many, accuracy is reported as unavailable.
MAX_GROUND_TRUTH_PAIRS = int(os.environ.get("COHORT_BUILDER_MAX_TRUTH_PAIRS", "5000000"))


class GroundTruthTooLarge(ValueError):
    """The ground-truth `cluster` column implies more true pairs than can be handled."""


# =============================================================================
# Pair / node keys
# =============================================================================

def _node_keys(unique_id: pd.Series, source: Optional[pd.Series], default_source: str) -> pd.Series:
    """``"<source>|<unique_id>"`` for every record."""
    src = source.astype(str) if source is not None else default_source
    return src + "|" + unique_id.astype(str)


def _edge_pair_keys(edges: pd.DataFrame) -> pd.Series:
    """Canonical (order-independent) pair key for every row of an edge table."""
    left = _node_keys(edges["unique_id_l"], edges.get("source_dataset_l"), "A")
    right = _node_keys(edges["unique_id_r"], edges.get("source_dataset_r"), "A")
    return _join_sorted(left, right)


def _join_sorted(left: pd.Series, right: pd.Series) -> pd.Series:
    """Join two key series into one canonical key, smaller key first."""
    swap = left > right
    first = left.where(~swap, right)
    second = right.where(~swap, left)
    return first + _KEY_SEP + second


def has_ground_truth(dataset_a: pd.DataFrame, dataset_b: Optional[pd.DataFrame] = None) -> bool:
    """True if every supplied dataset carries a ``cluster`` ground-truth column."""
    frames = [dataset_a] + ([dataset_b] if dataset_b is not None else [])
    return all(_GROUND_TRUTH_COLUMN in f.columns for f in frames)


def _check_pair_budget(n_pairs: int) -> None:
    if n_pairs > MAX_GROUND_TRUTH_PAIRS:
        raise GroundTruthTooLarge(
            f"The 'cluster' ground-truth column implies about {n_pairs:,} true pairs, more than the "
            f"{MAX_GROUND_TRUTH_PAIRS:,} this app will compute. Check that 'cluster' labels entities "
            "(records of the same entity share a value) rather than a broad category.")


def ground_truth_pairs(
    dataset_a: pd.DataFrame,
    dataset_b: Optional[pd.DataFrame],
    operation_mode: str,
) -> pd.Series:
    """All true pairs implied by the ``cluster`` column, as canonical keys.

    * ``dedupe``: every pair of records in Dataset A sharing a cluster value.
    * link mode : every A-B pair sharing a cluster value (same-dataset pairs
      are not candidates in a link-only run, so they are not ground truth).
    """
    def nodes(df: pd.DataFrame, default_source: str) -> pd.DataFrame:
        src = df["source_dataset"] if "source_dataset" in df.columns else None
        return pd.DataFrame({
            "key": _node_keys(df["unique_id"], src, default_source),
            "cluster": df[_GROUND_TRUTH_COLUMN].astype(str),
        })

    a = nodes(dataset_a, "A")

    if operation_mode == "dedupe" or dataset_b is None:
        # Singleton clusters contribute no pairs; dropping them first keeps the
        # self-join small on large, mostly-unique registries.
        a = a[a.duplicated("cluster", keep=False)]
        sizes = a.groupby("cluster").size().astype("int64")
        _check_pair_budget(int((sizes * sizes).sum()) // 2)        # the self-join builds size**2 rows
        pairs = a.merge(a, on="cluster", suffixes=("_l", "_r"))
        pairs = pairs[pairs["key_l"] < pairs["key_r"]]
        keys = pairs["key_l"] + _KEY_SEP + pairs["key_r"]
    else:
        b = nodes(dataset_b, "B")
        per_a, per_b = a.groupby("cluster").size(), b.groupby("cluster").size()
        _check_pair_budget(int((per_a * per_b.reindex(per_a.index, fill_value=0)).sum()))
        pairs = a.merge(b, on="cluster", suffixes=("_l", "_r"))
        keys = _join_sorted(pairs["key_l"], pairs["key_r"])

    return keys.drop_duplicates().reset_index(drop=True)


# =============================================================================
# Edge-level (df_predict) summaries
# =============================================================================

def _probability_histogram(edges: pd.DataFrame, width: float = 0.05) -> pd.DataFrame:
    """Edge counts in ``width``-wide match-probability bins."""
    if edges.empty or "match_probability" not in edges.columns:
        return pd.DataFrame(columns=["prob_bin", "n_edges"])
    bins = (np.floor(edges["match_probability"] / width) * width).round(2)
    out = bins.value_counts().sort_index().rename_axis("prob_bin").reset_index(name="n_edges")
    return out


def _weight_histogram(edges: pd.DataFrame) -> pd.DataFrame:
    """Edge counts per match weight rounded to one decimal place."""
    if edges.empty or "match_weight" not in edges.columns:
        return pd.DataFrame(columns=["weight_bin", "n_edges"])
    out = (edges["match_weight"].round(1).value_counts().sort_index()
           .rename_axis("weight_bin").reset_index(name="n_edges"))
    return out


def _probability_summary(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty or "match_probability" not in edges.columns:
        return pd.DataFrame()
    p = edges["match_probability"]
    return pd.DataFrame([{
        "mean_match_prob": round(p.mean(), 4),
        "median_match_prob": round(p.median(), 4),
        "min_match_prob": round(p.min(), 4),
        "max_match_prob": round(p.max(), 4),
        "stddev_match_prob": round(p.std(), 4) if len(p) > 1 else 0.0,
    }])


def _mean_gamma(edges: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
    """Mean agreement level per field. ``-1`` (null comparison) is ignored."""
    cols = columns if columns is not None else [c for c in edges.columns if c.startswith("gamma_")]
    if not cols:
        return pd.DataFrame()
    means = {}
    for col in cols:
        values = edges[col]
        means[col] = round(values[values >= 0].mean(), 4)
    return pd.DataFrame([means])


# =============================================================================
# Cluster-level (df_cluster) summaries
# =============================================================================

def _cluster_sizes(df_cluster: pd.DataFrame) -> pd.Series:
    return df_cluster.groupby("cluster_id").size()


def _size_distribution(df_cluster: pd.DataFrame) -> pd.DataFrame:
    """How many clusters have each size."""
    if df_cluster.empty:
        return pd.DataFrame(columns=["n_nodes", "n_clusters"])
    return (_cluster_sizes(df_cluster).value_counts().sort_index()
            .rename_axis("n_nodes").reset_index(name="n_clusters"))


def _singleton_split(sizes: pd.Series) -> pd.DataFrame:
    is_single = sizes == 1
    rows = []
    for label, mask in (("Multi-record clusters", ~is_single),
                        ("Single-record clusters", is_single)):
        if mask.any():
            rows.append({"cluster_type": label,
                         "n_clusters": int(mask.sum()),
                         "total_records": int(sizes[mask].sum())})
    return pd.DataFrame(rows, columns=["cluster_type", "n_clusters", "total_records"])


def _dataset_membership(df_cluster: pd.DataFrame) -> dict:
    """Cluster counts by which input datasets contribute records to them."""
    empty = {"overlap": pd.DataFrame(), "cross": 0, "venn": {"a_only": 0, "b_only": 0, "both_ab": 0}}
    if df_cluster.empty or "source_dataset" not in df_cluster.columns:
        return empty

    overlap = (df_cluster.groupby("source_dataset")["cluster_id"].nunique()
               .rename("clusters_with_records").reset_index())
    present = (df_cluster.assign(_a=df_cluster["source_dataset"] == "A",
                                 _b=df_cluster["source_dataset"] == "B")
               .groupby("cluster_id")[["_a", "_b"]].any())
    both = present["_a"] & present["_b"]
    sources_per_cluster = df_cluster.groupby("cluster_id")["source_dataset"].nunique()
    return {
        "overlap": overlap,
        "cross": int((sources_per_cluster > 1).sum()),
        "venn": {
            "a_only": int((present["_a"] & ~present["_b"]).sum()),
            "b_only": int((present["_b"] & ~present["_a"]).sum()),
            "both_ab": int(both.sum()),
        },
    }


def _value_frequencies(series: pd.Series, name: str) -> pd.DataFrame:
    """Frequency table ``[name, n_records, pct]`` for a column (nulls ignored)."""
    counts = series.dropna().value_counts()
    if counts.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        name: counts.index,
        "n_records": counts.values,
        "pct": (100.0 * counts.values / counts.sum()).round(1),
    })


def _possible_pairs(df_cluster: pd.DataFrame) -> int:
    """Size of the full comparison space (what blocking is reducing)."""
    n = len(df_cluster)
    if "source_dataset" in df_cluster.columns:
        per_source = df_cluster["source_dataset"].value_counts()
        if len(per_source) == 2:
            return int(per_source.iloc[0] * per_source.iloc[1])
    return n * (n - 1) // 2


# =============================================================================
# Public: single-run metrics
# =============================================================================

def compute_intra_metrics(df_predict: pd.DataFrame, df_cluster: pd.DataFrame) -> dict:
    """Summary statistics for one run (edges, clusters, demographics)."""
    metrics: dict = {}

    # ── Edges ────────────────────────────────────────────────────────────────
    metrics["n_edges"] = int(len(df_predict))
    if df_predict.empty:
        linked_nodes = 0
    else:
        left = _node_keys(df_predict["unique_id_l"], df_predict.get("source_dataset_l"), "A")
        right = _node_keys(df_predict["unique_id_r"], df_predict.get("source_dataset_r"), "A")
        linked_nodes = int(pd.concat([left, right]).nunique())
    metrics["n_unique_ids"] = linked_nodes

    metrics["match_prob_stats"] = _probability_summary(df_predict)
    metrics["weight_dist"] = _weight_histogram(df_predict)
    metrics["prob_dist"] = _probability_histogram(df_predict)
    metrics["gamma_means"] = _mean_gamma(df_predict)

    if "match_weight" in df_predict.columns and not df_predict.empty:
        q = np.percentile(df_predict["match_weight"], [10, 25, 75, 90])
        metrics["weight_percentiles"] = pd.DataFrame(
            [dict(zip(["p10", "p25", "p75", "p90"], np.round(q, 4)))])
    else:
        metrics["weight_percentiles"] = pd.DataFrame()

    # ── Clusters ─────────────────────────────────────────────────────────────
    sizes = _cluster_sizes(df_cluster) if not df_cluster.empty else pd.Series(dtype=int)
    metrics["n_clusters"] = int(len(sizes))
    metrics["cluster_sizes"] = _size_distribution(df_cluster)
    metrics["singleton_stats"] = _singleton_split(sizes)

    membership = _dataset_membership(df_cluster)
    metrics["source_overlap"] = membership["overlap"]
    metrics["n_cross_dataset"] = membership["cross"]
    metrics["venn"] = membership["venn"]

    if sizes.empty:
        metrics["cluster_stats"] = pd.DataFrame()
    else:
        multi = int((sizes > 1).sum())
        metrics["cluster_stats"] = pd.DataFrame([{
            "mean_cluster_size": round(float(sizes.mean()), 2),
            "max_cluster_size": int(sizes.max()),
            "median_cluster_size": float(sizes.median()),
            "multi_member_clusters": multi,
            "pct_multi_member": round(100.0 * multi / len(sizes), 2),
        }])

    n_records = len(df_cluster)
    metrics["linkage_rate"] = round(100.0 * linked_nodes / n_records, 2) if n_records else 0.0
    possible = _possible_pairs(df_cluster)
    metrics["reduction_ratio"] = (
        round(100.0 * (1 - metrics["n_edges"] / possible), 2) if possible else None
    )

    # ── Demographics ─────────────────────────────────────────────────────────
    metrics["gender_dist"] = (_value_frequencies(df_cluster["gender"], "gender")
                              if "gender" in df_cluster.columns else pd.DataFrame())
    metrics["city_dist"] = (_value_frequencies(df_cluster["city"], "city")
                            if "city" in df_cluster.columns else pd.DataFrame())
    metrics["demographics"] = compute_demographic_breakdowns(df_cluster)
    return metrics


# =============================================================================
# Public: comparing two runs
# =============================================================================

def _cluster_agreement(clusters_1: pd.DataFrame, clusters_2: pd.DataFrame) -> tuple[int, int]:
    """(identical clusters, partially overlapping cluster pairs) between two runs.

    Two clusters are *identical* if they contain exactly the same records.  A
    run-1 / run-2 cluster pair is *partial* if it shares records but is not
    identical.
    """
    def keyed(df: pd.DataFrame) -> pd.DataFrame:
        src = df["source_dataset"] if "source_dataset" in df.columns else None
        return pd.DataFrame({"key": _node_keys(df["unique_id"], src, "A"),
                             "cluster": df["cluster_id"].astype(str)})

    one, two = keyed(clusters_1), keyed(clusters_2)
    joined = one.merge(two, on="key", suffixes=("_1", "_2"))
    if joined.empty:
        return 0, 0

    shared = joined.groupby(["cluster_1", "cluster_2"]).size().rename("shared").reset_index()
    shared["size_1"] = shared["cluster_1"].map(one["cluster"].value_counts())
    shared["size_2"] = shared["cluster_2"].map(two["cluster"].value_counts())
    identical = (shared["shared"] == shared["size_1"]) & (shared["shared"] == shared["size_2"])
    return int(identical.sum()), int((~identical).sum())


def compute_inter_metrics(
    df_predict_run1: pd.DataFrame,
    df_predict_run2: pd.DataFrame,
    df_cluster_run1: pd.DataFrame,
    df_cluster_run2: pd.DataFrame,
) -> dict:
    """How Run 2 differs from Run 1 (edges, probabilities, clusters)."""
    # pandas hashing instead of Python sets: far less memory at millions of pairs
    keys_1 = pd.Index(_edge_pair_keys(df_predict_run1)).unique() if not df_predict_run1.empty else pd.Index([])
    keys_2 = pd.Index(_edge_pair_keys(df_predict_run2)).unique() if not df_predict_run2.empty else pd.Index([])
    shared = int(keys_2.isin(keys_1).sum())
    edge_diff = pd.DataFrame({
        "n": [shared, len(keys_2) - shared, len(keys_1) - shared],
        "category": ["shared", "added", "removed"],
    })

    def prob_row(label: str, edges: pd.DataFrame) -> dict:
        p = edges["match_probability"] if "match_probability" in edges.columns else pd.Series(dtype=float)
        return {"run": label,
                "mean_match_prob": round(p.mean(), 4) if len(p) else None,
                "median_match_prob": round(p.median(), 4) if len(p) else None,
                "n_edges": int(len(edges))}

    identical, partial = _cluster_agreement(df_cluster_run1, df_cluster_run2)

    # Deterministic runs have no gamma columns; compare only fields both runs share.
    shared_gamma = sorted({c for c in df_predict_run1.columns if c.startswith("gamma_")}
                          & {c for c in df_predict_run2.columns if c.startswith("gamma_")})
    if shared_gamma:
        gamma = pd.concat([
            _mean_gamma(df_predict_run1, shared_gamma).assign(run="Run 1"),
            _mean_gamma(df_predict_run2, shared_gamma).assign(run="Run 2"),
        ], ignore_index=True)
        gamma = gamma[["run", *shared_gamma]]
    else:
        gamma = pd.DataFrame()

    return {
        "edge_diff": edge_diff,
        "prob_comparison": pd.DataFrame([prob_row("Run 1", df_predict_run1),
                                         prob_row("Run 2", df_predict_run2)]),
        "n_exact_matching_clusters": identical,
        "n_partial_matching_clusters": partial,
        "prob_dist_run1": _probability_histogram(df_predict_run1),
        "prob_dist_run2": _probability_histogram(df_predict_run2),
        "cluster_sizes_run1": _size_distribution(df_cluster_run1),
        "cluster_sizes_run2": _size_distribution(df_cluster_run2),
        "gamma_comparison": gamma,
    }


# =============================================================================
# Public: accuracy against ground truth
# =============================================================================

def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _unavailable(df_predict: pd.DataFrame) -> dict:
    return {
        "tp": None, "fp": None, "fn": None, "n_gt_edges": None,
        "n_pred_edges": len(df_predict),
        "precision": None, "recall": None, "f1": None,
        "fdr": None, "fnr": None, "fstar": None,
        "unavailable": True,
        "unavailable_reason": (
            "Accuracy needs a ground-truth column named 'cluster' (records with the "
            "same value are the same real-world entity). It was not found in the "
            "input data. For uploaded data, ground truth is created automatically "
            "when Dataset B is generated from Dataset A; a separately uploaded "
            "Dataset B has none."
        ),
    }


def compute_confusion_matrix(
    df_predict: pd.DataFrame,
    dataset_a: pd.DataFrame,
    dataset_b: Optional[pd.DataFrame],
    operation_mode: str,
    min_match_probability: float = 0.0,
) -> dict:
    """Pair-level confusion matrix at one operating point.

    A pair counts as *predicted* when it is in ``df_predict`` with
    ``match_probability >= min_match_probability``.  Pass the run's cluster
    threshold to describe the cohort the run actually produces; the default
    (0) counts every scored pair.  ``compute_threshold_curve`` shows all
    thresholds at once.

    ``tp``  predicted pair that is a true pair
    ``fp``  predicted pair that is not a true pair
    ``fn``  true pair that was not predicted (including pairs blocking never
            generated)
    True negatives are omitted: they are every other possible pair and carry
    no information at this scale.
    """
    if not has_ground_truth(dataset_a, dataset_b):
        return _unavailable(df_predict)

    try:
        truth = ground_truth_pairs(dataset_a, dataset_b, operation_mode)
        edges = df_predict
        if min_match_probability > 0 and "match_probability" in edges.columns:
            edges = edges[edges["match_probability"] >= min_match_probability]
        predicted = _edge_pair_keys(edges).drop_duplicates() if len(edges) else pd.Series(dtype=str)
        tp = int(predicted.isin(truth).sum())
        fp = int(len(predicted) - tp)
        fn = int(len(truth) - tp)

        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        return {
            "tp": tp, "fp": fp, "fn": fn,
            "n_gt_edges": int(len(truth)),
            "n_pred_edges": int(len(edges)),
            "min_match_probability": min_match_probability,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(_ratio(2 * tp, 2 * tp + fp + fn), 4),
            "fdr": round(1.0 - precision, 4),
            "fnr": round(1.0 - recall, 4),
            "fstar": round(_ratio(tp, tp + fp + fn), 4),
        }
    except GroundTruthTooLarge as exc:
        return {**_unavailable(df_predict), "unavailable_reason": str(exc)}
    except Exception as exc:  # the UI shows this message instead of crashing
        logger.exception("confusion matrix failed")
        return {**_unavailable(df_predict), "unavailable": False,
                "unavailable_reason": None, "error": str(exc)}


def compute_threshold_curve(
    df_predict: pd.DataFrame,
    dataset_a: pd.DataFrame,
    dataset_b: Optional[pd.DataFrame],
    operation_mode: str,
) -> pd.DataFrame:
    """Precision / recall / F* at every distinct ``match_probability`` threshold.

    A pair is predicted a match at threshold ``t`` when its probability is
    ``>= t``.  Returns one row per distinct probability (highest first) with
    ``tp, fp, fn, precision, recall, fdr, fnr, fstar, f1``.  Empty if
    the data has no ground truth or there are no scored pairs.
    """
    if (df_predict.empty or "match_probability" not in df_predict.columns
            or not has_ground_truth(dataset_a, dataset_b)):
        return pd.DataFrame()

    try:
        truth = ground_truth_pairs(dataset_a, dataset_b, operation_mode)
        scored = pd.DataFrame({"pair": _edge_pair_keys(df_predict),
                               "p": df_predict["match_probability"].to_numpy()})
        scored = (scored.sort_values("p", ascending=False)
                  .drop_duplicates("pair"))                      # keep each pair's best score
        is_true = scored["pair"].isin(truth).to_numpy()
        prob = scored["p"].to_numpy()

        tp_cum = np.cumsum(is_true)
        fp_cum = np.cumsum(~is_true)
        last_of_group = np.append(prob[1:] != prob[:-1], True)   # one row per distinct threshold
        tp = tp_cum[last_of_group].astype(float)
        fp = fp_cum[last_of_group].astype(float)
        fn = len(truth) - tp

        with np.errstate(divide="ignore", invalid="ignore"):
            precision = tp / (tp + fp)
            recall = tp / len(truth) if len(truth) else np.full_like(tp, np.nan)
            fstar = tp / (tp + fp + fn)
            f1 = 2 * tp / (2 * tp + fp + fn)
        return pd.DataFrame({
            "match_probability": prob[last_of_group],
            "tp": tp.astype(int), "fp": fp.astype(int), "fn": fn.astype(int),
            "precision": precision, "recall": recall,
            "fdr": 1.0 - precision, "fnr": 1.0 - recall,
            "fstar": fstar, "f1": f1,
        })
    except GroundTruthTooLarge:
        return pd.DataFrame()
    except Exception:
        logger.exception("threshold curve failed")
        return pd.DataFrame()


def summarise_threshold_curve(curve: Optional[pd.DataFrame]) -> dict:
    """Headline numbers from :func:`compute_threshold_curve`.

    * ``average_precision`` - area under the step-wise precision-recall curve
    * ``best_f1`` / ``best_f1_threshold`` - best F1 and the cut-off that gives it
    * ``best_fstar`` / ``best_fstar_threshold`` - the same for F*

    Returns ``{}`` when there is no curve.
    """
    if curve is None or curve.empty:
        return {}
    usable = curve.dropna(subset=["precision", "recall"])
    if usable.empty:
        return {}

    recall = usable["recall"].to_numpy()
    precision = usable["precision"].to_numpy()
    recall_gain = np.diff(np.concatenate([[0.0], recall]))
    best_f1 = usable.loc[usable["f1"].idxmax()]
    best_fstar = usable.loc[usable["fstar"].idxmax()]
    return {
        "average_precision": float(np.sum(recall_gain * precision)),
        "best_f1": float(best_f1["f1"]),
        "best_f1_threshold": float(best_f1["match_probability"]),
        "best_fstar": float(best_fstar["fstar"]),
        "best_fstar_threshold": float(best_fstar["match_probability"]),
        "n_thresholds": int(len(usable)),
    }
