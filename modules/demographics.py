"""Demographic breakdowns for the cohort-profile and match-quality views.

Everything here is data-driven: a field is only summarised if the column is
actually present in the frame it is given, so the same functions serve the
synthetic voter registry, uploaded CSVs and Advanced-mode runs alike.
"""

import pandas as pd

from modules.cohort_filter import (
    RACE_CODE_LABELS, ETHNIC_CODE_LABELS, PARTY_CD_LABELS, GENDER_CODE_LABELS,
)

# field -> display config. kind: "pie" for low-cardinality codes, "bar" for
# wider categorical (e.g. state), "bar_binned" for numeric ranges that need
# bucketing before they're readable as a chart.
DEMOGRAPHIC_FIELD_CONFIG = {
    "gender_code":     {"label": "Gender",          "value_labels": GENDER_CODE_LABELS, "kind": "pie"},
    "race_code":       {"label": "Race",             "value_labels": RACE_CODE_LABELS,   "kind": "pie"},
    "ethnic_code":     {"label": "Ethnicity",        "value_labels": ETHNIC_CODE_LABELS, "kind": "pie"},
    "party_cd":        {"label": "Registered Party", "value_labels": PARTY_CD_LABELS,    "kind": "pie"},
    "birth_state":     {"label": "Birth State",      "value_labels": None,               "kind": "bar"},
    "birth_year":      {"label": "Decade of Birth",  "value_labels": None,               "kind": "bar_binned",
                         "bins": list(range(1900, 2031, 10)),
                         "bin_labels": [f"{y}s" for y in range(1900, 2021, 10)]},
    "age_at_year_end": {"label": "Age Bracket",      "value_labels": None,               "kind": "bar_binned",
                         "bins": [0, 18, 25, 35, 45, 55, 65, 75, 85, 130],
                         "bin_labels": ["<18", "18-24", "25-34", "35-44", "45-54",
                                        "55-64", "65-74", "75-84", "85+"]},
}


def compute_demographic_breakdowns(df_cluster: pd.DataFrame) -> dict:
    """Compute a distribution DataFrame for every known demographic field
    that is actually present in `df_cluster`.

    Returns {field: {"label": str, "kind": str, "df": DataFrame}} where df
    has columns [value, label, n_records, pct] — 'value' is the raw code,
    'label' is the human-readable version (falls back to the raw code when
    no mapping exists).
    """
    breakdowns = {}
    if df_cluster is None or df_cluster.empty:
        return breakdowns

    for field, cfg in DEMOGRAPHIC_FIELD_CONFIG.items():
        if field not in df_cluster.columns:
            continue

        if cfg["kind"] == "bar_binned":
            numeric = pd.to_numeric(df_cluster[field], errors="coerce").dropna()
            if numeric.empty:
                continue
            binned = pd.cut(numeric, bins=cfg["bins"], labels=cfg["bin_labels"],
                             right=False, include_lowest=True)
            counts = binned.value_counts().reindex(cfg["bin_labels"]).fillna(0).astype(int)
            df = pd.DataFrame({
                "value":     counts.index.astype(str),
                "label":     counts.index.astype(str),
                "n_records": counts.values,
            })
            total = df["n_records"].sum()
            df["pct"] = (100.0 * df["n_records"] / total).round(1) if total else 0.0
            df = df[df["n_records"] > 0].reset_index(drop=True)
        else:
            series = df_cluster[field].dropna()
            series = series[series.astype(str).str.strip() != ""]
            if series.empty:
                continue
            counts = series.value_counts()
            value_labels = cfg.get("value_labels") or {}
            df = pd.DataFrame({
                "value":     counts.index.astype(str),
                "n_records": counts.values,
            })
            df["label"] = df["value"].map(lambda v: value_labels.get(v, v))
            total = df["n_records"].sum()
            df["pct"] = (100.0 * df["n_records"] / total).round(1) if total else 0.0
            df = df.sort_values("n_records", ascending=False).reset_index(drop=True)

        if not df.empty:
            breakdowns[field] = {"label": cfg["label"], "kind": cfg["kind"], "df": df}

    return breakdowns


# Every demographic field the app knows how to chart: the voter-registry
# fields plus the demo dataset's gender/city. Intersected with the fields
# actually compared in a run.
DEMOGRAPHIC_REGISTRY_FIELDS = [*DEMOGRAPHIC_FIELD_CONFIG, "gender", "city"]


def compute_edge_demographic_quality(edges_df: pd.DataFrame, fields: list,
                                      threshold: float = 0.8) -> dict:
    """Edge-level match-quality breakdown, per demographic field, per
    category. Unlike compute_demographic_breakdowns() (population counts
    from df_cluster), this reads the category straight off each edge's
    left-hand value (`<field>_l`) and reports match QUALITY — it is far
    more sensitive to blocking rule toggles, since removing a rule directly
    removes the specific edges it contributed, immediately shifting the
    probability distribution of what's left, even when the raw touched-
    record population barely moves.

    Only fields with a `<field>_l` column in edges_df are included — i.e.
    fields actually selected as comparisons for that run (Splink only
    retains matching columns for compared fields), which is what keeps this
    naturally scoped to "fields selected in the configuration step".

    Returns {field: DataFrame[category, label, n_edges,
             avg_match_probability, pct_above_threshold]}.
    """
    results = {}
    if edges_df is None or edges_df.empty or "match_probability" not in edges_df.columns:
        return results

    for field in fields:
        col_l = f"{field}_l"
        if col_l not in edges_df.columns:
            continue

        cfg  = DEMOGRAPHIC_FIELD_CONFIG.get(field, {})
        kind = cfg.get("kind", "pie")

        sub = edges_df[[col_l, "match_probability"]].copy()

        if kind == "bar_binned":
            numeric = pd.to_numeric(sub[col_l], errors="coerce")
            binned  = pd.cut(numeric, bins=cfg["bins"], labels=cfg["bin_labels"],
                              right=False, include_lowest=True)
            sub["_cat"] = binned.astype(str)
            sub = sub[sub["_cat"] != "nan"]
        else:
            sub["_cat"] = sub[col_l].astype(str).str.strip()
            sub = sub[(sub["_cat"] != "") & (~sub["_cat"].str.lower().isin(["nan", "none"]))]

        if sub.empty:
            continue

        sub["_above"] = sub["match_probability"] >= threshold
        grouped = sub.groupby("_cat").agg(
            n_edges=("match_probability", "count"),
            avg_match_probability=("match_probability", "mean"),
            pct_above_threshold=("_above", "mean"),
        ).reset_index().rename(columns={"_cat": "category"})
        grouped["avg_match_probability"] = grouped["avg_match_probability"].round(3)
        grouped["pct_above_threshold"]   = (grouped["pct_above_threshold"] * 100).round(1)

        value_labels = cfg.get("value_labels") or {}
        grouped["label"] = grouped["category"].map(lambda v: value_labels.get(v, v))
        grouped = grouped.sort_values("n_edges", ascending=False).reset_index(drop=True)
        results[field] = grouped

    return results
