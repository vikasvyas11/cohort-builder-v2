"""The app's logic, free of any web framework: load data, run linkage, summarise results.

Each function takes a :class:`~api.store.Session` and returns plain Python / JSON-ready values,
so the same code serves the HTTP routes and the tests. The pages of the old Streamlit app map
onto these functions one to one: profile, configure, run, the seven result tabs, compare, export.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px

from api import figures as fig
from api.store import Run, Session, UserError
from modules.cohort_filter import (
    TIER1_CATEGORICAL_FIELDS, TIER1_RANGE_FIELDS, apply_cohort_filters, cohort_summary, default_filters,
)
from modules.corruption import CorruptionSettings, make_noisy_copy
from modules.data_builder import build_datasets, load_voter_datasets
from modules.demographics import (
    DEMOGRAPHIC_REGISTRY_FIELDS, compute_demographic_breakdowns, compute_edge_demographic_quality,
)
from modules.eda_engine import (
    find_high_correlation_pairs, infer_field_types, is_selective_field, run_full_eda,
    suggest_blocking_rules, suggest_comparison_types,
)
from modules.metrics_engine import (
    compute_confusion_matrix, compute_inter_metrics, compute_intra_metrics, compute_threshold_curve,
    summarise_threshold_curve,
)
from modules.report import confusion_figure, generate_report, venn_figure
from modules.splink_runner import (
    DEFAULT_CLUSTER_THRESHOLD, MAX_CANDIDATE_PAIRS, _get_or_build_covers_column, blocking_rule_patterns,
    build_coverage_matrix, candidate_pairs_by_rule, reconstruct_model_json, recluster_filtered, run_linkage,
    run_linkage_from_json,
)
from modules.voter_data import LINKAGE_FIELDS, blocking_defaults

COMPARISON_TYPES = ["NameComparison", "DateOfBirthComparison", "ExactMatch", "LevenshteinAtThresholds",
                    "JaroWinklerAtThresholds", "EmailComparison", "PostcodeComparison"]
META_COLUMNS = {"unique_id", "cluster", "source_dataset"}
PEOPLE_FIELDS = ["first_name", "surname", "dob", "city", "email", "gender", "postcode"]
LOW_INFORMATION_FIELDS = {"city", "gender"}
VOTER_SIZES = (1_200, 10_000, 50_000)
WARN_PAIRS = 500_000


# ── helpers ───────────────────────────────────────────────────────────────────

def records(df: Optional[pd.DataFrame], limit: Optional[int] = None) -> list:
    """A DataFrame as a list of JSON-safe dicts (NaN and numpy types handled by pandas)."""
    if df is None or df.empty:
        return []
    return json.loads((df.head(limit) if limit else df).to_json(orient="records", date_format="iso"))


def jsonable(value):
    """Recursively turn numpy scalars and NaN into plain JSON values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return records(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if value != value else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _run(session: Session, slot: str) -> Run:
    run = session.runs.get(slot)
    if run is None:
        raise UserError(f"There are no {slot.replace('run', 'Run ')} results yet.")
    return run


def has_ground_truth(df: Optional[pd.DataFrame]) -> bool:
    return df is not None and "cluster" in df.columns


# ── datasets ──────────────────────────────────────────────────────────────────

def _defaults(session: Session) -> dict:
    """Default comparison fields, blocking rules and comparison types for the loaded data."""
    cols = [c for c in session.a.columns if c not in META_COLUMNS]
    if session.source == "voters":
        fields = [f for f in LINKAGE_FIELDS if f in cols]
        on = blocking_defaults(len(session.a))
        blocking = {f: f in on for f in fields} | {rule: True for rule in on if "+" in rule}
    elif session.source == "people":
        fields = [f for f in PEOPLE_FIELDS if f in cols]
        blocking = {f: f not in LOW_INFORMATION_FIELDS for f in fields}
    else:
        fields = [c for c in cols if session.field_types.get(c) != "id"]
        suggested = suggest_blocking_rules(session.field_types)
        blocking = {f: bool(suggested.get(f)) for f in fields}
        if not any(blocking.values()):
            blocking = {f: is_selective_field(session.a[f]) for f in fields}
    comp = suggest_comparison_types(session.field_types) if session.source == "upload" else {}
    return {"fields": fields, "blocking_toggles": blocking, "comp_types": comp}


def overview(session: Session) -> dict:
    """What the profile and configure steps need to know about the loaded data."""
    a, b = session.a, session.b
    missing = (a.isna() | (a.astype(str).apply(lambda s: s.str.strip()) == "")).mean().round(4)
    snapshot = demographic_snapshot(a)
    return {
        "session_id": session.id, "source": session.source, "runs": sorted(session.runs),
        "rows_a": len(a), "rows_b": None if b is None else len(b), "columns": a.shape[1],
        "has_ground_truth": has_ground_truth(a) and (b is None or has_ground_truth(b)),
        "column_info": [{"column": c, "type": session.field_types.get(c, "text"),
                         "missing_pct": float(missing.get(c, 0) * 100),
                         "distinct": int(a[c].nunique())} for c in a.columns],
        "preview_a": records(a, 8), "preview_b": records(b, 5),
        "demographics": fig.breakdown_figures(snapshot),
        "defaults": _defaults(session),
        "comparison_types": COMPARISON_TYPES,
        "eda": session.eda,
        "cohort_filter": cohort_schema(session),
        "limits": {"max_candidate_pairs": MAX_CANDIDATE_PAIRS, "warn_pairs": WARN_PAIRS},
    }


def load_demo(source: str, n_rows: int) -> Session:
    from api.store import create_session
    if source == "people":
        _, a, b = build_datasets()
        return create_session(source="people", a=a, b=b, field_types=infer_field_types(a.columns))
    if source == "voters":
        if n_rows not in VOTER_SIZES:
            raise UserError(f"Choose a voter dataset size from {', '.join(f'{n:,}' for n in VOTER_SIZES)}.")
        a, b, types, log = load_voter_datasets(n_rows)
        return create_session(source="voters", a=a, b=b, field_types=types, eda=_eda_payload(log, None))
    raise UserError("Unknown dataset.")


def _ensure_unique_id(df: pd.DataFrame, id_col: Optional[str], prefix: str) -> pd.DataFrame:
    df = df.copy()
    if id_col and id_col in df.columns and id_col != "unique_id":
        df = df.rename(columns={id_col: "unique_id"})
    elif "unique_id" not in df.columns:
        df.insert(0, "unique_id", prefix + "_" + pd.Series(range(len(df))).astype(str))
    df["unique_id"] = df["unique_id"].astype(str)        # an int id is a known Splink gotcha
    return df


def _eda_payload(log: dict, corr: Optional[list]) -> dict:
    return {"log": jsonable(log), "high_correlation": [{"a": a, "b": b, "agreement": s} for a, b, s in (corr or [])]}


def load_upload(raw_a: pd.DataFrame, raw_b: Optional[pd.DataFrame], id_col_a: Optional[str],
                id_col_b: Optional[str], mode: str) -> Session:
    """Clean uploaded data and give it the ids and ground truth the linkage needs.

    ``mode``: ``dedupe_only`` (no Dataset B, so no ground truth), ``uploaded`` (B supplied) or
    ``sample`` (B is generated later from A with ``generate_b``).
    """
    from api.store import create_session
    a, types, _, log = run_full_eda(raw_a.copy(), id_col=id_col_a)
    corr = find_high_correlation_pairs(a, [c for c, t in types.items() if t == "id"])
    a = _ensure_unique_id(a, id_col_a if id_col_a in a.columns else None, "A")
    a["source_dataset"] = "A"
    if mode == "sample":
        a["cluster"] = a.index.astype(str)              # B copies this, so the link has a ground truth
    elif mode == "uploaded" and "cluster" not in a.columns:
        a["cluster"] = a["unique_id"].astype(str)
    b = None
    if mode == "uploaded":
        if raw_b is None:
            raise UserError("Upload Dataset B, or choose another way to set it up.")
        b, _, _, log_b = run_full_eda(raw_b.copy(), id_col=id_col_b)
        b = _ensure_unique_id(b, id_col_b if id_col_b in b.columns else None, "B")
        b["source_dataset"] = "B"
        if "cluster" not in b.columns:
            b["cluster"] = b["unique_id"].str.replace(r"_B$", "", regex=True)
        if not b["unique_id"].str.endswith("_B").all():
            b["unique_id"] = b["unique_id"] + "_B"
        log = {**log, "dataset_b": jsonable(log_b)}
    types = infer_field_types(a.columns, "unique_id")
    return create_session(source="upload", a=a, b=b, field_types=types, eda=_eda_payload(log, corr))


def generate_b(session: Session, sample_frac: float, rates: dict, letters: int, year_shift: int) -> dict:
    """Build Dataset B as a damaged sample of Dataset A (so every B record has a true match)."""
    eligible = [c for c in session.a.columns if c not in META_COLUMNS and session.field_types.get(c) != "id"]
    field_rates = {f: float(r) for f, r in rates.items() if f in eligible and float(r) > 0}
    a = session.a if "cluster" in session.a.columns else session.a.assign(cluster=session.a.index.astype(str))
    b = make_noisy_copy(a, field_rates=field_rates, sample_frac=sample_frac, seed=42,
                        field_types=session.field_types,
                        settings=CorruptionSettings(text_edits=letters, year_shift=year_shift),
                        id_columns=[c for c, t in session.field_types.items() if t == "id"] or ["unique_id"])
    if "source_dataset" in b.columns:
        b["source_dataset"] = "B"
    session.a, session.full_a, session.b = a, a, b
    session.runs.clear()
    return {"rows_b": len(b), "damaged_fields": sorted(field_rates)}


# ── cohort filter (voter-shaped data) ─────────────────────────────────────────

def cohort_schema(session: Session) -> Optional[dict]:
    """The filters the data supports, or None when it has no demographic columns."""
    a = session.full_a
    present = [f for f in TIER1_CATEGORICAL_FIELDS if f in a.columns]
    ranges = [f for f in TIER1_RANGE_FIELDS if f in a.columns]
    if not present and not ranges:
        return None
    defaults = default_filters(a)
    return {
        "categorical": [{"field": f, "options": [{"value": v, "label": (TIER1_CATEGORICAL_FIELDS[f] or {}).get(v, v)}
                                                 for v in sorted(a[f].dropna().astype(str).unique())]}
                        for f in present],
        "ranges": [{"field": f, "bounds": defaults["ranges"][f]} for f in ranges],
        "rows_unfiltered": len(a),
    }


def apply_cohort(session: Session, filters: dict) -> dict:
    """Restrict Dataset A to a demographic cohort (an empty filter restores the full dataset)."""
    session.a = apply_cohort_filters(session.full_a, filters)
    if session.a.empty:
        session.a = session.full_a
        raise UserError("That cohort contains no records; the filter was not applied.")
    session.runs.clear()
    session.cache.clear()
    return {"rows_a": len(session.a), "summary": cohort_summary(filters)}


# ── demographics ──────────────────────────────────────────────────────────────

def demographic_snapshot(df: pd.DataFrame) -> dict:
    """{field: {label, kind, df}} for any record table: voter demographics plus gender and city."""
    snapshot = compute_demographic_breakdowns(df)
    if df is None or df.empty:
        return snapshot
    for col, label, kind in (("gender", "Gender", "pie"), ("city", "City", "bar")):
        if col in df.columns and col not in snapshot:
            series = df[col].dropna()
            series = series[series.astype(str).str.strip() != ""]
            if not series.empty:
                counts = series.value_counts()
                out = pd.DataFrame({"value": counts.index.astype(str), "n_records": counts.values})
                out["label"] = out["value"]
                out["pct"] = (100.0 * out["n_records"] / out["n_records"].sum()).round(1)
                snapshot[col] = {"label": label, "kind": kind, "df": out}
    return snapshot


def _linked_vs_unlinked(df_cluster: pd.DataFrame) -> dict:
    if df_cluster is None or df_cluster.empty or "cluster_id" not in df_cluster.columns:
        return {"fields": [], "summary": []}
    sizes = df_cluster.groupby("cluster_id")["cluster_id"].transform("size")
    linked, unlinked = demographic_snapshot(df_cluster[sizes > 1]), demographic_snapshot(df_cluster[sizes == 1])
    common = sorted((f for f in linked if f in unlinked), key=lambda f: (f not in ("gender", "gender_code"), f))[:2]
    fields, summary = [], []
    for f in common:
        base, cur = linked[f], unlinked[f]
        fields.append({"field": f, "label": base["label"],
                       "linked": fig.breakdown_figures({f: base}, 280)[0]["figure"],
                       "unlinked": fig.breakdown_figures({f: cur}, 280)[0]["figure"]})
        b_pct = dict(zip(base["df"]["label"], base["df"]["pct"]))
        c_pct = dict(zip(cur["df"]["label"], cur["df"]["pct"]))
        shared = set(b_pct) & set(c_pct)
        if shared:
            top = max(shared, key=lambda k: abs(c_pct[k] - b_pct[k]))
            delta = c_pct[top] - b_pct[top]
            summary.append(f"{base['label']}: '{top}' is {abs(delta):.1f} points "
                           f"{'higher' if delta > 0 else 'lower'} among unlinked records "
                           f"({b_pct[top]:.1f}% linked vs {c_pct[top]:.1f}% unlinked).")
    return {"fields": fields, "summary": summary,
            "n_linked": int((sizes > 1).sum()), "n_unlinked": int((sizes == 1).sum())}


def run_demographics(session: Session, slot: str) -> dict:
    run = _run(session, slot)
    m = run.metrics
    legacy = []
    if not m["gender_dist"].empty:
        legacy.append(fig.pie(m["gender_dist"], "n_records", "gender", "Gender Distribution in Clusters"))
    if not m["city_dist"].empty:
        legacy.append(fig.bar(m["city_dist"].head(10), "city", "n_records", "Top 10 Cities in Clusters", fig.PURPLE))
    return {"legacy": legacy, "breakdowns": fig.breakdown_figures(m.get("demographics") or {}),
            "linked_vs_unlinked": _linked_vs_unlinked(run.results["df_cluster"])}


# ── running a linkage ─────────────────────────────────────────────────────────

def clean_config(cfg: dict) -> dict:
    """Validate and normalise a run configuration from the client."""
    hp = cfg.get("hyperparams") or {}
    out = {
        "fields": [str(f) for f in cfg.get("fields", [])],
        "blocking_toggles": {str(k): bool(v) for k, v in (cfg.get("blocking_toggles") or {}).items()},
        "blocking_mode": "AND" if cfg.get("blocking_mode") == "AND" else "OR",
        "operation_mode": "link_dedupe" if cfg.get("operation_mode") == "link_dedupe" else "dedupe",
        "linkage_type": "deterministic" if cfg.get("linkage_type") == "deterministic" else "probabilistic",
        "cluster_threshold": min(0.99, max(0.5, float(cfg.get("cluster_threshold") or DEFAULT_CLUSTER_THRESHOLD))),
        "hyperparams": {
            "max_iterations": int(min(500, max(5, hp.get("max_iterations", 25)))),
            "em_convergence": float(min(0.01, max(1e-8, hp.get("em_convergence", 0.0001)))),
            "recall_estimate": float(min(0.99, max(0.1, hp.get("recall_estimate", 0.6)))),
        },
        "comp_types": {str(k): str(v) for k, v in (cfg.get("comp_types") or {}).items() if v in COMPARISON_TYPES} or None,
        "from_model": bool(cfg.get("from_model")),
    }
    return out


def _validated(session: Session, cfg: dict) -> tuple:
    """(fields, blocking_toggles) restricted to columns every needed dataset has; raises if nothing is left."""
    cols = set(session.a.columns)
    if cfg["operation_mode"] == "link_dedupe":
        if session.b is None or session.b.empty:
            raise UserError("Linking needs Dataset B. Add one, or choose deduplication only.")
        cols &= set(session.b.columns)
    fields = [f for f in cfg["fields"] if f in cols]
    if not fields:
        raise UserError("None of the selected fields exist in the loaded data. "
                        f"Available columns: {sorted(cols - META_COLUMNS)}")
    blocking = {k: bool(on) and all(p.strip() in cols for p in k.split("+")) for k, on in cfg["blocking_toggles"].items()}
    if not any(blocking.values()):
        raise UserError("Switch on at least one blocking rule that uses a column present in the data.")
    return fields, blocking


def estimate(session: Session, cfg: dict) -> dict:
    """Pre-flight pair count for a blocking configuration, with a go/warn/refuse verdict."""
    cfg = clean_config(cfg)
    _, blocking = _validated(session, cfg)
    b = session.b if cfg["operation_mode"] == "link_dedupe" else None
    by_rule = candidate_pairs_by_rule(session.a, b, blocking, cfg["blocking_mode"], cfg["operation_mode"])
    pairs = sum(n for _, n in by_rule)
    level = "refuse" if pairs > MAX_CANDIDATE_PAIRS else "warn" if pairs > WARN_PAIRS else "ok"
    return {"pairs": pairs, "level": level, "limit": MAX_CANDIDATE_PAIRS, "warn_at": WARN_PAIRS,
            "by_rule": [{"rule": rule, "pairs": n} for rule, n in by_rule[:6]]}


def execute_run(session: Session, slot: str, raw_cfg: dict, stage) -> None:
    """Run one linkage and store everything the result tabs need. Raises UserError for fixable problems."""
    cfg = clean_config(raw_cfg)
    a = session.a
    b = session.b if cfg["operation_mode"] == "link_dedupe" else None
    mode, threshold = cfg["operation_mode"], cfg["cluster_threshold"]

    if cfg["from_model"]:
        if session.model_json is None:
            raise UserError("Upload a model JSON first.")
        if mode == "link_dedupe" and b is None:
            raise UserError("Linking needs Dataset B. Add one, or choose deduplication only.")
        stage("Predicting with the uploaded model")
        results = run_linkage_from_json(session.model_json, a, b, mode, threshold, linkage_type=cfg["linkage_type"])
        fields = results["run_config"]["selected_fields"]
    else:
        fields, blocking = _validated(session, cfg)
        stage("Counting candidate pairs")
        by_rule = candidate_pairs_by_rule(a, b, blocking, cfg["blocking_mode"], mode)
        pairs = sum(n for _, n in by_rule)
        if pairs > MAX_CANDIDATE_PAIRS:
            biggest = ", ".join(f"{rule} ({n:,})" for rule, n in by_rule[:3])
            raise UserError(
                f"This blocking configuration would create about {pairs:,} candidate pairs, more than the "
                f"{MAX_CANDIDATE_PAIRS:,} this app will score. The biggest rules: {biggest}. Switch off low-selectivity "
                "rules (fields with few distinct values such as gender or city), use AND blocking, or block on a "
                "combination of fields. A host with spare memory can raise the limit with the "
                "COHORT_BUILDER_MAX_CANDIDATE_PAIRS setting.")
        stage("Training the model and scoring pairs" if cfg["linkage_type"] == "probabilistic"
              else "Applying the exact-match rules")
        comp = {f: t for f, t in (cfg["comp_types"] or {}).items() if f in fields} or None
        results = run_linkage(dataset_a=a, dataset_b=b, selected_fields=fields, blocking_toggles=blocking,
                              operation_mode=mode, linkage_type=cfg["linkage_type"], cluster_threshold=threshold,
                              hyperparams=cfg["hyperparams"], comp_types=comp, blocking_mode=cfg["blocking_mode"])

    stage("Computing metrics")
    df_predict, df_cluster = results["df_predict"], results["df_cluster"]
    metrics = compute_intra_metrics(df_predict, df_cluster)
    cm = compute_confusion_matrix(df_predict, a, b, mode, threshold)
    if cfg["linkage_type"] == "probabilistic" or cfg["from_model"]:
        stage("Building the precision-recall curve")
        curve_df = compute_threshold_curve(df_predict, a, b, mode)
        curve = summarise_threshold_curve(curve_df)
    else:
        curve_df, curve = None, {}
    stage("Preparing the blocking explorer")
    coverage = build_coverage_matrix(df_predict, fields)
    session.runs[slot] = Run(results, metrics, cm, curve_df, curve, coverage)
    if slot == "run1":
        session.runs.pop("run2", None)                   # a new Run 1 invalidates the old comparison
    session.cache.pop("waterfall_patterns", None)


# ── result tabs ───────────────────────────────────────────────────────────────

def _confusion_view(run: Run) -> dict:
    cm = run.cm or {}
    if not cm or cm.get("unavailable"):
        return {"available": False, "reason": cm.get("unavailable_reason", "Not available.")}
    if "error" in cm:
        return {"available": False, "reason": f"Confusion matrix error: {cm['error']}"}
    return {
        "available": True, "cm": jsonable(cm), "figure": fig.to_json(confusion_figure(cm)),
        "curve_summary": jsonable(run.curve), "curve_figures": fig.threshold_figures(run.threshold_curve),
        "derived": [{"metric": "Precision", "value": cm.get("precision"), "meaning": "TP / (TP+FP)"},
                    {"metric": "Recall", "value": cm.get("recall"), "meaning": "TP / (TP+FN)"},
                    {"metric": "F1 Score", "value": cm.get("f1"), "meaning": "Harmonic mean"},
                    {"metric": "F* Score", "value": cm.get("fstar"), "meaning": "TP / (TP+FP+FN)"},
                    {"metric": "FDR", "value": cm.get("fdr"), "meaning": "False Discovery Rate"},
                    {"metric": "FNR", "value": cm.get("fnr"), "meaning": "False Negative Rate"}],
    }


def run_summary(session: Session, slot: str) -> dict:
    """Summary cards and the edge, cluster and accuracy tabs for one run."""
    run = _run(session, slot)
    r, m = run.results, run.metrics
    cfg = r["run_config"]
    prob = cfg["linkage_type"] == "probabilistic"
    edges = {"prob_stats": records(m["match_prob_stats"]), "weight_percentiles": records(m["weight_percentiles"]),
             "charts": []}
    if len(m["prob_dist"]) > 1:
        edges["charts"].append({"figure": fig.bar(m["prob_dist"], "prob_bin", "n_edges", "Match Probability Distribution"),
                                "caption": "Bars near 1.0 are confident predictions; bars spread across the middle are uncertain."})
    if len(m["weight_dist"]) > 1:
        edges["charts"].append({"figure": fig.bar(m["weight_dist"], "weight_bin", "n_edges", "Match Weight Histogram", fig.ORANGE),
                                "caption": "Match weight = log2(m/u). Positive values mean a match is more likely."})
    if not m["gamma_means"].empty and prob:
        g = m["gamma_means"].T.reset_index()
        g.columns = ["field", "mean_gamma"]
        g["field"] = g["field"].str.replace("gamma_", "", regex=False)
        edges["charts"].append({"figure": fig.bar(g, "field", "mean_gamma", "Mean Gamma Score per Field", fig.GREEN),
                                "caption": "Gamma 1 is exact agreement, 0 total disagreement."})
    venn = m.get("venn", {})
    clusters = {"n_clusters": m["n_clusters"], "n_cross_dataset": m["n_cross_dataset"],
                "singleton": records(m["singleton_stats"]), "stats": records(m["cluster_stats"]),
                "singleton_figure": fig.singleton_chart(m["singleton_stats"]) if not m["singleton_stats"].empty else None,
                "size_figure": (fig.bar(m["cluster_sizes"], "n_nodes", "n_clusters", "Cluster Size Distribution")
                                if not m["cluster_sizes"].empty else None),
                "venn": None}
    if cfg["operation_mode"] != "dedupe" and any(venn.values()):
        clusters["venn"] = {"rows": [{"category": "Dataset A only", "n": venn.get("a_only", 0)},
                                     {"category": "Both A and B", "n": venn.get("both_ab", 0)},
                                     {"category": "Dataset B only", "n": venn.get("b_only", 0)}],
                            "figure": fig.to_json(venn_figure(venn.get("a_only", 0), venn.get("both_ab", 0), venn.get("b_only", 0)))}
    return {
        "config": jsonable(cfg),
        "cards": {"records": r["n_input_records"], "edges": m["n_edges"], "clusters": m["n_clusters"],
                  "unique_ids": m["n_unique_ids"], "linkage_rate": m["linkage_rate"],
                  "reduction_ratio": m["reduction_ratio"]},
        "zero_edges": r["n_edges"] == 0, "zero_edge_diagnostic": jsonable(r.get("zero_edge_diagnostic") or []),
        "edges": edges, "clusters": clusters, "accuracy": _confusion_view(run),
        "rules": [{"rule": k, "on": bool(v)} for k, v in cfg["blocking_toggles"].items()],
    }


def raw_tables(session: Session, slot: str, limit: int = 100) -> dict:
    r = _run(session, slot).results
    return {"predict": records(r["df_predict"], limit), "cluster": records(r["df_cluster"], limit),
            "n_predict": len(r["df_predict"]), "n_cluster": len(r["df_cluster"])}


def studio_html(session: Session, slot: str) -> str:
    return _run(session, slot).results.get("cluster_html") or ""


# ── blocking explorer ─────────────────────────────────────────────────────────

def _active_mask(coverage: pd.DataFrame, toggles: dict) -> pd.Series:
    """Pairs covered by at least one active rule (composite 'a+b' rules need every part to agree)."""
    mask = pd.Series(False, index=coverage.index)
    for rule, on in toggles.items():
        if on:
            covers = _get_or_build_covers_column(coverage, rule)
            if covers is not None:
                mask |= covers
    return mask


def _filtered_edges(run: Run, toggles: dict) -> tuple:
    mask = _active_mask(run.coverage, toggles).to_numpy()
    df = run.results["df_predict"]
    if len(mask) != len(df):
        raise UserError("The explorer is out of sync with this run; run the analysis again.")
    return df[mask], mask


def explorer(session: Session, slot: str, toggles: Optional[dict], threshold: float) -> dict:
    """Blocking explorer for one run: live waterfall, rule cards, edge table and match quality per group."""
    run = _run(session, slot)
    r = run.results
    cfg = r["run_config"]
    # Only the rules this run used: it generated no pairs for any other, so switching one on adds nothing.
    base = {k: True for k, v in cfg["blocking_toggles"].items() if v}
    toggles = {k: bool(toggles.get(k, True)) for k in base} if toggles else dict(base)
    counts = {c["rule_sql"]: c["n"] for c in r.get("blocking_counts", [])}
    rules = [{"rule": k, "on": toggles[k],
              "pairs": counts.get(" AND ".join(f'l."{p}" = r."{p}"' for p in k.split("+")), 0)} for k in toggles]
    out = {"rules": rules, "and_mode": cfg.get("blocking_mode") == "AND", "waterfall": None}
    if cfg.get("blocking_mode") != "AND":
        out["waterfall"] = fig.waterfall(run.coverage, toggles, ("Original rules", "Toggled rules (live)"),
                                         baseline_toggles=base)
    filtered, mask = _filtered_edges(run, toggles)
    n_orig, n_filtered = len(r["df_predict"]), len(filtered)
    out["stats"] = {"candidate_pairs": n_filtered, "original_pairs": n_orig,
                    "rules_on": sum(toggles.values()), "rules_total": len(toggles),
                    "reduction_pct": round((1 - n_filtered / n_orig) * 100, 1) if n_orig else 0.0}
    if n_filtered:
        sample = filtered.head(200)
        first_active = [k for k, on in toggles.items() if on]
        eff = pd.Series("", index=sample.index, dtype=object)
        cov_sample = run.coverage[mask][:200]
        for rule in reversed(first_active):
            covers = _get_or_build_covers_column(cov_sample, rule)
            if covers is not None:
                eff = eff.mask(covers.to_numpy(), rule)
        sample = sample.assign(effective_rule=eff.to_numpy())
        ids = [c for c in ("unique_id_l", "unique_id_r", "source_dataset_l", "source_dataset_r") if c in sample.columns]
        gamma = [c for c in sample.columns if c.startswith("gamma_")][:4]
        score = [c for c in ("match_probability", "match_weight") if c in sample.columns]
        out["table"] = records(sample[ids + ["effective_rule"] + score + gamma])
    else:
        out["table"] = []
    out["match_quality"] = _match_quality(r["df_predict"], filtered, cfg.get("selected_fields", []), threshold)
    return out


def _match_quality(original: pd.DataFrame, current: pd.DataFrame, fields: list, threshold: float) -> list:
    """Edge-level match quality by demographic group, before and after toggling rules."""
    keep = [f for f in fields if f in DEMOGRAPHIC_REGISTRY_FIELDS]
    before = compute_edge_demographic_quality(original, keep, threshold)
    after = compute_edge_demographic_quality(current, keep, threshold)
    out = []
    for f, base in before.items():
        cur = after.get(f, pd.DataFrame(columns=base.columns))
        merged = base.merge(cur, on=["category", "label"], how="left", suffixes=("_original", "_toggled")).fillna(0)
        out.append({"field": f, "rows": records(merged.head(12))})
    return out


def recluster(session: Session, slot: str, toggles: dict, threshold: float) -> dict:
    run = _run(session, slot)
    cfg = run.results["run_config"]
    toggles = toggles or cfg["blocking_toggles"]          # no selection sent: the run's own rules
    filtered, _ = _filtered_edges(run, {k: bool(v) for k, v in toggles.items()})
    if filtered.empty:
        raise UserError("No pairs to cluster: switch on at least one rule that covers some pairs.")
    clusters = recluster_filtered(filtered, session.a, session.b if cfg["operation_mode"] != "dedupe" else None, threshold)
    if clusters.empty:
        raise UserError("Re-clustering returned no clusters. Try a lower threshold or more rules.")
    return {"original_clusters": run.metrics["n_clusters"], "new_clusters": int(clusters["cluster_id"].nunique()),
            "edges": len(filtered)}


# ── compare runs ──────────────────────────────────────────────────────────────

def _rules_for_waterfall(session: Session, toggles: dict) -> tuple:
    run1 = _run(session, "run1")
    baseline = {k: bool(v) for k, v in run1.results["run_config"]["blocking_toggles"].items()}
    live = {k: bool(v) for k, v in toggles.items()}
    rules = list(dict.fromkeys([*live, *baseline]))
    return rules, {r: live.get(r, False) for r in rules}, baseline


def live_waterfall(session: Session, toggles: dict) -> dict:
    """Cascading waterfall for the Run 2 selection against Run 1, counted from the data.

    Pair counts per combination of rules are computed once per set of rules (and cached in the
    session), so switching a rule on or off only re-weights that small table.
    """
    run1 = _run(session, "run1")
    cfg = run1.results["run_config"]
    rules, live, baseline = _rules_for_waterfall(session, toggles)
    mode = cfg["operation_mode"]
    key = (tuple(rules), mode, len(session.a), None if session.b is None else len(session.b))
    cache = session.cache.setdefault("waterfall_patterns", {})
    needed = frozenset(r for r in rules if live.get(r) or baseline.get(r))
    cached = cache.get(key)
    # Recount only if a rule that is now on was left out of the pair budget and has not been given
    # priority yet; a rule too big to ever fit stays in "skipped" without recounting on every toggle.
    if cached is None or any(r in needed - cached[2] and n is not None for r, n in cached[1]):
        b = session.b if mode == "link_dedupe" else None
        cache.clear()                                    # keep one table: they can be large
        cache[key] = (*blocking_rule_patterns(session.a, b, mode, rules, priority=needed), needed)
    patterns, skipped, _ = cache[key]
    if patterns.empty:
        return {"fields": [], "skipped": [{"rule": r, "pairs": n} for r, n in skipped]}
    out = fig.waterfall(patterns, live, ("Run 1 rules", "Run 2 selection (live)"), baseline_toggles=baseline)
    out["skipped"] = [{"rule": r, "pairs": n, "limit": MAX_CANDIDATE_PAIRS} for r, n in skipped]
    out["and_mode"] = cfg.get("blocking_mode") == "AND"
    return out


def compare(session: Session) -> dict:
    r1, r2 = _run(session, "run1"), _run(session, "run2")
    m1, m2 = r1.metrics, r2.metrics

    def prob(m):
        stats = m["match_prob_stats"]
        return None if stats.empty else float(stats["mean_match_prob"].iloc[0])

    p1, p2 = prob(m1), prob(m2)
    inter = compute_inter_metrics(r1.results["df_predict"], r2.results["df_predict"],
                                  r1.results["df_cluster"], r2.results["df_cluster"])
    edges = inter["edge_diff"].set_index("category")["n"].to_dict()
    accuracy = []
    if (r1.cm or {}).get("precision") is not None and (r2.cm or {}).get("precision") is not None:
        accuracy = [{"measure": label, "run1": r1.cm[k], "run2": r2.cm[k], "change": round(r2.cm[k] - r1.cm[k], 4)}
                    for label, k in (("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"))]
    charts = []
    for key, x, y, title in (("prob_dist", "prob_bin", "n_edges", "Match probability"),
                             ("cluster_sizes", "n_nodes", "n_clusters", "Cluster size")):
        one, two = inter[f"{key}_run1"].assign(run="Run 1"), inter[f"{key}_run2"].assign(run="Run 2")
        if not one.empty and not two.empty:
            f = px.bar(pd.concat([one, two]), x=x, y=y, color="run", barmode="group", template="simple_white",
                       title=f"{title}: Run 1 vs Run 2", color_discrete_sequence=[fig.BLUE, fig.ORANGE])
            charts.append(fig.to_json(f.update_layout(height=320)))
    return {
        "headline": {"edges": [m1["n_edges"], m2["n_edges"]], "clusters": [m1["n_clusters"], m2["n_clusters"]],
                     "mean_probability": [p1, p2]},
        "accuracy": accuracy,
        "changes": [{"measure": "Edges in both runs", "n": edges.get("shared", 0)},
                    {"measure": "Edges only in Run 2", "n": edges.get("added", 0)},
                    {"measure": "Edges only in Run 1", "n": edges.get("removed", 0)},
                    {"measure": "Clusters identical in both runs", "n": inter["n_exact_matching_clusters"]},
                    {"measure": "Cluster pairs that partly overlap", "n": inter["n_partial_matching_clusters"]}],
        "charts": charts,
    }


# ── export ────────────────────────────────────────────────────────────────────

def cohort_csv(session: Session, slot: str) -> bytes:
    """Every input column plus ``cluster_id``; records sharing a cluster_id are one predicted entity."""
    run = _run(session, slot)
    cfg = run.results["run_config"]
    records_df = session.a if cfg["operation_mode"] == "dedupe" else pd.concat([session.a, session.b], ignore_index=True)
    clusters = run.results["df_cluster"]
    keys = ["unique_id", "source_dataset"] if "source_dataset" in clusters.columns else ["unique_id"]
    return records_df.merge(clusters[keys + ["cluster_id"]], on=keys, how="left").to_csv(index=False).encode("utf-8")


def report_html(session: Session, slot: str) -> bytes:
    run = _run(session, slot)
    r = run.results
    curve = run.threshold_curve if run.threshold_curve is not None else pd.DataFrame()
    return generate_report(slot.replace("run", "Run "), r["run_config"], run.metrics, r["n_input_records"],
                           model_params=r.get("model_params"), missingness_a=r.get("missingness_a"),
                           missingness_b=r.get("missingness_b"), blocking_counts=r.get("blocking_counts"),
                           unlinkables=r.get("unlinkables"), settings_used=r.get("settings_used"),
                           confusion_matrix=run.cm, threshold_curve=curve, curve_summary=run.curve)


def model_json(session: Session, slot: str) -> dict:
    r = _run(session, slot).results
    if not r.get("settings_used"):
        raise UserError("No model settings are available for this run.")
    return reconstruct_model_json(r["settings_used"], r.get("model_params", {}),
                                  linkage_type=r["run_config"].get("linkage_type", "probabilistic"))


def read_model(raw: bytes) -> dict:
    """Parse an uploaded model JSON and describe it; the run itself screens it before executing."""
    try:
        model = json.loads(raw)
    except ValueError as exc:
        raise UserError(f"Cannot parse the JSON: {exc}") from exc
    if not isinstance(model, dict) or "comparisons" not in model:
        raise UserError("This does not look like a Splink model JSON (no 'comparisons').")
    comps = model.get("comparisons", [])
    marker = model.get("_app_linkage_type")
    trained = any(lvl.get("m_probability") is not None and lvl.get("u_probability") is not None
                  for c in comps for lvl in c.get("comparison_levels", []))
    detected = marker if marker in ("deterministic", "probabilistic") else "probabilistic" if trained else "deterministic"
    return {"model": model, "summary": {
        "link_type": model.get("link_type", "?"), "comparisons": len(comps),
        "blocking_rules": len(model.get("blocking_rules_to_generate_predictions", [])),
        "fields": [c.get("output_column_name", "?") for c in comps],
        "detected_linkage_type": detected, "from_marker": marker in ("deterministic", "probabilistic")}}
