"""Splink orchestration: linkage runs, model I/O and the blocking explorer.

Public entry points
-------------------
``run_linkage``            train (EM) or apply exact rules, predict, cluster
``run_linkage_from_json``  apply a saved Splink model without training
``reconstruct_model_json`` turn a finished run into a re-loadable model file
``build_coverage_matrix`` / ``filter_predict_by_active_rules`` /
``compute_blocking_waterfall`` / ``recluster_filtered``
                           the interactive blocking explorer
``estimate_candidate_pairs`` pre-flight size check for a blocking configuration

The training recipe (estimate the prior from a blocking rule, estimate
u-probabilities by random sampling, then EM on one or more blocking rules) is
the one documented in Splink's own tutorials.  Splink is MIT licensed:
https://github.com/moj-analytical-services/splink

Splink is pinned to the 4.0.x line (see requirements.txt): Splink 5 changed its
dataset and settings APIs and this app has not been ported to it.
"""

import copy
import math
import multiprocessing
import os
import re
import tempfile
import threading
import warnings
from contextlib import contextmanager
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from splink import DuckDBAPI, Linker
import splink.comparison_library as cl
import splink.blocking_rule_library as brl

# =============================================================================
# ── SANDBOXING ───────────────────────────────────────────────────────────────
# An uploaded Splink model JSON contains raw SQL (blocking rules and comparison
# levels) that DuckDB will execute. On a shared deployment that is untrusted
# input, so Splink runs on a locked-down connection and uploaded SQL is
# screened first. Defence in depth: neither layer is relied on alone.
# =============================================================================

# Both are overridable by the operator, e.g. COHORT_BUILDER_DUCKDB_MEMORY_LIMIT=1GB
DUCKDB_MEMORY_LIMIT = os.environ.get("COHORT_BUILDER_DUCKDB_MEMORY_LIMIT", "2GB")
UPLOADED_MODEL_TIMEOUT_S = float(os.environ.get("COHORT_BUILDER_MODEL_TIMEOUT_S", "180"))
# A run whose blocking would generate more candidate pairs than this is refused up front:
# Splink's results are converted to pandas outside DuckDB's own memory cap.
MAX_CANDIDATE_PAIRS = int(os.environ.get("COHORT_BUILDER_MAX_CANDIDATE_PAIRS", "5000000"))

# What may not appear in uploaded SQL once quoted identifiers and string literals
# are removed: statements, subqueries, file/table functions, and DDL/DML. Comparison
# levels and blocking rules are plain boolean expressions, so none of this is needed.
_FORBIDDEN_SQL = re.compile(
    r"(;|--|/\*|\b(select|with|copy|attach|detach|install|load|pragma|set|reset|export|import|"
    r"create|drop|alter|insert|update|delete|call|summarize|describe|explain|union|"
    r"range|generate_series|unnest|glob|getenv|current_setting|read_\w+|\w+_scan)\b)",
    re.IGNORECASE,
)
_QUOTED = re.compile(r""""(?:[^"]|"")*"|'(?:[^']|'')*'""")
_MAX_SQL_LENGTH = 4000


def _sql_expressions(settings: dict):
    """Every raw SQL string an uploaded settings dict would hand to DuckDB."""
    for rule in settings.get("blocking_rules_to_generate_predictions", []):
        yield rule.get("blocking_rule", "") if isinstance(rule, dict) else str(rule)
    for comparison in settings.get("comparisons", []):
        for level in comparison.get("comparison_levels", []):
            yield str(level.get("sql_condition", ""))
            for key in ("tf_adjustment_column",):
                if level.get(key):
                    yield str(level[key])


def validate_model_sql(settings: dict) -> None:
    """Raise ``ValueError`` if uploaded model settings contain anything but plain expressions."""
    for sql in _sql_expressions(settings):
        if len(sql) > _MAX_SQL_LENGTH:
            raise ValueError("The model file contains an unreasonably long SQL expression.")
        bare = _QUOTED.sub(" ", sql)
        hit = _FORBIDDEN_SQL.search(bare)
        if hit:
            raise ValueError(
                f"The model file contains SQL that is not allowed ('{hit.group(0)}'). Blocking rules and "
                "comparison levels must be plain comparison expressions, not statements or subqueries.")


def _guarded_db_api(timeout_s: Optional[float] = None):
    """A DuckDB-backed Splink API on a locked-down in-memory connection.

    Memory and threads are capped, file/network access is disabled, and the
    configuration is locked so SQL cannot undo any of that. With ``timeout_s``
    a watchdog interrupts the running query when time is up.

    Returns ``(db_api, watchdog)``; call ``watchdog.cancel()`` when done (it is
    ``None`` without a timeout).
    """
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={max(1, min(4, os.cpu_count() or 1))}")
    con.execute("SET enable_external_access=false")
    con.execute("SET lock_configuration=true")

    watchdog = None
    if timeout_s:
        watchdog = threading.Timer(timeout_s, con.interrupt)
        watchdog.daemon = True
        watchdog.start()
    return DuckDBAPI(con), watchdog


def _ident(name: str) -> str:
    """Quote a column name for use in generated SQL (``"`` doubled)."""
    return '"' + str(name).replace('"', '""') + '"'


# ─────────────────────────────────────────────────────────────────────────────
# FIELD → COMPARISON mapping
# Maps each dataset column to an appropriate Splink comparison strategy.
# NameComparison uses Jaro-Winkler fuzzy matching (good for typos in names).
# DateOfBirthComparison handles transpositions and date-range differences.
# ExactMatch is used for categorical fields (city, email, gender, postcode).
# ─────────────────────────────────────────────────────────────────────────────
_FIELD_COMPARISONS = {
    "first_name": lambda: cl.NameComparison("first_name"),
    "surname":    lambda: cl.NameComparison("surname"),
    "dob":        lambda: cl.DateOfBirthComparison("dob", input_is_string=True),
    "city":       lambda: cl.ExactMatch("city"),
    "email":      lambda: cl.ExactMatch("email"),
    "gender":     lambda: cl.ExactMatch("gender"),
    "postcode":   lambda: cl.ExactMatch("postcode"),
}

# ─────────────────────────────────────────────────────────────────────────────
# FIELD → BLOCKING RULE mapping
# Single-field blocking rules only.  Multi-field rules cause Splink 4.0.x to
# create SaltedBlockingRules that are incompatible with u-probability sampling
# on single-CPU environments.
# ─────────────────────────────────────────────────────────────────────────────
_FIELD_BLOCKING_RULES = {
    "first_name": lambda: brl.block_on("first_name"),
    "surname":    lambda: brl.block_on("surname"),
    "dob":        lambda: brl.block_on("dob"),
    "city":       lambda: brl.block_on("city"),
    "email":      lambda: brl.block_on("email"),
    "gender":     lambda: brl.block_on("gender"),
    "postcode":   lambda: brl.block_on("postcode"),
}

# Matches the left-hand column of a rule such as  l."first_name" = r."first_name"
_BLOCKED_COLUMN = re.compile(r'l\."([^"]+)"')

DEFAULT_CLUSTER_THRESHOLD     = 0.8    # Cluster together records above this probability
DEFAULT_MATCH_WEIGHT_THRESHOLD = -5.0  # Accept most edges; clustering threshold filters


# =============================================================================
# ── DATA EXTRACTION HELPERS ──────────────────────────────────────────────────
# These are called inside run_linkage() to capture extra data for the report.
# =============================================================================

def _compute_missingness(df: pd.DataFrame, fields: list) -> dict:
    """Compute per-field completeness (% non-null values) for each linkage field.
    Returns {field_name: pct_complete} where pct_complete is 0-100.
    Used by the report's data-quality section."""
    return {
        field: round(df[field].notna().mean() * 100, 1)  # Percentage complete
        for field in fields
        if field in df.columns    # Only include fields actually present in the DataFrame
    }


def _extract_model_params(linker: Linker) -> dict:
    """Read the trained m/u probabilities and match weights back from the linker.

    Uses Splink's public ``linker.misc.save_model_to_json()`` (a settings dict
    with the current parameter estimates) rather than private attributes.

    Returns::

        {
          "comparisons": [{"field": "first_name",
                           "levels": [{"label": "Exact match on first_name",
                                       "m_prob": 0.72, "u_prob": 0.008,
                                       "match_weight": 6.49, "is_null": False}, ...]}],
          "prior_log_odds": -6.3,          # log2(lambda / (1 - lambda))
          "training_complete": True,       # at least one level has trained m and u
        }

    ``m_prob`` / ``u_prob`` / ``match_weight`` are ``None`` for null levels and
    for levels Splink could not estimate. On any failure the result is the
    empty skeleton with ``training_complete = False``.
    """
    params = {"comparisons": [], "prior_log_odds": None, "training_complete": False}
    try:
        model = linker.misc.save_model_to_json()
    except Exception:
        return params

    prior = model.get("probability_two_random_records_match")
    if prior and 0 < prior < 1:
        params["prior_log_odds"] = math.log2(prior / (1.0 - prior))

    any_trained = False
    for comparison in model.get("comparisons", []):
        levels = []
        for level in comparison.get("comparison_levels", []):
            is_null = bool(level.get("is_null_level"))
            m, u = level.get("m_probability"), level.get("u_probability")
            weight = math.log2(m / u) if (not is_null and m and u and u > 0) else None
            any_trained = any_trained or weight is not None
            levels.append({
                "label": level.get("label_for_charts") or "Unknown level",
                "m_prob": m, "u_prob": u, "match_weight": weight, "is_null": is_null,
            })
        params["comparisons"].append({"field": comparison.get("output_column_name", ""), "levels": levels})

    params["training_complete"] = any_trained
    return params


def _extract_blocking_counts(df_predict: pd.DataFrame, blocking_rule_sqls: list) -> list:
    """Count pairwise comparisons generated by each blocking rule.

    Splink 4 adds a 'match_key' integer column to df_predict indicating which
    blocking rule (0-indexed) produced each candidate pair.

    Returns a list of dicts: [{"rule_index": 0, "rule_sql": "...", "n": 1234}, ...]
    Sorted by rule_index.  Returns empty list if match_key column is absent.
    """
    if "match_key" not in df_predict.columns:
        return []    # match_key not available (deterministic link may not include it)

    try:
        con = duckdb.connect()    # Temporary in-memory DuckDB connection
        con.register("df_predict", df_predict)

        # Count how many predictions each blocking rule contributed
        counts_df = con.sql("""
            SELECT CAST(match_key AS INTEGER) AS rule_index,
                   COUNT(*) AS n
            FROM df_predict
            GROUP BY rule_index
            ORDER BY rule_index
        """).df()
        con.close()

        results = []
        for _, row in counts_df.iterrows():
            idx = int(row["rule_index"])
            # Map rule index to its SQL string; fallback if index out of range
            sql = blocking_rule_sqls[idx] if idx < len(blocking_rule_sqls) else f"Rule {idx}"
            results.append({
                "rule_index": idx,
                "rule_sql":   sql,           # SQL string for the blocking rule
                "n":          int(row["n"]), # Number of comparisons from this rule
            })
        return results
    except Exception:
        return []    # Never crash; blocking counts are supplementary data


def _compute_unlinkables(df_predict: pd.DataFrame, n_records: int) -> tuple:
    """Share of input records that cannot reach each match-weight threshold.

    A record is *linkable at threshold t* if at least one of its candidate
    pairs (as either the left or the right record) has ``match_weight >= t``.
    Returns ``(thresholds, unlinkable_pcts)`` over -20..+20 in 0.5 steps; a
    record with no candidate pair at all is unlinkable at every threshold.
    """
    if "match_weight" not in df_predict.columns or n_records == 0:
        return [], []

    sides = pd.concat([
        df_predict[["unique_id_l", "match_weight"]].rename(columns={"unique_id_l": "uid"}),
        df_predict[["unique_id_r", "match_weight"]].rename(columns={"unique_id_r": "uid"}),
    ])
    best_weight = np.sort(sides.groupby("uid")["match_weight"].max().to_numpy())

    thresholds = [t * 0.5 for t in range(-40, 41)]
    # records linkable at t = those whose best weight is >= t
    n_linkable = len(best_weight) - np.searchsorted(best_weight, thresholds, side="left")
    pcts = np.clip((n_records - n_linkable) / n_records * 100.0, 0.0, 100.0).round(1)
    return thresholds, pcts.tolist()


# =============================================================================
# ── CORE SPLINK WORKFLOW FUNCTIONS ───────────────────────────────────────────
# =============================================================================

def _diagnose_zero_edges(dataset_a: pd.DataFrame, dataset_b, blocking_toggles: dict) -> list:
    """For each ENABLED blocking field, report how many distinct values it
    has vs. how many rows share a value with at least one other row (i.e.
    would actually generate a candidate pair). A field where every value is
    unique (n_repeated == 0) explains, on its own, why blocking on it
    contributes nothing — surfaced directly in the UI instead of leaving a
    bare "0 edges" with no way to tell why.
    """
    diagnostics = []
    enabled = [k for k, v in blocking_toggles.items() if v]
    frames = [dataset_a] + ([dataset_b] if dataset_b is not None and not dataset_b.empty else [])
    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else dataset_a

    for key in enabled:
        for field in key.split("+"):
            field = field.strip()
            if field not in combined.columns:
                diagnostics.append({"field": field, "issue": "column not found after cleaning"})
                continue
            series = combined[field].dropna().astype(str)
            series = series[series.str.strip() != ""]
            n_non_null = len(series)
            n_repeated = int((series.value_counts() > 1).sum())
            diagnostics.append({
                "field": field,
                "non_null_values": n_non_null,
                "distinct_values": series.nunique(),
                "values_shared_by_2plus_rows": n_repeated,
                "issue": "every value is unique — cannot match" if n_repeated == 0 else None,
            })
    return diagnostics


def _build_comparisons(selected_fields: list, comp_types: dict = None) -> list:
    """Return Splink comparison objects for selected fields.

    comp_types: optional dict mapping field_name → comparison type string.
    Supported strings: NameComparison, DateOfBirthComparison, ExactMatch,
    LevenshteinAtThresholds, JaroWinklerAtThresholds, EmailComparison,
    PostcodeComparison.
    Falls back to _FIELD_COMPARISONS for the demo dataset's fields, then ExactMatch.
    """
    comps = []
    for f in selected_fields:
        # User-specified comparison type takes priority (upload flow)
        if comp_types and f in comp_types:
            ct = comp_types[f]
            if ct == "NameComparison":
                comps.append(cl.NameComparison(f))
            elif ct == "DateOfBirthComparison":
                comps.append(cl.DateOfBirthComparison(f, input_is_string=True))
            elif ct == "LevenshteinAtThresholds":
                comps.append(cl.LevenshteinAtThresholds(f, [1, 2]))
            elif ct == "JaroWinklerAtThresholds":
                comps.append(cl.JaroWinklerAtThresholds(f, [0.9, 0.7]))
            elif ct == "EmailComparison":
                comps.append(cl.EmailComparison(f))
            elif ct == "PostcodeComparison":
                comps.append(cl.PostcodeComparison(f))
            else:
                comps.append(cl.ExactMatch(f))   # Default for any unknown type
        elif f in _FIELD_COMPARISONS:
            comps.append(_FIELD_COMPARISONS[f]()) # Known demo-dataset fields
        else:
            comps.append(cl.ExactMatch(f))        # Safe fallback for any other field
    return comps


def _build_blocking_rules(blocking_toggles: dict, blocking_mode: str = "OR") -> list:
    """Return active Splink blocking rule objects for any field name.

    blocking_mode:
      - "OR"  (default): each enabled field/composite becomes its own rule;
        a pair is a candidate if it satisfies ANY active rule (Splink's
        standard blocking semantics — union of all rules' pairs). This is
        how the app has always worked.
      - "AND": ALL currently-enabled fields (single or already-composite)
        are merged into ONE rule requiring simultaneous agreement on every
        one of them — a pair is a candidate only if ALL active fields
        match together. Much stricter (lower recall, higher precision);
        directly prevents the "any one shared field chains unrelated
        records into one giant cluster" failure mode that loose OR
        blocking can produce.

    Supports three rule types within a single OR-mode rule:
      - Single field:    "first_name"          → brl.block_on("first_name")
      - Composite field: "first_name+surname"  → brl.block_on("first_name","surname")
      - Any field not in _FIELD_BLOCKING_RULES uses generic brl.block_on(key)
        so uploaded datasets with arbitrary column names work correctly.
    """
    enabled_keys = [key for key, enabled in blocking_toggles.items() if enabled]
    if not enabled_keys:
        raise ValueError("At least one blocking rule must be enabled.")

    if blocking_mode == "AND":
        # Flatten every enabled key (single or "a+b" composite) into one
        # combined field list, then build ONE rule requiring all of them.
        all_fields = []
        for key in enabled_keys:
            for f in key.split("+"):
                f = f.strip()
                if f and f not in all_fields:
                    all_fields.append(f)
        return [brl.block_on(*all_fields)]

    # OR mode (default) — each key is its own independent rule
    active = []
    for key in enabled_keys:
        if "+" in key:
            fields = [f.strip() for f in key.split("+") if f.strip()]
            if len(fields) >= 2:
                active.append(brl.block_on(*fields))
        else:
            active.append(brl.block_on(key))
    if not active:
        raise ValueError("At least one blocking rule must be enabled.")
    return active


def _validate_and_filter_settings(settings: dict, input_tables: list) -> dict:
    """Remove comparisons and blocking rules for columns absent from any input table.

    This is critical for link_only mode where Dataset A and Dataset B may have
    different schemas (e.g. a registration file vs a voting-history file).
    If a blocking rule references a column that doesn't exist in one table,
    DuckDB raises a Binder Error at prediction time.

    Strategy:
      1. Find the intersection of columns present in ALL input tables.
      2. Drop any comparison whose output_column_name is not in common_cols.
      3. Drop any blocking rule that references a column not in common_cols.
      4. Raise ValueError if no blocking rules survive (nothing to link on).
    """
    # Compute the set of columns that exist in every input table
    common_cols = set(input_tables[0].columns)
    for df in input_tables[1:]:
        common_cols &= set(df.columns)

    # Filter comparisons: keep only those whose column exists in all tables
    original_comps = settings.get("comparisons", [])
    settings["comparisons"] = [
        c for c in original_comps
        if c.get("output_column_name", "") in common_cols
    ]
    dropped_comps = len(original_comps) - len(settings["comparisons"])
    if dropped_comps:
        warnings.warn(
            f"{dropped_comps} comparison(s) dropped: column(s) not present in all datasets."
        )

    # Filter blocking rules: parse column names from SQL and check all exist
    original_rules = settings.get("blocking_rules_to_generate_predictions", [])
    valid_rules = []
    for rule in original_rules:
        sql = rule.get("blocking_rule", "") if isinstance(rule, dict) else str(rule)
        # Extract all l."col" column references from the SQL string
        referenced_cols = _BLOCKED_COLUMN.findall(sql)
        if all(c in common_cols for c in referenced_cols):
            valid_rules.append(rule)

    settings["blocking_rules_to_generate_predictions"] = valid_rules

    if not valid_rules:
        common_sorted = sorted(common_cols - {"source_dataset"})
        raise ValueError(
            "No valid blocking rules remain after column validation. "
            "All blocking fields must exist in BOTH Dataset A and Dataset B. "
            f"Columns present in both datasets: {common_sorted}. "
            "Please reconfigure blocking rules to use only these columns."
        )
    return settings


def _render_cluster_studio_html(linker, df_predict, df_cluster) -> str:
    """Generate Splink cluster studio HTML for embedding in Streamlit.
    Returns empty string if generation fails (never crashes the app)."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            tmp_path = tmp.name

        linker.visualisations.cluster_studio_dashboard(
            df_predict=df_predict,
            df_clustered=df_cluster,
            out_path=tmp_path,
            overwrite=True,
            return_html_as_string=True,
        )

        if os.path.exists(tmp_path):
            with open(tmp_path, "r", encoding="utf-8") as f:
                html_str = f.read()
            os.remove(tmp_path)
            return html_str
    except Exception:
        pass
    return ""


def _blocking_fields(blocking_toggles: dict) -> list:
    """Distinct field names used by the enabled blocking rules ('a+b' -> a, b)."""
    fields: list = []
    for key, enabled in blocking_toggles.items():
        if enabled:
            for name in (part.strip() for part in key.split("+")):
                if name and name not in fields:
                    fields.append(name)
    return fields


def _deterministic_predictions(linker, comparison_fields: list, agreement_fields: list):
    """Exact-rule linkage: candidate pairs that agree on enough fields, scored 1.0.

    ``deterministic_link()`` returns every pair that satisfies *at least one*
    blocking rule and never checks the other fields.  With several loose
    single-field rules OR-ed together, records that share just one common value
    get chained into one giant cluster by connected-components clustering.  So
    a pair is only accepted when it agrees exactly (and non-null) on at least
    two of ``agreement_fields`` - the fields the user declared as identifying
    by turning their blocking rules on - or on the single field if only one.
    Falls back to every compared field when no blocking field is known.

    Returns the accepted pairs registered in the linker with
    ``match_probability = 1.0`` and ``match_weight = 100``.
    """
    pairs = linker.inference.deterministic_link().as_pandas_dataframe()

    fields = [f for f in (agreement_fields or comparison_fields)
              if f"{f}_l" in pairs.columns and f"{f}_r" in pairs.columns]
    if fields:
        agreeing = sum(
            (pairs[f"{f}_l"] == pairs[f"{f}_r"]) & pairs[f"{f}_l"].notna() for f in fields
        )
        pairs = pairs[agreeing >= min(2, len(fields))].copy()

    pairs["match_probability"] = 1.0
    pairs["match_weight"] = 100.0
    for side in ("l", "r"):
        if f"source_dataset_{side}" not in pairs.columns:
            pairs[f"source_dataset_{side}"] = "A"
    return linker.table_management.register_table(pairs, "df_predict_enriched")


# =============================================================================
# ── PUBLIC API ────────────────────────────────────────────────────────────────
# =============================================================================

def run_linkage_from_json(
    model_json:       dict,
    dataset_a:            pd.DataFrame,
    dataset_b:            Optional[pd.DataFrame],
    operation_mode:   str,
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    linkage_type:     str = "probabilistic",
) -> dict:
    """Run prediction from a pre-trained Splink model JSON (advanced flow).

    The model file is untrusted input: its SQL is screened by
    :func:`validate_model_sql` (``ValueError`` if it is not plain expressions)
    and runs on a locked-down DuckDB connection with a time limit
    (``COHORT_BUILDER_MODEL_TIMEOUT_S``, default 180 s). See ``_apply_model``
    for the processing itself.
    """
    validate_model_sql(model_json)
    db_api, watchdog = _guarded_db_api(UPLOADED_MODEL_TIMEOUT_S)
    try:
        return _apply_model(model_json, dataset_a, dataset_b, operation_mode,
                            cluster_threshold, linkage_type, db_api)
    except duckdb.InterruptException as exc:
        raise TimeoutError(
            f"The model took longer than {UPLOADED_MODEL_TIMEOUT_S:.0f} s to apply and was stopped.") from exc
    finally:
        if watchdog is not None:
            watchdog.cancel()


def _apply_model(
    model_json:       dict,
    dataset_a:            pd.DataFrame,
    dataset_b:            Optional[pd.DataFrame],
    operation_mode:   str,
    cluster_threshold: float,
    linkage_type:     str,
    db_api,
) -> dict:
    """Apply a saved Splink model to the data (no training).

    Accepts any valid Splink 4.x settings JSON that already contains trained
    m/u probabilities, OR a deterministic settings JSON with no m/u values.
    No EM training is performed here; the model is used as-is for inference.

    Args:
        model_json       : Parsed Splink settings dict (from uploaded .json file)
        dataset_a            : Dataset A (source_dataset = 'A')
        dataset_b            : Dataset B or None
        operation_mode   : 'dedupe' or 'link_dedupe'
        cluster_threshold: Match probability threshold for clustering
        linkage_type      : 'deterministic' or 'probabilistic'. Mirrors the same
                             flag used by run_linkage() so Standard, Upload, and
                             Advanced flows all resolve deterministic matches the
                             same way (deterministic_link() + multi-field
                             agreement check), instead of Advanced silently
                             always calling predict().

    Returns same dict structure as run_linkage() so the rest of the app
    (analysis, report, confusion matrix) works identically for both flows.
    """
    # Work on a copy so we never mutate the user's uploaded dict. Keys that start
    # with "_" are this app's own markers (e.g. "_app_linkage_type"); Splink
    # rejects unknown settings keys, so they must not reach it.
    settings = {k: v for k, v in model_json.items() if not k.startswith("_")}

    # Force column retention so metrics and explorer can inspect field values
    settings["retain_intermediate_calculation_columns"] = True
    settings["retain_matching_columns"]                 = True

    # Override link_type if the JSON doesn't match the chosen operation mode
    if operation_mode == "dedupe":
        settings["link_type"] = "dedupe_only"
        df_input   = dataset_a.copy()
        df_input["source_dataset"] = "A"
        input_tables    = [df_input]
        n_input_records = len(df_input)
    else:
        settings["link_type"] = "link_only"
        input_tables    = [dataset_a.copy(), dataset_b.copy() if dataset_b is not None else None]
        input_tables    = [t for t in input_tables if t is not None]
        n_input_records = len(dataset_a) + (len(dataset_b) if dataset_b is not None else 0)

    # Defensive cast (Advanced flow): a JSON model can be replayed against
    # ANY loaded dataset — demo, voter registry, or a re-uploaded one — so cast
    # unique_id to string here too, consistent with run_linkage().
    for _t in input_tables:
        if "unique_id" in _t.columns:
            _t["unique_id"] = _t["unique_id"].astype(str)

    # Pull the field names used in comparisons for missingness reporting
    comparison_fields = [
        c.get("output_column_name", "")
        for c in settings.get("comparisons", [])
        if c.get("output_column_name")
    ]

    # Compute missingness before building the linker
    missingness_a = _compute_missingness(dataset_a, comparison_fields)
    missingness_b = (
        _compute_missingness(dataset_b, comparison_fields)
        if dataset_b is not None and operation_mode != "dedupe"
        else {}
    )

    # Blocking-rule SQL and the fields those rules reference
    blocking_rule_sqls = [
        (r.get("blocking_rule", "") if isinstance(r, dict) else str(r))
        for r in settings.get("blocking_rules_to_generate_predictions", [])
    ]
    agreement_fields = list(dict.fromkeys(
        name for sql in blocking_rule_sqls for name in _BLOCKED_COLUMN.findall(sql)
    ))

    # Build linker from uploaded settings – no training step
    linker  = Linker(
        input_table_or_tables=input_tables,
        settings=settings,
        db_api=db_api,
        set_up_basic_logging=False,
    )

    # ── Run inference: mirrors run_linkage()'s deterministic/probabilistic
    #    branch exactly, so a model trained in Standard/Upload mode and
    #    replayed here behaves identically instead of Advanced mode always
    #    forcing predict() (which produces garbage on a deterministic model
    #    with null m/u probabilities). ─────────────────────────────────────
    if linkage_type == "deterministic":
        df_predict = _deterministic_predictions(linker, comparison_fields, agreement_fields)
    else:
        # Run prediction (threshold very low so all pairs are returned)
        df_predict = linker.inference.predict(
            threshold_match_weight=DEFAULT_MATCH_WEIGHT_THRESHOLD
        )

    # Cluster
    df_cluster = linker.clustering.cluster_pairwise_predictions_at_threshold(
        df_predict,
        threshold_match_probability=cluster_threshold,
    )

    df_predict_pd = df_predict.as_pandas_dataframe()
    df_cluster_pd = df_cluster.as_pandas_dataframe()

    # Extract trained model parameters for the match weights chart
    # (empty/partial for deterministic models — expected, matches run_linkage())
    model_params = _extract_model_params(linker)

    blocking_counts = _extract_blocking_counts(df_predict_pd, blocking_rule_sqls)

    # Unlinkable records curve
    thresh, pcts = _compute_unlinkables(df_predict_pd, n_input_records)

    # Cluster studio HTML
    cluster_html = _render_cluster_studio_html(linker, df_predict, df_cluster)

    # Build a run_config that the rest of the app can consume
    # Parse blocking toggles: each SQL like 'l."field" = r."field"' → field name
    blocking_toggles_from_json = {}
    for sql in blocking_rule_sqls:
        # Extract the field name from 'l."field" = r."field"' pattern
        matches = _BLOCKED_COLUMN.findall(sql)
        if matches:
            blocking_toggles_from_json[matches[0]] = True

    run_config = {
        "operation_mode":    operation_mode,
        "linkage_type":      linkage_type,   # now reflects the actual chosen/detected methodology
        "selected_fields":   comparison_fields,
        "blocking_toggles":  blocking_toggles_from_json,
        "cluster_threshold": cluster_threshold,
        "link_type":         settings.get("link_type", "dedupe_only"),
        "from_json":         True,             # Flag so UI can show "Advanced flow"
    }

    return {
        "df_predict":       df_predict_pd,
        "df_cluster":       df_cluster_pd,
        "cluster_html":     cluster_html,
        "n_edges":          len(df_predict_pd),
        "n_clusters":       df_cluster_pd["cluster_id"].nunique(),
        "n_input_records":  n_input_records,
        "settings_used":    settings,
        "model_params":     model_params,
        "missingness_a":    missingness_a,
        "missingness_b":    missingness_b,
        "blocking_counts":  blocking_counts,
        "unlinkables":      {"thresholds": thresh, "pcts": pcts},
        "run_config":       run_config,
    }


# =============================================================================
# ── INTERACTIVE BLOCKING EXPLORER ────────────────────────────────────────────
# Lets users toggle blocking rules on/off and see df_predict update live.
# Uses retain_matching_columns=True so field values are already in df_predict,
# meaning no extra join to the original datasets is needed.
# =============================================================================

def build_coverage_matrix(
    df_predict:     pd.DataFrame,
    active_fields:  list,
) -> pd.DataFrame:
    """Compute which blocking rules would cover each pair in df_predict.

    Because retain_matching_columns=True, df_predict already contains
    field_l and field_r columns for every comparison field.  A blocking rule
    for field X covers a pair if field_X_l == field_X_r (exact match).

    Runs as a single DuckDB query instead of sequential pandas column
    operations: DuckDB's vectorised engine computes every covers_<field>
    column in one pass without materialising per-field .astype(str)
    intermediate Series — matters once df_predict is a multi-million-row
    pairs table (voter-registry scale). Cached: identical (df_predict,
    active_fields) combinations — e.g. re-rendering after an unrelated
    widget interaction — return instantly instead of recomputing.

    Returns a slim DataFrame with:
      unique_id_l, unique_id_r, source_dataset_l, source_dataset_r,
      match_key, match_probability, match_weight,
      covers_<field>  (bool)  for each active field
    """
    id_cols    = ["unique_id_l", "unique_id_r", "source_dataset_l", "source_dataset_r"]
    score_cols = ["match_key", "match_probability", "match_weight"]
    keep       = [c for c in id_cols + score_cols if c in df_predict.columns]

    select_parts = [_ident(c) for c in keep]
    valid_fields  = []
    for field in active_fields:
        col_l, col_r = f"{field}_l", f"{field}_r"
        if col_l in df_predict.columns and col_r in df_predict.columns:
            select_parts.append(
                f'({_ident(col_l)} IS NOT NULL AND {_ident(col_r)} IS NOT NULL '
                f'AND CAST({_ident(col_l)} AS VARCHAR) = CAST({_ident(col_r)} AS VARCHAR)) '
                f'AS {_ident("covers_" + field)}'
            )
            valid_fields.append(field)

    con = duckdb.connect()
    try:
        con.register("df_predict_cov", df_predict)
        result = con.sql(f"SELECT {', '.join(select_parts)} FROM df_predict_cov").df()
    finally:
        con.close()

    # Fields whose _l/_r columns weren't retained (never chosen as a
    # comparison field) are simply not covered by anything.
    for field in active_fields:
        if field not in valid_fields:
            result[f"covers_{field}"] = False

    return result


def filter_predict_by_active_rules(
    df_predict:       pd.DataFrame,
    coverage_matrix:  pd.DataFrame,
    active_toggles:   dict,
) -> pd.DataFrame:
    """Filter df_predict to pairs covered by at least one active blocking rule.

    If a pair was originally captured by rule A (now disabled) but would also
    be captured by rule B (still active), the pair is retained.  The coverage
    matrix encodes all rules that WOULD cover each pair, not just the one
    that originally generated it (match_key).

    Returns a filtered df_predict with a new 'effective_rule' column showing
    the name of the first active rule that covers each pair.
    """
    active_fields = [f for f, v in active_toggles.items() if v]
    if not active_fields:
        # No active rules → empty table
        return df_predict.iloc[0:0].copy()

    cover_cols = [f"covers_{f}" for f in active_fields
                  if f"covers_{f}" in coverage_matrix.columns]
    if not cover_cols:
        return df_predict.copy()

    # A pair is included if ANY active coverage column is True
    mask = coverage_matrix[cover_cols].any(axis=1)

    # Get the pair IDs (+ coverage columns) that survive the filter
    id_cols   = ["unique_id_l", "unique_id_r", "source_dataset_l", "source_dataset_r"]
    id_cols   = [c for c in id_cols if c in coverage_matrix.columns]
    surviving = coverage_matrix.loc[mask, id_cols + cover_cols].copy()

    # ── Vectorised 'effective_rule' ────────────────────────────────────────────
    # Previously computed with a Python-level `for _, row in df.iterrows()`
    # loop, which does not scale: probabilistic mode on a large registry
    # data can produce hundreds of thousands of candidate pairs (unlike
    # deterministic mode, which is pruned by the multi-field agreement
    # check), and iterrows() over that made the explorer hang/appear
    # unresponsive. This does the same "first active field wins" logic with
    # vectorised pandas operations: iterate active_fields in REVERSE so the
    # first field in the original priority order is applied last and wins.
    effective_rule = pd.Series("unknown", index=surviving.index)
    for f in reversed(active_fields):
        col = f"covers_{f}"
        if col in surviving.columns:
            effective_rule = effective_rule.mask(surviving[col], f)
    surviving["effective_rule"] = effective_rule
    surviving = surviving.drop(columns=cover_cols)

    # Join back to get all original df_predict columns for surviving pairs
    merge_keys = [c for c in id_cols if c in df_predict.columns]
    filtered = df_predict.merge(surviving, on=merge_keys, how="inner")

    return filtered


def _pair_table(dataset_a: pd.DataFrame, dataset_b: Optional[pd.DataFrame], operation_mode: str,
                columns: list) -> tuple:
    """``(table, guard, use_b)``: the records of A (and B for link modes) as strings with a row id,
    plus the SQL condition that picks each candidate pair once (A x B, or i < j within A)."""
    use_b = operation_mode != "dedupe" and dataset_b is not None and not dataset_b.empty
    frames = [dataset_a[columns].astype("string").assign(_src=0)]
    if use_b:
        frames.append(dataset_b[columns].astype("string").assign(_src=1))
    table = pd.concat(frames, ignore_index=True)
    table["_rid"] = range(len(table))
    guard = "l._src = 0 AND r._src = 1" if use_b else "l._rid < r._rid"
    return table, guard, use_b


def _agree(parts: list) -> str:
    return " AND ".join(f"l.{_ident(p)} = r.{_ident(p)}" for p in parts)


def candidate_pairs_by_rule(
    dataset_a:        pd.DataFrame,
    dataset_b:        Optional[pd.DataFrame],
    blocking_toggles: dict,
    blocking_mode:    str,
    operation_mode:   str,
) -> list:
    """``[(rule, pairs), ...]``, largest first: how many candidate pairs each switched-on rule generates.

    One DuckDB hash join per rule (fast even at 50,000+ records) without building the wide output a real run
    produces. In AND mode all the rules are one rule, named by joining its fields with ``+``.
    """
    named = [(key, [p.strip() for p in key.split("+") if p.strip()]) for key, on in blocking_toggles.items() if on]
    available = set(dataset_a.columns) if dataset_a is not None else set()
    named = [(key, parts) for key, parts in named if parts and all(p in available for p in parts)]
    if not named or dataset_a.empty:
        return []
    if blocking_mode == "AND":
        merged = sorted({p for _, parts in named for p in parts})
        named = [("+".join(merged), merged)]

    table, guard, _ = _pair_table(dataset_a, dataset_b, operation_mode, sorted({p for _, parts in named for p in parts}))
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    try:
        con.register("t", table)
        counts = [(key, int(con.sql(f"SELECT COUNT(*) FROM t l JOIN t r ON {_agree(parts)} AND {guard}").fetchone()[0]))
                  for key, parts in named]
    finally:
        con.close()
    return sorted(counts, key=lambda kv: -kv[1])


def estimate_candidate_pairs(
    dataset_a:            pd.DataFrame,
    dataset_b:            Optional[pd.DataFrame],
    selected_fields:  list,
    blocking_toggles: dict,
    blocking_mode:    str,
    operation_mode:   str,
) -> int:
    """Pre-flight count of the candidate pairs the blocking configuration will generate.

    OR mode sums the rules, so pairs found by several rules are counted more than once: a slight
    overestimate, which is the safe side for a size check. AND mode counts the pairs agreeing on every
    field of every rule.
    """
    return sum(n for _, n in candidate_pairs_by_rule(dataset_a, dataset_b, blocking_toggles, blocking_mode, operation_mode))


def blocking_rule_patterns(
    dataset_a:      pd.DataFrame,
    dataset_b:      Optional[pd.DataFrame],
    operation_mode: str,
    rules:          list,
    max_pairs:      Optional[int] = None,
    priority:       tuple = (),
) -> tuple:
    """Which blocking rules would generate each candidate pair, counted straight from the data.

    This is what makes the Compare Runs waterfall live: unlike a coverage matrix built from a
    finished run's predictions, it knows about rules the run never used, including new
    composite rules such as ``first_name+last_name`` (a pair must agree on every part).

    A rule is a column name or an ``a+b`` composite. Candidate pairs are pairs of records in
    Dataset A for ``dedupe``, otherwise A x B pairs, matching what the linkage itself scores.
    The result has one row per distinct *pattern* (which rules cover the pair) with the number
    of pairs showing it, so memory stays small however many pairs there are:

        covers_<rule> (bool) for each included rule, ``n_pairs`` (int)

    Rules in ``priority`` (those switched on) are admitted first, then the rest, each group from most
    to least selective, until ``max_pairs`` (the app's pair budget) would be exceeded; the rest are
    returned in ``skipped`` as ``(rule, pair_count)`` and left
    out of the table. Rules naming a column the data lacks are skipped with ``pair_count`` None.

    Returns ``(patterns, skipped)``.
    """
    max_pairs = MAX_CANDIDATE_PAIRS if max_pairs is None else max_pairs
    use_b = operation_mode != "dedupe" and dataset_b is not None and not dataset_b.empty
    available = set(dataset_a.columns) & (set(dataset_b.columns) if use_b else set(dataset_a.columns))

    parts_of, skipped = {}, []
    for rule in dict.fromkeys(rules):
        parts = [p.strip() for p in rule.split("+") if p.strip()]
        if parts and all(p in available for p in parts):
            parts_of[rule] = parts
        else:
            skipped.append((rule, None))
    if not parts_of or dataset_a.empty:
        return pd.DataFrame(columns=["n_pairs"]), skipped

    table, guard, use_b = _pair_table(dataset_a, dataset_b, operation_mode,
                                      sorted({p for parts in parts_of.values() for p in parts}))

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    try:
        con.register("t", table)
        sizes = {rule: int(con.sql(f"SELECT COUNT(*) FROM t l JOIN t r ON {_agree(parts)} AND {guard}")
                           .fetchone()[0]) for rule, parts in parts_of.items()}
        included, used = [], 0
        for rule in sorted(sizes, key=lambda r: (r not in priority, sizes[r])):
            if used + sizes[rule] <= max_pairs:
                included.append(rule)
                used += sizes[rule]
            else:
                skipped.append((rule, sizes[rule]))
        if not included:
            return pd.DataFrame(columns=["n_pairs"]), skipped

        pairs = " UNION ".join(
            f"SELECT l._rid AS i, r._rid AS j FROM t l JOIN t r ON {_agree(parts_of[rule])} AND {guard}"
            for rule in included)
        flags = ", ".join(f"COALESCE({_agree(parts_of[rule])}, FALSE) AS {_ident('covers_' + rule)}"
                          for rule in included)
        positions = ", ".join(str(k + 1) for k in range(len(included)))
        patterns = con.sql(
            f"WITH p AS ({pairs}) SELECT {flags}, COUNT(*) AS n_pairs "
            f"FROM p JOIN t l ON l._rid = p.i JOIN t r ON r._rid = p.j GROUP BY {positions}").df()
    finally:
        con.close()
    return patterns, skipped


def _pair_weights(coverage_matrix: pd.DataFrame) -> pd.Series:
    """Pairs each row stands for: ``n_pairs`` for a pattern table, else 1 per pair."""
    if "n_pairs" in coverage_matrix.columns:
        return coverage_matrix["n_pairs"].astype("int64")
    return pd.Series(1, index=coverage_matrix.index, dtype="int64")


def _get_or_build_covers_column(coverage_matrix: pd.DataFrame, field_key: str):
    """Return the covers_<field_key> boolean series from coverage_matrix,
    building it on the fly for composite ('a+b') keys by AND-ing each
    component's own covers_ column — composite blocking rules require
    simultaneous agreement on every part. Returns None if the field (or any
    of its composite parts) isn't present in the coverage matrix."""
    col = f"covers_{field_key}"
    if col in coverage_matrix.columns:
        return coverage_matrix[col]
    if "+" in field_key:
        parts = [p.strip() for p in field_key.split("+") if p.strip()]
        part_cols = [f"covers_{p}" for p in parts]
        if parts and all(c in coverage_matrix.columns for c in part_cols):
            combined = coverage_matrix[part_cols[0]].copy()
            for c in part_cols[1:]:
                combined = combined & coverage_matrix[c]
            return combined
    return None


def determine_cascade_order(coverage_matrix: pd.DataFrame, fields: list) -> list:
    """Fixed rule priority order for the waterfall chart: descending by each
    field's own standalone pair count (how many edges it covers on its own,
    regardless of other rules — including composite 'a+b' rules, computed
    as the AND of their component fields), tie-broken by the order fields
    appear in `fields` (i.e. configuration selection order).
    """
    scored = []
    weights = _pair_weights(coverage_matrix)
    for i, f in enumerate(fields):
        series = _get_or_build_covers_column(coverage_matrix, f)
        count = int(weights[series].sum()) if series is not None else 0
        scored.append((-count, i, f))
    scored.sort()
    return [f for _, _, f in scored]


def compute_blocking_waterfall(coverage_matrix: pd.DataFrame, cascade_order: list,
                                active_toggles: dict) -> dict:
    """Computes cascading blocking-rule attribution for the waterfall chart.

    Addresses the core accuracy issue with the old per-rule pair-count
    badges: disabling a rule does NOT mean all its edges are lost — later
    rules in the cascade may already cover the same pairs and "catch" them.
    This mirrors filter_predict_by_active_rules()'s effective_rule logic
    (first ENABLED rule in cascade order that covers a pair), computed
    twice: once assuming every cascade field is active (the theoretical
    baseline / reference values), and once using only the CURRENTLY enabled
    fields (the live, redistributed values).

    Returns:
      fields            : valid cascade field/composite-rule names, in order
      all_active_count  : {field: baseline standalone unique share} — shown
                           as the red reference bar when a field is disabled
      active_only_count : {field: current redistributed share} — what the
                           field actually contributes right now, including
                           any overflow caught from disabled upstream rules
      grand_total       : total distinct edges covered if ALL cascade fields
                           were active
      active_total      : total distinct edges actually recoverable under
                           the current toggle state
    """
    valid_fields, cover_series = [], {}
    for f in cascade_order:
        series = _get_or_build_covers_column(coverage_matrix, f)
        if series is not None:
            valid_fields.append(f)
            cover_series[f] = series

    if not valid_fields or coverage_matrix.empty:
        return {"fields": [], "all_active_count": {}, "active_only_count": {},
                "grand_total": 0, "active_total": 0}

    idx = coverage_matrix.index
    weights = _pair_weights(coverage_matrix)

    def share_by_rule(effective: pd.Series, covered: pd.Series) -> dict:
        return {k: int(v) for k, v in weights[covered].groupby(effective[covered]).sum().items()}

    # ── Baseline: effective rule assuming ALL cascade fields active ─────────
    eff_all = pd.Series("unknown", index=idx)
    any_covered_all = pd.Series(False, index=idx)
    for f in reversed(valid_fields):
        eff_all = eff_all.mask(cover_series[f], f)
        any_covered_all = any_covered_all | cover_series[f]
    all_active_count = share_by_rule(eff_all, any_covered_all)
    grand_total = int(weights[any_covered_all].sum())

    # ── Current: effective rule among only ENABLED cascade fields ───────────
    enabled_fields = [f for f in valid_fields if active_toggles.get(f, False)]
    if enabled_fields:
        eff_active = pd.Series("unknown", index=idx)
        any_covered_active = pd.Series(False, index=idx)
        for f in reversed(enabled_fields):
            eff_active = eff_active.mask(cover_series[f], f)
            any_covered_active = any_covered_active | cover_series[f]
        active_only_count = share_by_rule(eff_active, any_covered_active)
        active_total = int(weights[any_covered_active].sum())
    else:
        active_only_count = {}
        active_total = 0

    return {
        "fields": valid_fields,
        "all_active_count": all_active_count,
        "active_only_count": active_only_count,
        "grand_total": grand_total,
        "active_total": active_total,
    }


def recluster_filtered(
    df_predict_filtered: pd.DataFrame,
    dataset_a:               pd.DataFrame,
    dataset_b:               Optional[pd.DataFrame],
    threshold:           float = DEFAULT_CLUSTER_THRESHOLD,
) -> pd.DataFrame:
    """Re-cluster a filtered df_predict using Splink's standalone clustering.

    Does not require a Linker instance; uses the standalone function from
    splink.clustering which runs connected components on the edge list.
    This is fast even for thousands of pairs since the graph is small.

    Returns a df_cluster DataFrame (unique_id, cluster_id, source_dataset).
    Returns empty DataFrame if clustering fails.
    """
    try:
        # Standalone clustering function (no Linker required)
        from splink.clustering import cluster_pairwise_predictions_at_threshold as _cluster

        # Build the nodes table from original datasets
        if dataset_b is not None:
            nodes = pd.concat([dataset_a, dataset_b], ignore_index=True)
        else:
            nodes = dataset_a.copy()

        db_api, _ = _guarded_db_api()    # Fresh locked-down in-memory DuckDB
        result = _cluster(
            nodes=nodes,                        # All records (nodes in the graph)
            edges=df_predict_filtered,           # Filtered edge list
            db_api=db_api,
            node_id_column_name="unique_id",     # Column that uniquely identifies records
            threshold_match_probability=threshold,
        )
        return result.as_pandas_dataframe()
    except Exception:
        return pd.DataFrame()                    # Return empty on any error; never crash


# =============================================================================
# ── HYPERPARAMETER-AWARE TRAINING ─────────────────────────────────────────────
# Updated run_linkage signature: accepts hyperparams dict so the UI can
# expose EM iterations, convergence, recall estimate, and sample size.
# =============================================================================

def run_linkage(
    dataset_a:            pd.DataFrame,
    dataset_b:            Optional[pd.DataFrame],
    selected_fields:  list,
    blocking_toggles: dict,
    operation_mode:   str,
    linkage_type:     str,
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    hyperparams:      Optional[dict] = None,   # EM training hyperparameters
    composite_rules:  Optional[dict] = None,   # composite blocking rules e.g. "first_name+dob"
    comp_types:       Optional[dict] = None,   # field → comparison type (upload flow)
    blocking_mode:    str = "OR",              # "OR" (any rule matches) or "AND" (all rules together)
) -> dict:
    """Thin wrapper: merges composite rules into blocking_toggles, then delegates
    to the internal logic.  All hyperparams are forwarded to the model settings
    and training functions.

    hyperparams keys (all optional, defaults in parentheses):
      max_iterations  : int   (25)      - max EM iterations
      em_convergence  : float (0.0001)  - stop when change < this
      recall_estimate : float (0.6)     - used in prior probability estimate
    """
    hp = hyperparams or {}

    # Merge composite rules (e.g. "first_name+surname") into blocking_toggles
    merged_toggles = dict(blocking_toggles)
    for key, enabled in (composite_rules or {}).items():
        merged_toggles[key] = enabled

    # ── Determine Splink link_type ─────────────────────────────────────────────
    link_type = "dedupe_only" if operation_mode == "dedupe" else "link_only"

    # ── Prepare input tables ───────────────────────────────────────────────────
    if operation_mode == "dedupe":
        df_for_dedupe = dataset_a.copy()
        df_for_dedupe["source_dataset"] = "A"
        input_tables    = [df_for_dedupe]
        n_input_records = len(df_for_dedupe)
    else:
        input_tables    = [dataset_a, dataset_b]
        n_input_records = len(dataset_a) + len(dataset_b)

    # ── Missingness ────────────────────────────────────────────────────────────
    missingness_a = _compute_missingness(dataset_a, selected_fields)
    missingness_b = (
        _compute_missingness(dataset_b, selected_fields)
        if dataset_b is not None and operation_mode != "dedupe"
        else {}
    )

    # ── Build settings (now uses hyperparams for EM config) ───────────────────
    settings = _build_model_settings_hp(
        link_type, selected_fields, merged_toggles, hp,
        comp_types=comp_types,    # Pass user-specified comparison types
        blocking_mode=blocking_mode,
    )

    # ── Ensure Splink-required columns exist in EVERY input table ────────────
    # unique_id and source_dataset MUST be present in all tables before the
    # Linker is created.  If either is absent from any table we add it now,
    # so the column intersection used by the UNION ALL alignment below always
    # includes these two columns.
    for _i, _df in enumerate(input_tables):
        _df = _df.copy()
        if "unique_id" not in _df.columns:
            _prefix = ("AB"[_i] if _i < 2 else str(_i))
            _df.insert(0, "unique_id",
                       _prefix + "_" + pd.Series(range(len(_df))).astype(str))
        # Defensive cast: guards against any dataset entry point (demo,
        # voter registry, uploaded CSV) that slips through with a non-string
        # unique_id (e.g. an int column) — Splink handles mixed dtypes
        # across tables poorly.
        _df["unique_id"] = _df["unique_id"].astype(str)
        if "source_dataset" not in _df.columns:
            _df["source_dataset"] = "A" if _i == 0 else "B"
        input_tables[_i] = _df

    # ── Column alignment for link mode ────────────────────────────────────────
    # Splink generates a UNION ALL of all input tables to create an internal
    # concatenated table used for all subsequent SQL. UNION ALL requires
    # IDENTICAL column lists in every table. If Dataset A has columns that
    # Dataset B lacks (e.g. a registration file has first_name but a voting
    # history file does not), the UNION ALL SQL fails with "column not found".
    # Fix: restrict ALL input tables to the intersection of their columns
    # BEFORE passing them to the Linker. Comparisons and blocking rules are
    # then validated against this common schema by _validate_and_filter_settings.
    if len(input_tables) > 1:
        # Compute the set of columns that exist in every input table
        _common_schema = set(input_tables[0].columns)
        for _df in input_tables[1:]:
            _common_schema &= set(_df.columns)
        # Drop columns that are not shared so the UNION ALL schema is consistent
        input_tables = [
            _df[[c for c in _df.columns if c in _common_schema]].copy()
            for _df in input_tables
        ]

    # Remove comparisons and blocking rules whose columns are not in all tables.
    settings = _validate_and_filter_settings(settings, input_tables)

    db_api, _ = _guarded_db_api()
    linker  = Linker(
        input_table_or_tables=input_tables,
        settings=settings,
        db_api=db_api,
        set_up_basic_logging=False,
    )

    # ── Run model ─────────────────────────────────────────────────────────────
    model_params = {}
    if linkage_type == "deterministic":
        df_predict = _deterministic_predictions(
            linker, selected_fields, _blocking_fields(merged_toggles))
    else:
        # Probabilistic with user-supplied hyperparams
        _train_probabilistic_hp(linker, selected_fields, hp, blocking_toggles=merged_toggles)
        model_params = _extract_model_params(linker)
        df_predict   = linker.inference.predict(
            threshold_match_weight=DEFAULT_MATCH_WEIGHT_THRESHOLD
        )

    # ── Cluster ────────────────────────────────────────────────────────────────
    df_cluster = linker.clustering.cluster_pairwise_predictions_at_threshold(
        df_predict,
        threshold_match_probability=cluster_threshold,
    )

    df_predict_pd = df_predict.as_pandas_dataframe()
    df_cluster_pd = df_cluster.as_pandas_dataframe()

    # ── Blocking counts, unlinkables, cluster studio ──────────────────────────
    blocking_rule_sqls = [
        r["blocking_rule"]
        for r in settings["blocking_rules_to_generate_predictions"]
    ]
    blocking_counts = _extract_blocking_counts(df_predict_pd, blocking_rule_sqls)
    thresh, pcts    = _compute_unlinkables(df_predict_pd, n_input_records)
    cluster_html    = _render_cluster_studio_html(linker, df_predict, df_cluster)

    return {
        "df_predict":       df_predict_pd,
        "df_cluster":       df_cluster_pd,
        "cluster_html":     cluster_html,
        "n_edges":          len(df_predict_pd),
        "n_clusters":       df_cluster_pd["cluster_id"].nunique(),
        "n_input_records":  n_input_records,
        "settings_used":    settings,
        "model_params":     model_params,
        "missingness_a":    missingness_a,
        "missingness_b":    missingness_b,
        "blocking_counts":  blocking_counts,
        "unlinkables":      {"thresholds": thresh, "pcts": pcts},
        "run_config": {
            "operation_mode":    operation_mode,
            "linkage_type":      linkage_type,
            "selected_fields":   selected_fields,
            "blocking_toggles":  merged_toggles,
            "cluster_threshold": cluster_threshold,
            "link_type":         link_type,
            "hyperparams":       hp,
            "blocking_mode":     blocking_mode,
            "from_json":         False,
        },
        # Per-field cardinality diagnostic — only computed when the run
        # produced zero edges, so it costs nothing on a normal successful
        # run. Answers the question "which enabled blocking field(s) have
        # no repeated values at all" directly, instead of leaving 0 edges
        # as an unexplained dead end.
        "zero_edge_diagnostic": (
            _diagnose_zero_edges(dataset_a, dataset_b, merged_toggles)
            if len(df_predict_pd) == 0 else None
        ),
    }


def _build_model_settings_hp(link_type, selected_fields, blocking_toggles, hp,
                              comp_types: dict = None, blocking_mode: str = "OR") -> dict:
    """Build Splink settings dict accepting hyperparams and comparison types."""
    comparisons    = _build_comparisons(selected_fields, comp_types)  # user types
    blocking_rules = _build_blocking_rules(blocking_toggles, blocking_mode)  # handles composites + AND/OR
    return {
        "link_type":         link_type,
        "unique_id_column_name": "unique_id",
        "comparisons": [c.create_comparison_dict("duckdb") for c in comparisons],
        "blocking_rules_to_generate_predictions": [
            r.create_blocking_rule_dict("duckdb") for r in blocking_rules
        ],
        "retain_matching_columns":                True,
        "retain_intermediate_calculation_columns": True,
        "max_iterations":  hp.get("max_iterations", 25),      # exposed to UI
        "em_convergence":  hp.get("em_convergence", 0.0001),  # exposed to UI
    }


def _select_training_blocking_fields(selected_fields: list, blocking_toggles: dict = None) -> list:
    """Rank candidate fields for EM training blocking rules, most selective
    first.

    Prefers fields the user has already enabled as ACTIVE blocking rules
    (works for any dataset), falling back to a broader
    set of conventionally high-selectivity field-name hints, then to
    whatever fields were selected for comparison. Returns the FULL ranked
    list (not just the top 1-2) so _train_probabilistic_hp can fall through
    to the next candidate if a field turns out to produce zero record pairs
    — e.g. a normally-good field like last_name can have no duplicate
    values at all on a small or narrowly cohort-filtered dataset.
    """
    active_blocking = [f for f, v in (blocking_toggles or {}).items()
                        if v and "+" not in f and f in selected_fields]
    NAME_HINTS = ["first_name", "surname", "last_name", "dob", "birth_year",
                  "city", "res_city_desc", "email", "gender", "gender_code",
                  "postcode", "zip_code"]
    candidates = active_blocking or list(selected_fields)
    candidates = sorted(
        candidates,
        key=lambda f: (NAME_HINTS.index(f) if f in NAME_HINTS else len(NAME_HINTS), f),
    )
    return candidates or list(selected_fields)


_CPU_PATCH_LOCK = threading.Lock()


@contextmanager
def _at_least_two_cpus():
    """Make ``multiprocessing.cpu_count()`` report >= 2 while Splink samples u-probabilities.

    Splink splits that sampling into one salted partition per CPU and raises
    ``Salting partitions must be specified and > 1`` on a 1-CPU host (common on
    small hosted instances). ``cpu_count`` is process-global, so the patch is
    applied only when needed and under a lock - concurrent sessions cannot
    race on restoring it.
    """
    if (os.cpu_count() or 1) >= 2:
        yield
        return
    with _CPU_PATCH_LOCK:
        original = multiprocessing.cpu_count
        multiprocessing.cpu_count = lambda: 2
        try:
            yield
        finally:
            multiprocessing.cpu_count = original


def _train_probabilistic_hp(linker, selected_fields, hp, blocking_toggles: dict = None) -> None:
    """Train probabilistic model using user-supplied hyperparameters.

    Ranks candidate blocking fields (see _select_training_blocking_fields)
    and tries them in order at every training step, skipping any field
    whose blocking rule produces zero record pairs instead of letting
    Splink's hard error ("... resulted in no record pairs") crash the whole
    run. This happens most often on small or heavily cohort-filtered
    datasets where a normally-good field has no duplicate values present.
    """
    recall = hp.get("recall_estimate", 0.6)      # User-adjustable recall for prior
    candidates = _select_training_blocking_fields(selected_fields, blocking_toggles)

    def _is_empty_blocking_error(exc: Exception) -> bool:
        return "no record pairs" in str(exc).lower()

    # ── Step 1: Prior estimate — try each candidate until one works ──────────
    prior_field = None
    for field in candidates:
        try:
            linker.training.estimate_probability_two_random_records_match(
                [brl.block_on(field)], recall=recall
            )
            prior_field = field
            break
        except Exception as e:
            if not _is_empty_blocking_error(e):
                raise
            continue

    if prior_field is None:
        raise ValueError(
            "Could not train a probabilistic model: none of the selected blocking "
            f"fields ({', '.join(candidates)}) produced any matching record pairs "
            "in this dataset. This usually means the cohort is too small or too "
            "narrowly filtered for these fields to have duplicate values. Try "
            "enabling a broader blocking field, selecting additional fields, or "
            "widening the cohort filter."
        )

    # ── Step 2: u-probabilities ──────────────────────────────────────────────
    with _at_least_two_cpus():
        linker.training.estimate_u_using_random_sampling(1e5, seed=42)

    # ── Step 3: EM training — primary field, then up to one more that works ───
    em_order = [prior_field] + [f for f in candidates if f != prior_field]
    trained = 0
    for field in em_order:
        try:
            linker.training.estimate_parameters_using_expectation_maximisation(
                brl.block_on(field), fix_u_probabilities=True
            )
            trained += 1
            if trained >= 2:   # primary + one secondary pass is enough
                break
        except Exception as e:
            if not _is_empty_blocking_error(e):
                raise
            continue


# (Stale _build_blocking_rules removed: now handled by the generic version above which uses brl.block_on(key) for ANY field name)


# =============================================================================
# ── SAVE MODEL AS JSON ────────────────────────────────────────────────────────
# Reconstruct a full Splink model JSON from settings + extracted model params.
# The output is accepted by run_linkage_from_json() and the advanced flow.
# =============================================================================

def reconstruct_model_json(settings_used: dict, model_params: dict,
                            linkage_type: str = "probabilistic") -> dict:
    """Build a Splink model JSON from settings_used + trained model_params.

    Takes the settings dict returned by run_linkage() and the model_params
    dict from _extract_model_params(), and injects the trained m/u probabilities
    back into the comparison levels so the JSON can be used to skip training.

    Works for both probabilistic (m/u populated) and deterministic runs
    (m/u will be null in the output, which is valid for deterministic replay).

    linkage_type: the mode this model was trained under ('deterministic' or
    'probabilistic'). Stamped into the JSON as "_app_linkage_type" so that
    re-uploading this file in Advanced mode auto-selects the same matching
    methodology instead of always assuming probabilistic — this is what
    keeps the deterministic flow consistent across all three app flows.

    Returns a dict that json.dumps() can serialise directly.
    """
    model = copy.deepcopy(settings_used)    # Never mutate the original
    model["_app_linkage_type"] = linkage_type   # Round-trip marker, read back in Advanced mode

    # ── Inject prior probability ───────────────────────────────────────────────
    prior_log_odds = model_params.get("prior_log_odds")
    if prior_log_odds is not None:
        try:
            # Convert log-odds back to probability: P = 2^W / (1 + 2^W)
            prob = 2 ** prior_log_odds / (1 + 2 ** prior_log_odds)
            model["probability_two_random_records_match"] = round(prob, 8)
        except Exception:
            pass

    # ── Inject m/u probabilities into each comparison level ──────────────────
    # Build a field → levels list lookup from model_params
    params_by_field = {
        comp["field"]: comp.get("levels", [])
        for comp in model_params.get("comparisons", [])
        if "field" in comp
    }

    for comp_dict in model.get("comparisons", []):
        field       = comp_dict.get("output_column_name", "")
        lvl_params  = params_by_field.get(field, [])    # May be empty for deterministic

        for j, lvl in enumerate(comp_dict.get("comparison_levels", [])):
            if j < len(lvl_params):
                lp = lvl_params[j]
                # Only inject if the value is not None (null levels stay null)
                if lp.get("m_prob") is not None:
                    lvl["m_probability"] = lp["m_prob"]
                if lp.get("u_prob") is not None:
                    lvl["u_probability"] = lp["u_prob"]

    return model
