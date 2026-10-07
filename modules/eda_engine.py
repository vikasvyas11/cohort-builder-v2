"""Automated cleaning, profiling and field-type inference for uploaded data.

``run_full_eda`` is the single entry point used by the Upload flow and by the
voter-registry loader.  The individual steps are public so they can be unit
tested and reused.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

# Text spellings of "missing" that should become real nulls.
_NULL_TOKENS = ["nan", "NAN", "None", "", "null", "NULL", "<NA>", "NA", "N/A", "n/a", " "]

# Bookkeeping columns that are never compared as ordinary fields.
_BOOKKEEPING = ("unique_id", "cluster", "source_dataset")

DEFAULT_CORRELATION_THRESHOLD = 0.95


# =============================================================================
# Cleaning steps
# =============================================================================

def clean_field_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Lower-case column names, replacing anything but ``[a-z0-9_]`` with ``_``.

    Returns the renamed frame and ``{old_name: new_name}`` for names that changed.
    """
    renames = {}
    for col in df.columns:
        clean = re.sub(r"[^a-z0-9_]", "_", str(col).strip().lower())
        clean = re.sub(r"_+", "_", clean).strip("_")
        if clean != col:
            renames[col] = clean
    return df.rename(columns=renames), renames


def remove_null_elements(df: pd.DataFrame, min_populated: int = 3) -> tuple[pd.DataFrame, dict]:
    """Drop all-null columns, then rows that carry too little information.

    Rows are removed in three passes so the log shows how many fell into each
    bucket: rows with no values, rows with exactly one, and rows with fewer than
    ``min_populated`` (default 3, so rows with exactly two). A row needs at least
    ``min_populated`` populated values to stay.
    """
    initial_rows = len(df)
    null_columns = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=null_columns)

    df = df.dropna(how="all")
    after_empty = len(df)
    df = df.dropna(thresh=min(2, min_populated))
    after_one = len(df)
    df = df.dropna(thresh=min_populated)

    return df, {
        "null_columns_dropped": null_columns,
        "all_empty_rows": initial_rows - after_empty,
        "one_value_rows": after_empty - after_one,
        "two_value_rows": after_one - len(df),
        "partial_null_removed": after_empty - len(df),
    }


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Trim and collapse whitespace; turn textual null markers into real nulls."""
    out = df.copy()
    for col in out.columns:
        try:
            out[col] = (out[col].astype(str).str.strip()
                        .str.replace(r"\s+", " ", regex=True)
                        .replace(_NULL_TOKENS, np.nan))
        except (TypeError, ValueError):
            continue    # leave columns that cannot be treated as text untouched
    return out


def remove_duplicate_records(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows that are exact duplicates of an earlier row."""
    deduped = df.drop_duplicates()
    return deduped, len(df) - len(deduped)


def _looks_like_full_date(series: pd.Series) -> bool:
    """True if most sampled values contain a date separator (``-``, ``/`` or space).

    A bare year such as ``1975`` does not count: ``pd.to_datetime`` would turn
    it into ``1975-01-01`` and silently fabricate a day and month.
    """
    sample = series.dropna().astype(str).str.strip()
    sample = sample[sample != ""].head(50)
    return (not sample.empty) and sample.str.contains(r"[-/ ]", regex=True).mean() > 0.5


_SLASH_DATE = re.compile(r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-]\d{2,4}")


def parse_mixed_dates(series: pd.Series) -> pd.Series:
    """Parse a column whose values may use different date formats.

    ``pd.to_datetime`` normally infers *one* format from the first value and
    turns every value that disagrees into NaT, so a column mixing
    ``"04/02/1975"`` and ``"1980-01-31"`` would silently lose rows.  Here each
    value is parsed on its own (``format="mixed"``).

    Dates like ``04/02/1975`` are ambiguous.  They are read month-first unless
    the column itself proves otherwise: a first component above 12 (e.g.
    ``25/12/1990``) can only be a day, so the whole column is read day-first.
    """
    parts = series.dropna().astype(str).str.extract(_SLASH_DATE).astype(float)
    day_first = bool((parts[0] > 12).any() and not (parts[1] > 12).any())
    return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=day_first)


def standardise_date_formats(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rewrite date-like columns as ``YYYY-MM-DD`` strings.

    Only columns whose name mentions ``date``, ``dob`` or ``birth`` *and* whose
    values look like full dates are touched, so year-only fields such as
    ``birth_year`` are left alone.
    """
    out = df.copy()
    converted = {}
    candidates = [c for c in out.columns
                  if any(tag in c for tag in ("date", "dob", "birth")) and _looks_like_full_date(out[c])]
    for col in candidates:
        parsed = parse_mixed_dates(out[col])
        if parsed.notna().any():
            out[col] = parsed.dt.strftime("%Y-%m-%d")
            converted[col] = "YYYY-MM-DD"
    return out, converted


# =============================================================================
# Field typing and suggestions
# =============================================================================

def infer_field_types(columns, id_col: Optional[str] = None) -> dict:
    """Guess a semantic type for each column from its name.

    Types: ``id, first_name, surname, full_name, dob, email, postcode, gender,
    location, text``.  Order matters: the first matching rule wins.
    """
    rules = [
        (("first", "given"), "first_name"),
        (("last", "sur", "family"), "surname"),
        (("name",), "full_name"),
        (("date", "dob", "birth"), "dob"),
        (("email",), "email"),
        (("post", "zip"), "postcode"),
        (("gender", "sex"), "gender"),
        (("city", "town", "county"), "location"),
    ]
    types = {}
    for col in columns:
        if col == "unique_id" or col == id_col:
            types[col] = "id"
            continue
        if "birth" in col and ("year" in col or "state" in col):
            types[col] = "text"                       # a birth year/state is not a full date of birth
            continue
        types[col] = next((t for tags, t in rules if any(tag in col for tag in tags)), "text")
    return types


def run_full_eda(df: pd.DataFrame, id_col: Optional[str] = None) -> tuple:
    """Run the whole cleaning pipeline.

    Returns ``(clean_df, field_types, summary, log)`` where ``summary`` holds
    row/column counts and ``log`` the per-step details shown in the UI.
    """
    original_rows = len(df)
    log: dict = {}

    work, renamed = clean_field_names(df)
    log["field_names"] = {"changed": renamed}

    work, nulls = remove_null_elements(work)
    log["null_columns_dropped"] = nulls["null_columns_dropped"]
    log["null_rows_removed"] = {k: nulls[k] for k in ("all_empty_rows", "one_value_rows", "two_value_rows")}

    work = clean_text_fields(work)
    work, dates = standardise_date_formats(work)
    log["dates_standardised"] = dates

    work, n_duplicates = remove_duplicate_records(work)
    log["duplicates_removed"] = n_duplicates

    log["summary"] = {
        "original_rows": original_rows,
        "rows_removed": original_rows - len(work),
        "final_rows": len(work),
        "cols_removed": len(nulls["null_columns_dropped"]),
    }

    field_types = infer_field_types(work.columns, renamed.get(id_col, id_col) if id_col else None)
    return work, field_types, log["summary"], log


def find_high_correlation_pairs(
    df: pd.DataFrame,
    id_cols: list,
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> list:
    """Column pairs whose values agree row-for-row at least ``threshold`` of the time.

    Near-duplicate columns add no linkage evidence and double-count what they
    do add.  Returns ``[(col_a, col_b, agreement), ...]``, most-agreeing first.
    """
    features = [c for c in df.columns if c not in id_cols and c not in _BOOKKEEPING]
    pairs = []
    for i, col_a in enumerate(features):
        for col_b in features[i + 1:]:
            agreement = (df[col_a] == df[col_b]).mean()
            if agreement >= threshold:
                pairs.append((col_a, col_b, float(agreement)))
    return sorted(pairs, key=lambda p: p[2], reverse=True)


_COMPARISON_BY_TYPE = {
    "first_name": "JaroWinklerAtThresholds",
    "surname": "JaroAtThresholds",
    "dob": "DateOfBirthComparison",
    "email": "EmailComparison",
    "postcode": "PostcodeComparison",
}
_BLOCKING_TYPES = ("first_name", "surname", "dob", "postcode", "location")


def suggest_comparison_types(field_types: dict) -> dict:
    """Suggested Splink comparison per field (``ExactMatch`` when nothing better fits)."""
    return {col: _COMPARISON_BY_TYPE.get(ftype, "ExactMatch")
            for col, ftype in field_types.items() if ftype != "id"}


def is_selective_field(series: pd.Series, min_distinct_ratio: float = 0.05) -> bool:
    """True if a column has enough distinct values to be worth blocking on.

    Blocking on a column with few distinct values (gender, a handful of
    cities) pairs up a large share of the dataset. ``min_distinct_ratio`` is the
    smallest ``distinct values / populated values`` that counts as selective.
    """
    populated = series.dropna()
    return len(populated) > 0 and populated.nunique() / len(populated) >= min_distinct_ratio


def suggest_blocking_rules(field_types: dict) -> dict:
    """Pre-tick stable, fairly selective fields as blocking-rule candidates."""
    return {col: ftype in _BLOCKING_TYPES for col, ftype in field_types.items()}
