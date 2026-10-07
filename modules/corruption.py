"""Controlled data corruption for building test "Dataset B" files.

To evaluate a linkage model you need pairs of files that describe the same
people but disagree the way real files do.  :func:`make_noisy_copy` takes a
clean table, samples some of its rows and damages chosen fields at a chosen
rate.  The sample keeps a ``cluster`` ground-truth column so that accuracy can
be measured afterwards.

Corruption is driven by a field's *kind*, not its name:

==========  ===========================================================
kind        what a corrupted value looks like
==========  ===========================================================
text        ``n`` letters replaced (names, free text)
date        shifted by up to ``date_shift_days``, day/month swapped, or
            blanked (ISO ``YYYY-MM-DD`` strings)
year        shifted by +/- ``year_shift`` (bare four-digit years)
email       dots removed, digit appended, or domain swapped
location    abbreviated to initials / three-letter upper-case tag
postcode    one character changed, or whitespace removed
category    replaced by a *different value observed in the same column*,
            so coded fields never acquire values outside their code set
==========  ===========================================================

Everything is driven by a seeded ``random.Random`` so a run is reproducible.
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass
from datetime import timedelta
from typing import Mapping, Optional, Sequence

import pandas as pd

BOOKKEEPING_COLUMNS = frozenset({"unique_id", "source_dataset", "cluster"})

#: Semantic types (from ``eda_engine.run_full_eda``) with one fixed corruption kind.
_KIND_BY_FIELD_TYPE = {
    "first_name": "text", "surname": "text", "full_name": "text",
    "email": "email", "postcode": "postcode", "location": "location",
}
_FIXED_KIND_TYPES = frozenset(_KIND_BY_FIELD_TYPE)

_MAX_CATEGORY_LEVELS = 12
_YEAR_PATTERN = re.compile(r"^(18|19|20)\d{2}$")


# =============================================================================
# Per-value corruptions (each takes a non-null value and returns a new one)
# =============================================================================

def corrupt_text(value: str, rng: random.Random, n_edits: int = 1) -> str:
    """Replace ``n_edits`` alphabetic characters with a different letter of the same case."""
    chars = list(str(value))
    positions = [i for i, ch in enumerate(chars) if ch.isalpha()]
    if not positions:
        return str(value)
    for pos in rng.sample(positions, min(max(n_edits, 0), len(positions))):
        alphabet = string.ascii_uppercase if chars[pos].isupper() else string.ascii_lowercase
        chars[pos] = rng.choice([c for c in alphabet if c.lower() != chars[pos].lower()])
    return "".join(chars)


def corrupt_date(value: str, rng: random.Random, max_shift_days: int = 1) -> Optional[str]:
    """Shift, day/month-swap or blank an ISO date string."""
    parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        return value
    roll = rng.random()
    if roll < 0.5:
        shift = rng.choice([-1, 1]) * rng.randint(1, max(max_shift_days, 1))
        return (parsed + timedelta(days=shift)).strftime("%Y-%m-%d")
    if roll < 0.8 and parsed.day <= 12 and parsed.day != parsed.month:
        return parsed.replace(month=parsed.day, day=parsed.month).strftime("%Y-%m-%d")
    return None


def corrupt_year(value: str, rng: random.Random, max_shift: int = 1) -> str:
    try:
        return str(int(str(value).strip()) + rng.choice([-1, 1]) * max(max_shift, 1))
    except ValueError:
        return str(value)


def corrupt_email(value: str, rng: random.Random) -> str:
    """Typical address drift: separator style, a tag, a trailing digit, a new provider."""
    text = str(value)
    if "@" not in text:
        return text + str(rng.randint(1, 9))
    local, domain = text.split("@", 1)
    return rng.choice([
        local.replace(".", "_") + "@" + domain,
        f"{local}+{rng.choice(['news', 'shop', 'home'])}@{domain}",
        f"{local.rstrip('0123456789')}{rng.randint(1, 99)}@{domain}",
        local + "@" + rng.choice(["gmail.com", "outlook.com", "yahoo.com"]),
    ])


def corrupt_location(value: str, rng: random.Random) -> str:
    """Abbreviate a place name: ``New Haven`` -> ``N. Haven``, ``Bristol`` -> ``Bris.``"""
    text = str(value)
    words = text.split()
    if len(words) > 1:
        return f"{words[0][0].upper()}. {' '.join(words[1:])}"
    return text[:4] + "." if len(text) > 5 else text.swapcase()


def corrupt_postcode(value: str, rng: random.Random) -> str:
    text = str(value)
    if " " in text and rng.random() < 0.5:
        return text.replace(" ", "")
    return corrupt_text(text, rng, 1) if any(c.isalpha() for c in text) else text[:-1] + str(rng.randint(0, 9))


# =============================================================================
# Field classification
# =============================================================================

def classify_field(series: pd.Series, field_type: Optional[str] = None) -> str:
    """Choose a corruption kind for a column from its values (and optional semantic type)."""
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return "text"

    # A semantic type that names a fixed corruption (email, postcode, ...) wins
    # outright: a name column in a tiny cohort must not be mistaken for a code.
    if field_type in _FIXED_KIND_TYPES:
        return _KIND_BY_FIELD_TYPE[field_type]

    sample = values if len(values) <= 500 else values.sample(500, random_state=0)
    # Bare years are tested before dates: pandas parses "1999" as a full date.
    if sample.str.match(_YEAR_PATTERN).mean() > 0.5:
        return "year"
    looks_dated = sample.str.contains(r"[-/]", regex=True).mean() > 0.5
    if looks_dated and pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() > 0.5:
        return "date"
    if field_type == "gender" or values.nunique() <= _MAX_CATEGORY_LEVELS:
        return "category"
    return "text"


# =============================================================================
# Public API
# =============================================================================

@dataclass(frozen=True)
class CorruptionSettings:
    """Magnitude controls shared by every corrupted field."""
    text_edits: int = 1          # letters changed per corrupted text value
    year_shift: int = 1          # +/- years for year-only fields
    date_shift_days: int = 1     # maximum day shift for full dates


DEFAULT_SETTINGS = CorruptionSettings()


def make_noisy_copy(
    df: pd.DataFrame,
    field_rates: Mapping[str, float],
    *,
    sample_frac: float = 0.5,
    seed: int = 42,
    field_types: Optional[Mapping[str, str]] = None,
    missing_rates: Optional[Mapping[str, float]] = None,
    settings: CorruptionSettings = DEFAULT_SETTINGS,
    id_columns: Sequence[str] = ("unique_id",),
) -> pd.DataFrame:
    """Return a sampled, damaged copy of ``df`` labelled ``source_dataset = "B"``.

    Args:
        df: clean source table (Dataset A).
        field_rates: ``{column: probability}`` that a non-null value in that
            column is corrupted.  ``1.0`` corrupts every populated row.
        sample_frac: fraction of ``df`` rows to keep.
        seed: seeds both the row sample and the corruption.
        field_types: optional ``{column: semantic type}`` used to pick a kind.
        missing_rates: ``{column: probability}`` that a value is blanked out.
        settings: magnitudes (letters changed, year shift, ...).
        id_columns: columns that receive a ``_B`` suffix so B ids never
            collide with A ids.

    The result's ``cluster`` column is Dataset A's ``cluster`` if it has one,
    otherwise A's ``unique_id`` - so a B record and the A record it was copied
    from always share a cluster value (the ground truth for accuracy metrics).
    """
    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be in (0, 1]")

    rng = random.Random(seed)
    sample = df.sample(frac=sample_frac, random_state=seed).copy()

    if "cluster" not in sample.columns:
        ids = sample["unique_id"] if "unique_id" in sample.columns else pd.Series(sample.index, index=sample.index)
        sample["cluster"] = ids.astype(str)

    for column, rate in field_rates.items():
        if column in BOOKKEEPING_COLUMNS or column not in sample.columns or rate <= 0:
            continue
        kind = classify_field(df[column], (field_types or {}).get(column))
        observed = df[column].dropna().unique().tolist() if kind == "category" else []
        sample[column] = sample[column].astype(object)
        for idx in sample.index[sample[column].notna()]:
            if rng.random() >= rate:
                continue
            sample.at[idx, column] = _corrupt_value(sample.at[idx, column], kind, rng, settings, observed)

    for column, rate in (missing_rates or {}).items():
        if column in BOOKKEEPING_COLUMNS or column not in sample.columns or rate <= 0:
            continue
        sample[column] = sample[column].astype(object)
        for idx in sample.index[sample[column].notna()]:
            if rng.random() < rate:
                sample.at[idx, column] = None

    for column in id_columns:
        if column in sample.columns:
            sample[column] = sample[column].astype(str).map(
                lambda v: v if v.endswith("_B") else v + "_B")

    sample["source_dataset"] = "B"
    return sample


def _corrupt_value(value, kind: str, rng: random.Random, settings: CorruptionSettings, observed: list):
    if kind == "date":
        return corrupt_date(value, rng, settings.date_shift_days)
    if kind == "year":
        return corrupt_year(value, rng, settings.year_shift)
    if kind == "email":
        return corrupt_email(value, rng)
    if kind == "location":
        return corrupt_location(value, rng)
    if kind == "postcode":
        return corrupt_postcode(value, rng)
    if kind == "category":
        alternatives = [v for v in observed if str(v) != str(value)]
        return rng.choice(alternatives) if alternatives else value
    return corrupt_text(value, rng, settings.text_edits)
