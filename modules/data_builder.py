"""Cached dataset loaders for the built-in demos (no network, no real records)."""

from __future__ import annotations


from modules.eda_engine import run_full_eda
from modules.synthetic_data import DEFAULT_SEED, build_demo_datasets as _build_demo_datasets
from modules.voter_data import BLOCKING_FIELDS, LINKAGE_FIELDS, build_voter_datasets


def build_datasets(n_records: int = 1000, seed: int = DEFAULT_SEED) -> tuple:
    """``(base, dataset_a, dataset_b)`` for the people demo."""
    return _build_demo_datasets(n_records=n_records, seed=seed)


def load_voter_datasets(n_rows: int = 1200, seed: int = DEFAULT_SEED) -> tuple:
    """Voter-registry Dataset A and Dataset B, cleaned: ``(a, b, field_types, eda_log)``.

    Both frames carry ``unique_id`` (the voter's ``ncid``), ``source_dataset`` and the
    ``cluster`` ground truth. B is a damaged sample of A, so every B record has a true match.
    """
    raw_a, raw_b = build_voter_datasets(n_rows, seed)
    a, field_types, _, eda_log = run_full_eda(raw_a)
    b, _, _, _ = run_full_eda(raw_b)
    for frame, source in ((a, "A"), (b, "B")):
        frame["unique_id"] = frame["ncid"].astype(str)
        frame["source_dataset"] = source
    return a, b, field_types, eda_log


def voter_linkage_defaults(columns) -> tuple:
    """Default comparison fields and blocking toggles for the voter registry."""
    fields = [f for f in LINKAGE_FIELDS if f in columns]
    return fields, {f: f in BLOCKING_FIELDS for f in fields}
