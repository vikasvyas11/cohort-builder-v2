"""Street address and phone number columns for the synthetic voter registry.

The registry the app builds in memory has no ``res_street_address`` or ``full_phone_number``: they are the two
most identifying columns of the real voter file, and the two most useful for linking. This module adds them
for the downloadable upload samples (``tools/make_upload_samples.py``).

Nothing here comes from a person. ``modules/nc_voter_extras.json`` (built by ``tools/build_voter_extras.py``)
holds only shapes: street names shared by hundreds of addresses, how house numbers are spread, how often an
address has a unit, how often a phone is present and which area codes occur. Addresses are a random house
number on one of those street names. Phone numbers are random digits in the file's format (ten digits, no
separators) under a real area code, so they are not numbers taken from the file.
"""

from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from modules.voter_data import DEFAULT_SEED, _draw

EXTRAS_PATH = Path(__file__).with_name("nc_voter_extras.json")
UNIT_WORDS = ("APT", "UNIT", "LOT")
SUFFIX_FORMS = {" ST": " STREET", " DR": " DRIVE", " RD": " ROAD", " LN": " LANE", " AVE": " AVENUE", " CT": " COURT"}


@lru_cache(maxsize=1)
def load_extras() -> dict:
    if not EXTRAS_PATH.is_file():
        raise FileNotFoundError(f"{EXTRAS_PATH.name} is missing. Rebuild it with: "
                                "python tools/build_voter_extras.py <ncvoter_Statewide.txt>")
    return json.loads(EXTRAS_PATH.read_text(encoding="utf-8"))


def _addresses(n: int, rng: np.random.Generator, py: random.Random) -> list:
    extras = load_extras()
    bins = [tuple(int(x) for x in label.split("-")) for label in extras["house_number_bins"]]
    weights = np.array(list(extras["house_number_bins"].values()), dtype=float)
    chosen = rng.choice(len(bins), size=n, p=weights / weights.sum())
    streets = _draw(rng, extras["street_names"], n)
    out = []
    for k, street in zip(chosen, streets):
        low, high = bins[k]
        address = f"{py.randint(low, high)} {street}"
        if py.random() < extras["share_with_unit"]:
            address += f" {py.choice(UNIT_WORDS)} {py.randint(1, 40)}"
        out.append(address)
    return out


def _phones(n: int, rng: np.random.Generator, py: random.Random) -> list:
    extras = load_extras()
    areas = _draw(rng, extras["area_codes"], n)
    present = rng.random(n) < extras["share_with_phone"]
    return [f"{area}{py.randint(200, 999)}{py.randint(0, 9999):04d}" if has else None for area, has in zip(areas, present)]


def add_address_and_phone(registry: pd.DataFrame, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Return ``registry`` with ``res_street_address`` and ``full_phone_number`` added.

    Registrations of the same voter (same ``cluster``) share a phone number and, unless the second registration is in
    a different city (the voter moved), an address, so duplicates look like one person who re-registered.
    """
    rng, py = np.random.default_rng(seed + 1), random.Random(seed + 1)
    out = registry.copy()
    out["res_street_address"] = _addresses(len(out), rng, py)
    out["full_phone_number"] = _phones(len(out), rng, py)
    if "cluster" in out.columns:
        first = out.groupby("cluster").cumcount() == 0
        anchor = out[first].set_index("cluster")
        same_voter = ~first
        moved = out["res_city_desc"].values != out["cluster"].map(anchor["res_city_desc"]).values
        keep_address = same_voter & ~pd.Series(moved, index=out.index)
        out.loc[same_voter, "full_phone_number"] = out.loc[same_voter, "cluster"].map(anchor["full_phone_number"])
        out.loc[keep_address, "res_street_address"] = out.loc[keep_address, "cluster"].map(anchor["res_street_address"])
    # place the new columns beside the fields they belong with
    columns = [c for c in out.columns if c not in ("res_street_address", "full_phone_number")]
    columns.insert(columns.index("res_city_desc"), "res_street_address")
    columns.insert(columns.index("zip_code") + 1, "full_phone_number")
    return out[columns]


def damage_address_and_phone(damaged: pd.DataFrame, seed: int = DEFAULT_SEED, address_rate: float = 0.12,
                             phone_rate: float = 0.10, missing_address: float = 0.05, missing_phone: float = 0.15) -> pd.DataFrame:
    """Dataset B's address and phone as a second source would record them: a unit dropped, a street type spelled out,
    a letter wrong, a phone digit changed, some values missing."""
    py = random.Random(seed + 2)
    out = damaged.copy()
    addresses = out["res_street_address"].tolist()
    for i, value in enumerate(addresses):
        if not isinstance(value, str):
            continue
        roll = py.random()
        if roll < missing_address:
            addresses[i] = None
        elif roll < missing_address + address_rate:
            kind = py.choice(("unit", "suffix", "typo"))
            without_unit = re.sub(r"\s+(?:APT|UNIT|LOT)\s+\d+$", "", value)
            if kind == "unit" and without_unit != value:
                addresses[i] = without_unit
            elif kind == "suffix" and any(without_unit.endswith(s) for s in SUFFIX_FORMS):
                short = next(s for s in SUFFIX_FORMS if without_unit.endswith(s))
                addresses[i] = without_unit[: -len(short)] + SUFFIX_FORMS[short]
            else:
                letters = [j for j, ch in enumerate(value) if ch.isalpha()]
                if letters:
                    j = py.choice(letters)
                    addresses[i] = value[:j] + py.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + value[j + 1:]
    out["res_street_address"] = addresses
    phones = out["full_phone_number"].tolist()
    for i, value in enumerate(phones):
        if not isinstance(value, str):
            continue
        roll = py.random()
        if roll < missing_phone:
            phones[i] = None
        elif roll < missing_phone + phone_rate:
            phones[i] = value[:-1] + str((int(value[-1]) + py.randint(1, 9)) % 10)
    out["full_phone_number"] = phones
    return out
