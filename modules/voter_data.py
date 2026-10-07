"""Synthetic North Carolina-style voter registry, with a damaged second file for linking.

Real voter files list named individuals, so none of their records are used here.
Instead :mod:`tools.build_voter_profile` reduces the statewide file to aggregate
counts (``nc_voter_profile.json``: demographic mix, birth years, counties, cities,
zip codes, and only names shared by hundreds of voters) and this module samples
brand-new voters from those distributions. The result *looks* like the registry
statistically - same race/party/gender/age mix, same places, same common names -
but no row corresponds to a real person.

* :func:`generate_voter_registry` - Dataset A: one row per registration, with a few
  voters registered twice (the realistic duplicate).
* :func:`build_voter_datasets` - Dataset A plus Dataset B, a damaged sample of A,
  both carrying a ``cluster`` ground-truth label.
"""

from __future__ import annotations

import json
import random
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from modules.corruption import CorruptionSettings, make_noisy_copy

PROFILE_PATH = Path(__file__).with_name("nc_voter_profile.json")
DEFAULT_SEED = 42
REFERENCE_YEAR = 2026
DUPLICATE_RATE = 0.05          # share of voters registered twice

#: Fields the linkage uses by default (identifiers and demographics are excluded).
LINKAGE_FIELDS = ["first_name", "last_name", "middle_name", "dob", "gender_code",
                  "res_city_desc", "zip_code", "birth_state"]
BLOCKING_FIELDS = ["first_name", "last_name", "dob"]
LARGE_DATASET_ROWS = 5_000


def blocking_defaults(n_rows: int) -> list:
    """Default blocking rules for a registry of ``n_rows``.

    Single names are fine at 1,200 rows but pair up millions of records at 50,000 (the 2,000
    first names alone give about 4.3M pairs), so larger registries default to date of birth
    plus two name/place combinations that stay in the tens of thousands of pairs.
    """
    return BLOCKING_FIELDS if n_rows <= LARGE_DATASET_ROWS else ["dob", "first_name+last_name", "last_name+zip_code"]

#: How Dataset B is damaged by default (per populated value).
B_ERROR_RATES = {"first_name": 0.10, "last_name": 0.08, "middle_name": 0.10, "dob": 0.06,
                 "res_city_desc": 0.08, "zip_code": 0.12, "gender_code": 0.03}
B_MISSING_RATES = {"middle_name": 0.15, "birth_state": 0.10, "zip_code": 0.05}


@lru_cache(maxsize=1)
def load_profile() -> dict:
    """The aggregate statistics the generator samples from."""
    if not PROFILE_PATH.is_file():
        raise FileNotFoundError(
            f"{PROFILE_PATH.name} is missing. Rebuild it with: "
            "python tools/build_voter_profile.py <ncvoter_Statewide.txt>")
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _draw(rng: np.random.Generator, counts: dict, n: int) -> np.ndarray:
    keys = list(counts)
    weights = np.array([counts[k] for k in keys], dtype=float)
    return rng.choice(keys, size=n, p=weights / weights.sum())


def _entities(n: int, rng: np.random.Generator, py: random.Random) -> pd.DataFrame:
    """``n`` new voters sampled from the profile."""
    p = load_profile()
    gender = _draw(rng, {g: p["gender_code"].get(g, 0) for g in ("F", "M", "U") if p["gender_code"].get(g)}, n)

    pools = p["first_names"]
    mixed = {**pools["F"], **{k: pools["F"].get(k, 0) + v for k, v in pools["M"].items()}}
    first = np.array([""] * n, dtype=object)
    middle = np.array([None] * n, dtype=object)
    for g, source in (("F", pools["F"]), ("M", pools["M"]), ("U", mixed)):
        idx = np.where(gender == g)[0]
        if len(idx):
            first[idx] = _draw(rng, source, len(idx))
            has_middle = rng.random(len(idx)) < 0.65
            middle[idx[has_middle]] = _draw(rng, source, int(has_middle.sum()))

    years = {int(y): c for y, c in p["birth_year"].items() if y.isdigit() and 1908 <= int(y) <= REFERENCE_YEAR - 18}
    birth_year = _draw(rng, years, n).astype(int)
    dob = [date(int(y), py.randint(1, 12), py.randint(1, 28)).isoformat() for y in birth_year]

    race = _draw(rng, p["race_code"], n)
    county = _draw(rng, p["county_desc"], n)

    def draw_by_group(groups: np.ndarray, pools, fallback) -> np.ndarray:
        """One draw per row from that row's group's pool, batched per group (fast at 50k rows)."""
        out = np.empty(n, dtype=object)
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            pool = pools(g)
            out[idx] = _draw(rng, pool, len(idx)) if pool else [fallback(g) for _ in idx]
        return out

    party = draw_by_group(race, lambda r: p["party_given_race"].get(r) or p["party_cd"], None)
    city = draw_by_group(county, lambda c: p["cities_by_county"].get(c), lambda c: c)
    zip_code = draw_by_group(city, lambda c: p["zips_by_city"].get(c), lambda c: str(py.randint(27006, 28909)))

    reg_years = {int(y): c for y, c in p["registr_year"].items() if y.isdigit() and 1950 <= int(y) <= REFERENCE_YEAR}
    reg_year = np.maximum(_draw(rng, reg_years, n).astype(int), birth_year + 18)
    registered = [f"{py.randint(1, 12):02d}/{py.randint(1, 28):02d}/{min(y, REFERENCE_YEAR)}" for y in reg_year]

    birth_state = _draw(rng, p["birth_state"], n).astype(object)
    birth_state[rng.random(n) < 0.06] = None

    return pd.DataFrame({
        "last_name": _draw(rng, p["last_names"], n), "first_name": first, "middle_name": middle,
        "dob": dob, "birth_year": birth_year.astype(str), "age_at_year_end": (REFERENCE_YEAR - birth_year).astype(str),
        "gender_code": gender, "race_code": race, "ethnic_code": _draw(rng, p["ethnic_code"], n),
        "party_cd": party, "birth_state": birth_state, "county_desc": county, "res_city_desc": city,
        "zip_code": zip_code, "registr_dt": registered, "status_cd": _draw(rng, p["status_cd"], n),
    })


def _swap_letter(word: str, py: random.Random) -> str:
    pos = py.randrange(len(word))
    return word[:pos] + py.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + word[pos + 1:]


def generate_voter_registry(n_rows: int = 1200, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Dataset A: about ``n_rows`` registrations, a few voters appearing twice.

    Columns: ``voter_reg_num, ncid`` (unique per registration), name, ``dob`` and the
    demographic, place and registration fields of the NC file (no street addresses,
    phone numbers or ids of real people), plus ``cluster`` - the voter's identity, shared
    by their duplicate registrations (the ground truth for accuracy).
    """
    rng, py = np.random.default_rng(seed), random.Random(seed)
    n_dups = int(n_rows * DUPLICATE_RATE)
    voters = _entities(n_rows - n_dups, rng, py)
    voters["cluster"] = [f"V{i:07d}" for i in range(len(voters))]

    dups = voters.sample(n_dups, random_state=seed).copy()
    for idx in dups.index:                                   # a second registration drifts a little
        if py.random() < 0.25:
            dups.at[idx, "first_name"] = _swap_letter(dups.at[idx, "first_name"], py)
        if py.random() < 0.40:
            dups.at[idx, "middle_name"] = None
        if py.random() < 0.30:                               # moved: new city and zip within the county
            p = load_profile()
            cities = p["cities_by_county"].get(dups.at[idx, "county_desc"])
            if cities:
                city = _draw(rng, cities, 1)[0]
                dups.at[idx, "res_city_desc"] = city
                zips = p["zips_by_city"].get(city)
                dups.at[idx, "zip_code"] = _draw(rng, zips, 1)[0] if zips else dups.at[idx, "zip_code"]

    registry = pd.concat([voters, dups], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    registry.insert(0, "ncid", [f"{py.choice('ABCDEFGHJKLMNPRSTVWXYZ')}{py.choice('ABCDEFGHJKLMNPRSTVWXYZ')}{i:06d}"
                                for i in rng.permutation(len(registry))])
    registry.insert(0, "voter_reg_num", [str(100000 + i) for i in rng.permutation(len(registry))])
    return registry


def build_voter_datasets(n_rows: int = 1200, seed: int = DEFAULT_SEED, b_fraction: float = 0.5):
    """``(dataset_a, dataset_b)`` raw frames, both with ``cluster`` ground truth.

    ``dataset_b`` is a ``b_fraction`` sample of A, damaged at :data:`B_ERROR_RATES`
    and :data:`B_MISSING_RATES`; run both through the EDA step before linking.
    """
    from modules.eda_engine import infer_field_types
    a = generate_voter_registry(n_rows, seed)
    b = make_noisy_copy(a, B_ERROR_RATES, sample_frac=b_fraction, seed=seed,
                        field_types=infer_field_types(a.columns), missing_rates=B_MISSING_RATES,
                        settings=CorruptionSettings(text_edits=1), id_columns=("ncid", "voter_reg_num"))
    return a, b
