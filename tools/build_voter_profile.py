"""Distil the NC statewide voter file into a small, anonymous statistical profile.

    python tools/build_voter_profile.py ncvoter_Statewide.txt

The raw file lists real, named people with street addresses and phone numbers, so
nothing from it may be copied into this project. This script reads only a few
columns (no addresses, phone numbers or ids), keeps **aggregate counts**, and
drops every name seen fewer than ``MIN_NAME_COUNT`` times, so no individual can
be recovered. The output, ``modules/nc_voter_profile.json``, is what
``modules/synthetic_data.py`` samples from to build synthetic voters.

Memory use is small and constant: the file is streamed in chunks.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "modules" / "nc_voter_profile.json"
MIN_NAME_COUNT = 500          # a name is kept only if at least this many voters share it
MIN_PLACE_COUNT = 50
CHUNK_ROWS = 200_000
COLUMNS = ["county_desc", "last_name", "first_name", "status_cd", "res_city_desc", "zip_code", "registr_dt",
           "race_code", "ethnic_code", "party_cd", "gender_code", "birth_year", "birth_state"]


def clean(series: pd.Series) -> pd.Series:
    return series.fillna("").str.strip().str.upper()


def main(path: str) -> None:
    simple = {k: Counter() for k in ("gender_code", "race_code", "ethnic_code", "party_cd", "status_cd",
                                     "birth_year", "birth_state", "registr_year", "county_desc")}
    race_party: dict = defaultdict(Counter)
    first_by_gender: dict = defaultdict(Counter)
    last_names, cities_by_county = Counter(), defaultdict(Counter)
    zips_by_city: dict = defaultdict(Counter)
    total = 0

    reader = pd.read_csv(path, sep="\t", dtype=str, usecols=COLUMNS, encoding="latin-1",
                         chunksize=CHUNK_ROWS, on_bad_lines="skip", quotechar='"')
    for chunk in reader:
        chunk = chunk.apply(clean)
        total += len(chunk)
        for col in ("gender_code", "race_code", "ethnic_code", "party_cd", "status_cd", "birth_year",
                    "birth_state", "county_desc"):
            simple[col].update(chunk[col].value_counts().to_dict())
        simple["registr_year"].update(chunk["registr_dt"].str[-4:].value_counts().to_dict())
        race_party_counts = chunk.groupby(["race_code", "party_cd"]).size()
        for (race, party), n in race_party_counts.items():
            race_party[race][party] += int(n)
        for gender, names in chunk.groupby("gender_code")["first_name"]:
            first_by_gender[gender].update(names.value_counts().to_dict())
        last_names.update(chunk["last_name"].value_counts().to_dict())
        for (county, city), n in chunk.groupby(["county_desc", "res_city_desc"]).size().items():
            cities_by_county[county][city] += int(n)
        for (city, zip_code), n in chunk.groupby(["res_city_desc", chunk["zip_code"].str[:5]]).size().items():
            zips_by_city[city][zip_code] += int(n)
        print(f"{total:,} rows", file=sys.stderr, flush=True)

    def keep(counter: Counter, minimum: int) -> dict:
        return {k: v for k, v in counter.items() if k and v >= minimum}

    profile = {
        "source": "aggregate counts from the NC State Board of Elections statewide voter file; no records kept",
        "n_rows": total,
        "min_name_count": MIN_NAME_COUNT,
        **{k: keep(v, 1) for k, v in simple.items() if k not in ("birth_state", "county_desc")},
        "birth_state": keep(simple["birth_state"], 1000),
        "county_desc": keep(simple["county_desc"], 1),
        "party_given_race": {r: keep(c, 1) for r, c in race_party.items() if r},
        "first_names": {g: keep(c, MIN_NAME_COUNT) for g, c in first_by_gender.items() if g in ("F", "M")},
        "last_names": keep(last_names, MIN_NAME_COUNT),
        "cities_by_county": {co: dict(sorted(keep(c, MIN_PLACE_COUNT).items(), key=lambda kv: -kv[1])[:12])
                             for co, c in cities_by_county.items() if co},
        "zips_by_city": {ci: dict(sorted(keep(z, MIN_PLACE_COUNT).items(), key=lambda kv: -kv[1])[:6])
                         for ci, z in zips_by_city.items() if ci},
    }
    OUT.write_text(json.dumps(profile, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB) from {total:,} rows", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ncvoter_Statewide.txt")
