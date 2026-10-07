"""Distil two more columns of the NC statewide voter file, street address and phone number, into aggregate shapes.

    python tools/build_voter_extras.py ncvoter_Statewide.txt [memory_cap_gb]

Like ``build_voter_profile.py``, nothing about an individual is kept. The raw file lists real people with house
addresses and phone numbers, so this reads only those two columns, in small chunks, and keeps counts:

* street names shared by at least ``MIN_STREET_COUNT`` addresses (``MAIN ST``, ``OAK DR``), never a house number;
* how house numbers are spread (counts per range), how often an address has an apartment or unit part;
* how often a phone number is present, the area codes shared by at least ``MIN_AREA_COUNT`` numbers, and the
  usual text format with every digit replaced by 9.

``modules/nc_voter_extras.json`` is what ``modules.voter_data.add_address_and_phone`` samples from. The phone
numbers it creates are random digits in the right format, not numbers taken from the file.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.memory import guard  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "modules" / "nc_voter_extras.json"
MIN_STREET_COUNT = 500
MIN_AREA_COUNT = 500
CHUNK_ROWS = 50_000
HOUSE_BINS = [1, 100, 500, 1000, 2000, 5000, 10000, 100000]
UNIT = re.compile(r"\s+(?:APT|UNIT|LOT|STE|SUITE|TRLR|BLDG|SPC|#).*$|\s*#.*$")
NUMBER = re.compile(r"^(\d+)[A-Z]?\s+")


def main(path: str) -> None:
    streets, formats, areas = Counter(), Counter(), Counter()
    house_bins = Counter()
    addresses = with_unit = with_number = phones = rows = 0

    reader = pd.read_csv(path, sep="\t", dtype=str, usecols=["res_street_address", "full_phone_number"],
                         encoding="latin-1", chunksize=CHUNK_ROWS, on_bad_lines="skip", quotechar='"')
    for chunk in reader:
        rows += len(chunk)
        address = chunk["res_street_address"].fillna("").str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
        address = address[address != ""]
        addresses += len(address)
        with_unit += int(address.str.contains(UNIT).sum())
        base = address.str.replace(UNIT, "", regex=True)
        number = base.str.extract(NUMBER)[0]
        with_number += int(number.notna().sum())
        for value in pd.to_numeric(number.dropna(), errors="coerce").dropna():
            for low, high in zip(HOUSE_BINS, HOUSE_BINS[1:]):
                if low <= value < high:
                    house_bins[f"{low}-{high - 1}"] += 1
                    break
        names = base.str.replace(NUMBER, "", regex=True).str.strip()
        streets.update(names[names.str.match(r"^[A-Z]")].value_counts().to_dict())

        phone = chunk["full_phone_number"].fillna("").str.strip()
        phone = phone[phone != ""]
        phones += len(phone)
        formats.update(phone.str.replace(r"\d", "9", regex=True).value_counts().head(20).to_dict())
        digits = phone.str.replace(r"\D", "", regex=True)
        areas.update(digits[digits.str.len() >= 10].str[:3].value_counts().to_dict())
        if rows % 1_000_000 < CHUNK_ROWS:
            print(f"{rows:,} rows", file=sys.stderr, flush=True)

    streets.pop("REMOVED", None)             # the file's placeholder for voters whose address was withheld
    profile = {
        "source": "aggregate shapes from the NC State Board of Elections statewide voter file; no records kept",
        "n_rows": rows,
        "min_street_count": MIN_STREET_COUNT,
        "street_names": dict(sorted(((k, v) for k, v in streets.items() if v >= MIN_STREET_COUNT),
                                    key=lambda kv: -kv[1])[:600]),
        "house_number_bins": dict(house_bins),
        "share_with_unit": round(with_unit / max(addresses, 1), 4),
        "share_with_house_number": round(with_number / max(addresses, 1), 4),
        "share_with_phone": round(phones / max(rows, 1), 4),
        "area_codes": {k: v for k, v in areas.items() if v >= MIN_AREA_COUNT},
        "phone_formats": dict(formats.most_common(5)),
    }
    OUT.write_text(json.dumps(profile, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB) from {rows:,} rows: {len(profile['street_names'])} street "
          f"names, {len(profile['area_codes'])} area codes, phone present {profile['share_with_phone']:.1%}, "
          f"unit part {profile['share_with_unit']:.1%}, formats {profile['phone_formats']}", file=sys.stderr)


if __name__ == "__main__":
    guard(float(sys.argv[2]) if len(sys.argv) > 2 else 0.7)       # this machine had little free memory: stop, don't swap
    main(sys.argv[1] if len(sys.argv) > 1 else "ncvoter_Statewide.txt")
