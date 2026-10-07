"""Write sample files for the Upload flow: 10,000 synthetic voters (Dataset A) and a damaged copy (Dataset B).

    python tools/make_upload_samples.py [output_folder] [rows]

Both files carry the registry columns plus two the built-in demo does not have, ``res_street_address`` and
``full_phone_number``, and a ``cluster`` column that is the ground truth (records with the same value are the
same voter). Everything is synthetic; see ``modules/voter_contact.py``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.corruption import CorruptionSettings, make_noisy_copy  # noqa: E402
from modules.eda_engine import infer_field_types  # noqa: E402
from modules.voter_contact import add_address_and_phone, damage_address_and_phone  # noqa: E402
from modules.voter_data import B_ERROR_RATES, B_MISSING_RATES, DEFAULT_SEED, generate_voter_registry  # noqa: E402


def main(out: Path, rows: int = 10_000) -> None:
    out.mkdir(parents=True, exist_ok=True)
    a = add_address_and_phone(generate_voter_registry(rows, DEFAULT_SEED), DEFAULT_SEED)
    b = make_noisy_copy(a, B_ERROR_RATES, sample_frac=0.5, seed=DEFAULT_SEED, field_types=infer_field_types(a.columns),
                        missing_rates=B_MISSING_RATES, settings=CorruptionSettings(text_edits=1),
                        id_columns=("ncid", "voter_reg_num"))
    b = damage_address_and_phone(b, DEFAULT_SEED).drop(columns=["source_dataset"], errors="ignore")
    a.to_csv(out / f"voters_{rows // 1000}k_A.csv", index=False)
    b.to_csv(out / f"voters_{rows // 1000}k_B.csv", index=False)
    print(f"A: {len(a):,} rows x {a.shape[1]} columns; B: {len(b):,} rows x {b.shape[1]} columns -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "examples", int(sys.argv[2]) if len(sys.argv) > 2 else 10_000)
