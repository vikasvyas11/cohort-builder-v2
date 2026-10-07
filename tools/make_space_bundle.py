"""Build the folder to push to a Hugging Face Space (Gradio SDK): only what the Space needs, plus its README header.

    python tools/make_space_bundle.py [output_folder]

The folder is recreated each time. Copy its contents into your Space's git repo (top level, replacing the old
files, README.md included), commit and push.

The bundle's requirements.txt names the same Gradio version as the README's sdk_version. That is deliberate: the Space
installs Gradio at sdk_version first and then this file, and a Gradio left over from an older sdk_version can be
incompatible with the newer web libraries this app installs (the import then fails). One version, written twice,
removes that. The Cohort Builder Assistant Space does the same.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT.parent / "cohort-builder-space"
FILES = ["app.py"]
FOLDERS = ["api", "modules", "utils", "web"]
SDK_VERSION = "6.20.0"          # the Gradio version the Cohort Builder Assistant Space already builds with
PYTHON_VERSION = "3.12.12"      # one of the two Python versions the ZeroGPU documentation lists as supported

HEADER = f"""---
title: Cohort Builder
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: {SDK_VERSION}
python_version: "{PYTHON_VERSION}"
app_file: app.py
pinned: false
---

# Cohort Builder

Record linkage and deduplication with Splink and DuckDB. Open the app at this Space's main address.
"""

EXTRA_REQUIREMENTS = f"""
# Space only. Must equal sdk_version in README.md (see tools/make_space_bundle.py).
gradio=={SDK_VERSION}
# Pre-installed on every Space; listed so the build is the same everywhere.
spaces>=0.51.0
"""


def main(out: Path = DEFAULT_OUT) -> None:
    OUT = Path(out)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for name in FILES:
        shutil.copy2(ROOT / name, OUT / name)
    for name in FOLDERS:
        shutil.copytree(ROOT / name, OUT / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (OUT / "README.md").write_text(HEADER, encoding="utf-8")
    base = (ROOT / "requirements.txt").read_text(encoding="utf-8").rstrip() + "\n"
    (OUT / "requirements.txt").write_text(base + EXTRA_REQUIREMENTS, encoding="utf-8")
    count = sum(1 for f in OUT.rglob("*") if f.is_file())
    print(f"Space bundle written to {OUT} ({count} files). Copy everything in it to the top level of your Space repo.")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
