"""The Hugging Face Space bundle must be self-consistent: Gradio is named once in the README and once in
requirements, at the same version, and nothing private ships."""

import re

from tools.make_space_bundle import SDK_VERSION, main


def test_bundle_pins_one_gradio_version_everywhere(tmp_path):
    main(tmp_path)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    requirements = (tmp_path / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(rf"^sdk_version: {re.escape(SDK_VERSION)}$", readme, re.M)
    assert re.search(rf"^gradio=={re.escape(SDK_VERSION)}$", requirements, re.M)
    assert "sdk: gradio" in readme and "app_file: app.py" in readme


def test_bundle_has_the_app_and_no_data(tmp_path):
    main(tmp_path)
    names = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert {"app.py", "api/main.py", "modules/splink_runner.py", "web/index.html", "web/js/main.js"} <= names
    assert not [n for n in names if n.endswith((".csv", ".zip")) or "ncvoter" in n or n.startswith(("tests/", ".venv"))]
