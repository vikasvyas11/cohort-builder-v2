"""The static UI is plain JavaScript with no build step, so check what a build would: every file it
references exists, and the API serves it."""

import re
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app

WEB = Path(__file__).resolve().parent.parent / "web"


def test_every_script_stylesheet_and_import_exists():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    local = [m for m in re.findall(r'(?:src|href)="([^"]+)"', html) if not m.startswith(("http", "#", "data:"))]
    assert local, "index.html should reference its own files"
    for ref in local:
        assert (WEB / ref).is_file(), f"index.html references missing {ref}"
    for script in (WEB / "js").glob("*.js"):
        for target in re.findall(r'from "\./([\w.]+)"', script.read_text(encoding="utf-8")):
            assert (WEB / "js" / target).is_file(), f"{script.name} imports missing {target}"


def test_imported_names_are_exported():
    exports = {p.name: set(re.findall(r"export (?:async )?(?:function|const) (\w+)", p.read_text(encoding="utf-8")))
               for p in (WEB / "js").glob("*.js")}
    for script in (WEB / "js").glob("*.js"):
        for names, target in re.findall(r'import \{([^}]*)\} from "\./([\w.]+)"', script.read_text(encoding="utf-8")):
            for name in (n.strip() for n in names.split(",") if n.strip()):
                assert name in exports[target], f"{script.name} imports {name}, which {target} does not export"


def test_the_api_serves_the_page_and_scripts():
    client = TestClient(app)
    assert "Cohort Builder" in client.get("/").text
    assert client.get("/js/main.js").status_code == 200
    assert client.get("/style.css").status_code == 200


def test_the_app_py_entry_point_imports_without_starting_a_server():
    import app as entry
    assert entry.app is app
