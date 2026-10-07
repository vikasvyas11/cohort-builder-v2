"""Drive the API at 50,000 records and report time and memory at each step.

    python tools/scale_check.py [rows] [memory_cap_gb]

Runs the whole app flow in-process (no server needed): generate the voter datasets, dedupe,
link, a live waterfall, the explorer, Run 2, compare and the cohort export. A watchdog stops the
process if its memory passes the cap (default 3 GB), so a mistake cannot take the machine down.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CAP_GB = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
os.environ.setdefault("COHORT_BUILDER_MAX_SESSIONS", "4")

from tools.memory import guard, memory_mb  # noqa: E402

guard(CAP_GB)

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=True)
clock = time.time()


def lap(label: str) -> None:
    global clock
    now, (cur, peak) = time.time(), memory_mb()
    print(f"{label:<44} {now - clock:7.1f}s   mem {cur:5d} MB (peak {peak} MB)", flush=True)
    clock = now


def run(sid: str, slot: str, config: dict) -> dict:
    job = client.post(f"/api/sessions/{sid}/runs/{slot}", json=config).json()["job_id"]
    while True:
        status = client.get(f"/api/jobs/{job}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert status["status"] == "done", status
    return client.get(f"/api/sessions/{sid}/runs/{slot}").json()


over = client.post("/api/sessions", json={"source": "voters", "n_rows": ROWS}).json()
sid = over["session_id"]
lap(f"create {ROWS:,} voters (A {over['rows_a']:,}, B {over['rows_b']:,}) + profile")
defaults = over["defaults"]
config = {"fields": defaults["fields"], "blocking_toggles": defaults["blocking_toggles"],
          "operation_mode": "dedupe", "linkage_type": "probabilistic"}
est = client.post(f"/api/sessions/{sid}/estimate", json=config).json()
lap(f"estimate: {est['pairs']:,} pairs ({est['level']})")

for mode in ("dedupe", "link_dedupe"):
    r1 = run(sid, "run1", {**config, "operation_mode": mode})
    acc = r1["accuracy"]["cm"]
    lap(f"run 1 {mode}: {r1['cards']['edges']:,} edges, P={acc['precision']:.3f} R={acc['recall']:.3f}")
    client.get(f"/api/sessions/{sid}/runs/run1/demographics")
    lap("  demographics tab")
    ex = client.post(f"/api/sessions/{sid}/runs/run1/explorer", json={})
    assert ex.status_code == 200, ex.text
    lap("  explorer + waterfall + match quality")
    base = {k: v for k, v in config["blocking_toggles"].items()}
    wf = client.post(f"/api/sessions/{sid}/waterfall", json={"toggles": {**base, "first_name": True}})
    assert wf.status_code == 200, wf.text
    lap(f"  live waterfall (adds first_name: {wf.json()['totals']['after']:,} pairs)")
    wf = client.post(f"/api/sessions/{sid}/waterfall", json={"toggles": {**base, "first_name": True, "dob": False}})
    lap("  live waterfall again (cached patterns)")

r2 = run(sid, "run2", {**config, "operation_mode": "link_dedupe"})
lap(f"run 2: {r2['cards']['edges']:,} edges")
assert client.get(f"/api/sessions/{sid}/compare").status_code == 200
lap("compare run 1 vs run 2")
csv = client.get(f"/api/sessions/{sid}/runs/run2/cohort.csv").content
lap(f"cohort export ({len(csv) / 1e6:.1f} MB)")
rep = client.get(f"/api/sessions/{sid}/runs/run2/report.html").content
lap(f"HTML report ({len(rep) / 1e6:.1f} MB)")
print(f"\nDONE. Peak memory {memory_mb()[1]} MB.")
