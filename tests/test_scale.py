"""The voter flow at a size where blocking choices matter. 10,000 records always run; set
RUN_SCALE_TESTS=1 for the 50,000-record flow (about 1 GB of memory and half a minute)."""

import os
import time

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def run(sid: str, config: dict) -> dict:
    job = client.post(f"/api/sessions/{sid}/runs/run1", json=config).json()["job_id"]
    for _ in range(600):
        status = client.get(f"/api/jobs/{job}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert status["status"] == "done", status
    return client.get(f"/api/sessions/{sid}/runs/run1").json()


@pytest.mark.parametrize("rows", [10_000, pytest.param(50_000, marks=pytest.mark.skipif(
    not os.environ.get("RUN_SCALE_TESTS"), reason="set RUN_SCALE_TESTS=1 for the 50,000-record flow"))])
def test_voter_flow_at_scale(rows):
    over = client.post("/api/sessions", json={"source": "voters", "n_rows": rows}).json()
    assert over["rows_a"] == rows and over["rows_b"] == rows // 2
    defaults = over["defaults"]
    assert defaults["blocking_toggles"]["first_name+last_name"] and not defaults["blocking_toggles"]["first_name"]
    sid = over["session_id"]
    config = {"fields": defaults["fields"], "blocking_toggles": defaults["blocking_toggles"],
              "operation_mode": "link_dedupe", "linkage_type": "probabilistic"}
    estimate = client.post(f"/api/sessions/{sid}/estimate", json=config).json()
    assert estimate["level"] == "ok"
    summary = run(sid, config)
    cm = summary["accuracy"]["cm"]
    assert cm["precision"] > 0.9 and cm["recall"] > 0.9
    # a loose rule the defaults avoid is counted, not run: the live waterfall sees millions of pairs
    loose = client.post(f"/api/sessions/{sid}/waterfall",
                        json={"toggles": {**defaults["blocking_toggles"], "first_name": True}}).json()
    assert loose["totals"]["after"] > loose["totals"]["before"]
