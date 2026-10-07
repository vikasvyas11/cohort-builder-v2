"""Real Splink runs on the synthetic demo data, plus model JSON and PDF round trips."""

import json

import pytest

from modules.metrics_engine import (
    compute_confusion_matrix, compute_intra_metrics, compute_threshold_curve, summarise_threshold_curve,
)
from modules.report import generate_report
from modules.splink_runner import reconstruct_model_json, run_linkage, run_linkage_from_json

MODES = [("dedupe", "deterministic"), ("dedupe", "probabilistic"),
         ("link_dedupe", "deterministic"), ("link_dedupe", "probabilistic")]


@pytest.mark.parametrize("mode,method", MODES)
def test_runs_produce_sensible_accuracy(linkage_runs, demo, mode, method):
    _, a, b = demo
    run = linkage_runs[(mode, method)]
    cm = compute_confusion_matrix(run["df_predict"], a, b if mode != "dedupe" else None, mode)
    assert run["n_edges"] > 0 and run["n_clusters"] > 0
    assert cm["precision"] > 0.85 and cm["recall"] > 0.85


def test_deterministic_does_not_chain_everything_into_giant_clusters(linkage_runs):
    run = linkage_runs[("dedupe", "deterministic")]
    sizes = run["df_cluster"].groupby("cluster_id").size()
    assert sizes.max() <= 12                       # a person appears at most 4 times; allow slack


def test_probabilistic_run_reports_model_and_unlinkables(linkage_runs):
    run = linkage_runs[("dedupe", "probabilistic")]
    assert run["model_params"]["training_complete"]
    assert run["unlinkables"]["thresholds"]
    pcts = run["unlinkables"]["pcts"]
    assert len(pcts) == len(run["unlinkables"]["thresholds"])
    assert all(0 <= p <= 100 for p in pcts) and pcts == sorted(pcts)      # stricter threshold -> more unlinkable
    assert run["blocking_counts"]


def test_threshold_curve_matches_confusion_matrix_at_lowest_threshold(linkage_runs, demo):
    _, a, _ = demo
    run = linkage_runs[("dedupe", "probabilistic")]
    curve = compute_threshold_curve(run["df_predict"], a, None, "dedupe")
    cm = compute_confusion_matrix(run["df_predict"], a, None, "dedupe")
    last = curve.iloc[-1]                          # lowest threshold = every scored pair predicted
    assert (last["tp"], last["fp"], last["fn"]) == (cm["tp"], cm["fp"], cm["fn"])
    assert 0.9 < summarise_threshold_curve(curve)["average_precision"] <= 1.0


def test_model_json_roundtrip_reproduces_the_run(linkage_runs, demo):
    _, a, _ = demo
    original = linkage_runs[("dedupe", "probabilistic")]
    model = json.loads(json.dumps(reconstruct_model_json(original["settings_used"], original["model_params"])))
    replay = run_linkage_from_json(model, a, None, "dedupe", linkage_type="probabilistic")
    assert replay["n_edges"] == original["n_edges"]
    assert replay["n_clusters"] == original["n_clusters"]


def test_deterministic_model_json_roundtrip(linkage_runs, demo):
    _, a, _ = demo
    original = linkage_runs[("dedupe", "deterministic")]
    model = reconstruct_model_json(original["settings_used"], original["model_params"], "deterministic")
    assert model["_app_linkage_type"] == "deterministic"
    replay = run_linkage_from_json(model, a, None, "dedupe", linkage_type="deterministic")
    assert replay["n_edges"] == original["n_edges"]


@pytest.mark.parametrize("mode,method", MODES)
def test_html_report_generates(linkage_runs, demo, mode, method):
    _, a, b = demo
    other = b if mode != "dedupe" else None
    run = linkage_runs[(mode, method)]
    metrics = compute_intra_metrics(run["df_predict"], run["df_cluster"])
    cm = compute_confusion_matrix(run["df_predict"], a, other, mode)
    curve = compute_threshold_curve(run["df_predict"], a, other, mode) if method == "probabilistic" else None
    pdf = generate_report(
        "Run 1", run["run_config"], metrics, run["n_input_records"], run["model_params"],
        run["missingness_a"], run["missingness_b"], run["blocking_counts"], run["unlinkables"],
        run["settings_used"], cm, curve, summarise_threshold_curve(curve))
    assert pdf.startswith(b"<!doctype html>") and b"Linkage run report" in pdf and len(pdf) > 10_000


def test_report_survives_minimal_inputs():
    pdf = generate_report("Run X", {"operation_mode": "dedupe", "linkage_type": "deterministic"}, {}, 0)
    assert b"Linkage run report" in pdf


def test_zero_edge_runs_explain_themselves(demo):
    _, a, _ = demo
    unique = a.copy()
    unique["email"] = unique["unique_id"] + "@x.example"             # every value unique -> nothing to block on
    run = run_linkage(unique, None, ["email"], {"email": True}, "dedupe", "deterministic")
    assert run["n_edges"] == 0
    assert run["zero_edge_diagnostic"][0]["issue"]
