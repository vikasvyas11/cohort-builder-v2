"""Charts and the downloadable run report.

Everything here uses Plotly (already needed by the app) and plain HTML, so the
report needs no extra libraries: it is one self-contained ``.html`` file with the
charts embedded, readable offline, and printable to PDF from any browser.
"""

from __future__ import annotations

import html
import math
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

BLUE, ORANGE, GREEN, GREY = "#0E6B63", "#E0A21B", "#A3205B", "#6B7280"   # teal (A), saffron (B), plum (link)


# =============================================================================
# Charts (Plotly figures, used both in the app and in the report)
# =============================================================================

def _style(fig: go.Figure, title: str, height: int = 320) -> go.Figure:
    fig.update_layout(title=title, template="simple_white", height=height,
                      margin=dict(l=40, r=20, t=50, b=40))
    return fig


def venn_figure(a_only: int, both: int, b_only: int, title: str = "Clusters by source dataset") -> go.Figure:
    """Two overlapping circles: clusters with A-only, both, or B-only records."""
    radius, y_mid = 1.6, 2.0
    fig = go.Figure()
    for cx, color in ((2.2, BLUE), (4.0, ORANGE)):          # centres 1.8 apart, radius 1.6: a real overlap
        fig.add_shape(type="circle", x0=cx - radius, x1=cx + radius, y0=y_mid - radius, y1=y_mid + radius,
                      fillcolor=color, opacity=0.55, line=dict(color=color, width=2), layer="below")
    for x, text in ((1.25, f"<b>{a_only:,}</b><br>A only"), (3.1, f"<b>{both:,}</b><br>both"),
                    (4.95, f"<b>{b_only:,}</b><br>B only")):
        fig.add_annotation(x=x, y=y_mid, text=text, showarrow=False, font=dict(size=15, color="#111827"))
    for x, text in ((1.7, "Dataset A"), (4.5, "Dataset B")):
        fig.add_annotation(x=x, y=y_mid + radius + 0.35, text=f"<b>{text}</b>", showarrow=False, font=dict(size=13))
    fig.update_xaxes(visible=False, range=[0, 6.2])
    fig.update_yaxes(visible=False, range=[0, 4.4], scaleanchor="x", scaleratio=1)
    return _style(fig, title, 300)


def match_weights_figure(model_params: dict) -> Optional[go.Figure]:
    """log2(m/u) for every comparison level, plus the prior."""
    labels, weights = [], []
    if model_params.get("prior_log_odds") is not None:
        labels.append("Prior"), weights.append(model_params["prior_log_odds"])
    for comp in model_params.get("comparisons", []):
        for level in comp["levels"]:
            if level.get("match_weight") is not None:
                labels.append(f"{comp['field']}: {level['label']}"), weights.append(level["match_weight"])
    if not labels:
        return None
    fig = go.Figure(go.Bar(x=weights, y=labels, orientation="h", marker_color=BLUE))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Match weight = log2(m / u)")
    return _style(fig, "Evidence carried by each comparison level", max(320, 24 * len(labels) + 100))


def histogram_figure(frame: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str) -> Optional[go.Figure]:
    if frame is None or frame.empty:
        return None
    fig = go.Figure(go.Bar(x=frame[x], y=frame[y], marker_color=BLUE))
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title=ylabel)
    return _style(fig, title)


def unlinkables_figure(unlinkables: dict) -> Optional[go.Figure]:
    if not unlinkables.get("thresholds"):
        return None
    fig = go.Figure(go.Scatter(x=unlinkables["thresholds"], y=unlinkables["pcts"], fill="tozeroy",
                               line=dict(color=BLUE)))
    fig.update_xaxes(title="Match-weight threshold")
    fig.update_yaxes(title="Records that cannot reach it (%)", range=[0, 105])
    return _style(fig, "Unlinkable records by threshold")


def blocking_figure(blocking_counts: list) -> Optional[go.Figure]:
    if not blocking_counts:
        return None
    labels = [f"Rule {c['rule_index']}" for c in blocking_counts]
    fig = go.Figure(go.Bar(x=labels, y=[c["n"] for c in blocking_counts], marker_color=BLUE))
    fig.update_yaxes(title="Candidate pairs")
    return _style(fig, "Candidate pairs by blocking rule")


def confusion_figure(cm: dict) -> go.Figure:
    tp, fp, fn = cm.get("tp", 0), cm.get("fp", 0), cm.get("fn", 0)
    fig = go.Figure(go.Heatmap(
        z=[[tp, fp], [fn, 0]], text=[[f"TP<br>{tp:,}", f"FP<br>{fp:,}"], [f"FN<br>{fn:,}", "TN<br>(not counted)"]],
        texttemplate="%{text}", colorscale=[[0, "#B3261E"], [0.5, "#D8DCD3"], [1, "#0E6B63"]], showscale=False))
    fig.update_xaxes(tickvals=[0, 1], ticktext=["Predicted match", "Predicted non-match"])
    fig.update_yaxes(tickvals=[0, 1], ticktext=["True match", "True non-match"], autorange="reversed")
    return _style(fig, "Pair-level confusion matrix", 280)


def precision_recall_figure(curve: pd.DataFrame) -> Optional[go.Figure]:
    """Precision-recall curve (left) and F* against threshold (right)."""
    if curve is None or curve.empty:
        return None
    pr = curve.dropna(subset=["precision", "recall"])
    if pr.empty:
        return None
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Precision-recall curve", "F* against threshold"))
    fig.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], line=dict(color=BLUE)), row=1, col=1)
    fstar = curve.dropna(subset=["fstar", "match_probability"])
    fig.add_trace(go.Scatter(x=fstar["match_probability"], y=fstar["fstar"], line=dict(color=GREEN)), row=1, col=2)
    fig.update_xaxes(title_text="Recall", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Precision", range=[0, 1.05], row=1, col=1)
    fig.update_xaxes(title_text="Match-probability threshold", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text="F* = TP / (TP + FP + FN)", range=[0, 1.05], row=1, col=2)
    fig.update_layout(showlegend=False)
    return _style(fig, "", 340)


# =============================================================================
# HTML report
# =============================================================================

_CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#1f2937}
h1{color:#17212B}h2{color:#17212B;border-bottom:2px solid #0E6B63;padding-bottom:.2rem;margin-top:2.2rem}
table{border-collapse:collapse;margin:.6rem 0}th{background:#0E6B63;color:#fff;text-align:left}
th,td{padding:.35rem .7rem;border:1px solid #d1d5db}tr:nth-child(even) td{background:#f3f4f6}
code,pre{background:#f0f3fa;padding:.15rem .4rem;border-radius:4px;font-size:.85rem}pre{padding:.5rem;overflow-x:auto}
.muted{color:#6b7280;font-size:.9rem}@media print{h2{break-after:avoid}.chart{break-inside:avoid}}
"""


def _table(headers: list, rows: list) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _facts(rows: list) -> str:
    return _table(["", ""], rows).replace("<tr><th></th><th></th></tr>", "")


def _num(value, digits: int = 4) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


class _Page:
    """Accumulates HTML; the first chart embeds plotly.js, later ones reuse it."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self._js_included = False

    def h2(self, text: str) -> None:
        self.parts.append(f"<h2>{html.escape(text)}</h2>")

    def p(self, text: str, muted: bool = False) -> None:
        self.parts.append(f'<p class="{"muted" if muted else ""}">{html.escape(text)}</p>')

    def raw(self, markup: str) -> None:
        self.parts.append(markup)

    def chart(self, fig: Optional[go.Figure]) -> None:
        if fig is None:
            return
        markup = fig.to_html(full_html=False, include_plotlyjs=not self._js_included)
        self._js_included = True
        self.parts.append(f'<div class="chart">{markup}</div>')

    def render(self, title: str) -> str:
        return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
                f"<style>{_CSS}</style></head><body>{''.join(self.parts)}</body></html>")


def generate_report(
    run_label: str,
    run_config: dict,
    metrics: dict,
    n_input_records: int,
    model_params: Optional[dict] = None,
    missingness_a: Optional[dict] = None,
    missingness_b: Optional[dict] = None,
    blocking_counts: Optional[list] = None,
    unlinkables: Optional[dict] = None,
    settings_used: Optional[dict] = None,
    confusion_matrix: Optional[dict] = None,
    threshold_curve: Optional[pd.DataFrame] = None,
    curve_summary: Optional[dict] = None,
) -> bytes:
    """Build the run report as a self-contained HTML document (UTF-8 bytes).

    Only ``run_label``, ``run_config``, ``metrics`` and ``n_input_records`` are
    required; sections without data are skipped or say why they are empty.
    """
    model_params, settings_used = model_params or {}, settings_used or {}
    blocking_counts, unlinkables = blocking_counts or [], unlinkables or {}
    cm, curve = confusion_matrix or {}, threshold_curve if threshold_curve is not None else pd.DataFrame()
    mode = run_config.get("operation_mode", "dedupe")
    method = run_config.get("linkage_type", "deterministic")
    has_accuracy = bool(cm) and not cm.get("unavailable") and "error" not in cm

    page = _Page()
    page.raw(f"<h1>Linkage run report</h1><p class='muted'>{html.escape(run_label)} &middot; "
             f"{html.escape(mode.replace('_', ' '))}, {html.escape(method)} &middot; "
             f"generated {datetime.now():%Y-%m-%d %H:%M}</p>")

    # -- summary and settings --------------------------------------------------
    stats = metrics.get("cluster_stats", pd.DataFrame())
    rows = [["Input records", f"{n_input_records:,}"],
            ["Candidate pairs scored", f"{metrics.get('n_edges', 0):,}"],
            ["Records with at least one link", f"{metrics.get('linkage_rate', 0)}%"],
            ["Entity clusters", f"{metrics.get('n_clusters', 0):,}"]]
    if not stats.empty:
        rows.append(["Clusters with 2+ records", f"{int(stats['multi_member_clusters'].iloc[0]):,}"])
    if has_accuracy:
        rows += [["Precision", _num(cm.get("precision"))], ["Recall", _num(cm.get("recall"))], ["F1", _num(cm.get("f1"))]]
    if curve_summary:
        rows.append(["Average precision", _num(curve_summary.get("average_precision"))])
    page.h2("At a glance")
    page.raw(_facts(rows))

    active = [k for k, v in run_config.get("blocking_toggles", {}).items() if v]
    page.h2("Settings")
    page.raw(_facts([
        ["Fields compared", ", ".join(run_config.get("selected_fields", [])) or "-"],
        ["Active blocking rules", ", ".join(active) or "-"],
        ["Blocking mode", run_config.get("blocking_mode", "OR")],
        ["Cluster threshold (probability)", run_config.get("cluster_threshold", 0.8)],
    ]))

    # -- input data -------------------------------------------------------------
    page.h2("Input data")
    page.p("Completeness is the share of non-null values in each compared field. Sparse fields add little "
           "evidence and make poor blocking keys.")
    fields = list(missingness_a or {})
    if fields:
        header = ["Field", "Dataset A (%)"] + (["Dataset B (%)"] if missingness_b and mode != "dedupe" else [])
        page.raw(_table(header, [[f, f"{missingness_a.get(f, 0):.1f}"]
                                 + ([f"{missingness_b.get(f, 0):.1f}"] if len(header) == 3 else []) for f in fields]))

    # -- blocking ---------------------------------------------------------------
    page.h2("Candidate pairs (blocking)")
    page.p("Blocking rules choose which pairs are worth scoring. A true match that no rule selects can never "
           "be found, whatever the model says.")
    counts = {c["rule_index"]: c["n"] for c in blocking_counts}
    rules = settings_used.get("blocking_rules_to_generate_predictions", [])
    page.raw(_table(["Rule", "SQL", "Pairs generated"], [
        [i, (r.get("blocking_rule", "") if isinstance(r, dict) else str(r)), f"{counts[i]:,}" if i in counts else "n/a"]
        for i, r in enumerate(rules)]))
    page.chart(blocking_figure(blocking_counts))

    # -- scoring ----------------------------------------------------------------
    page.h2("Scoring")
    if method == "probabilistic":
        hp = run_config.get("hyperparams") or {}
        page.p("Pairs are scored with a Fellegi-Sunter model fitted by expectation-maximisation. A level's match "
               "weight is log2(m / u): m is how often true matches show that level, u how often non-matches do.")
        page.raw(_facts([["Maximum EM iterations", hp.get("max_iterations", "default")],
                         ["EM convergence threshold", hp.get("em_convergence", "default")],
                         ["Recall estimate for the prior", hp.get("recall_estimate", "default")],
                         ["Prior match weight", _num(model_params.get("prior_log_odds"))]]))
        param_rows = [[c["field"], lv["label"], _num(lv["m_prob"]), _num(lv["u_prob"], 6), _num(lv["match_weight"])]
                      for c in model_params.get("comparisons", []) for lv in c["levels"]
                      if lv.get("match_weight") is not None]
        if param_rows:
            page.raw(_table(["Field", "Level", "m", "u", "Weight"], param_rows))
        page.chart(match_weights_figure(model_params))
    else:
        page.p("Deterministic runs score nothing: a candidate pair that agrees exactly on enough of the blocking "
               "fields is accepted as a match with probability 1.")

    # -- scores and clusters ----------------------------------------------------
    page.h2("Scores and clusters")
    page.chart(histogram_figure(metrics.get("weight_dist"), "weight_bin", "n_edges",
                                "Distribution of match weights", "Match weight", "Candidate pairs")
               if len(metrics.get("weight_dist", [])) > 1 else None)
    if method == "probabilistic":
        page.chart(unlinkables_figure(unlinkables))
    page.p(f"Pairs at or above match probability {run_config.get('cluster_threshold', 0.8)} are joined into "
           "clusters of connected records. One oversized cluster usually means a loose rule chained strangers.")
    split = metrics.get("singleton_stats", pd.DataFrame())
    if not split.empty:
        page.raw(_table(["Cluster type", "Clusters", "Records"],
                        [[r.cluster_type, f"{r.n_clusters:,}", f"{r.total_records:,}"] for r in split.itertuples()]))
    page.chart(histogram_figure(metrics.get("cluster_sizes"), "n_nodes", "n_clusters",
                                "Cluster sizes", "Records in cluster", "Clusters"))
    if mode != "dedupe":
        venn = metrics.get("venn", {})
        page.chart(venn_figure(venn.get("a_only", 0), venn.get("both_ab", 0), venn.get("b_only", 0)))

    # -- accuracy ---------------------------------------------------------------
    if cm:
        page.h2("Accuracy against ground truth")
        if not has_accuracy:
            page.p(cm.get("unavailable_reason") or cm.get("error") or "Accuracy is not available for this dataset.")
        else:
            page.p("Ground truth is the data's 'cluster' column: records with the same value are the same entity. "
                   f"Pairs count as predicted at match probability >= {cm.get('min_match_probability', 0):.2f}. "
                   "True negatives are not counted.")
            page.raw(_table(["Metric", "Value"], [
                ["True / predicted pairs", f"{cm.get('n_gt_edges') or 0:,} / {cm.get('n_pred_edges') or 0:,}"],
                ["TP / FP / FN", f"{cm.get('tp'):,} / {cm.get('fp'):,} / {cm.get('fn'):,}"],
                ["Precision", _num(cm.get("precision"))], ["Recall", _num(cm.get("recall"))],
                ["F1", _num(cm.get("f1"))], ["F* = TP / (TP + FP + FN)", _num(cm.get("fstar"))]]))
            page.chart(confusion_figure(cm))
            if method == "probabilistic":
                page.chart(precision_recall_figure(curve))
                if curve_summary:
                    page.raw(_facts([
                        ["Average precision", _num(curve_summary.get("average_precision"))],
                        ["Best F1", f"{_num(curve_summary.get('best_f1'))} at probability >= "
                                    f"{_num(curve_summary.get('best_f1_threshold'), 3)}"]]))

    page.h2("References")
    page.raw("<ul class='muted'><li>Fellegi &amp; Sunter (1969). A theory for record linkage. JASA 64(328).</li>"
             "<li>Linacre et al. (2022). Splink: Free software for probabilistic record linkage at scale. "
             "IJPDS 7(3).</li><li>Hand, Christen &amp; Kirielle (2021). F*: an interpretable transformation of "
             "the F-measure. Machine Learning 110.</li></ul>")
    return page.render(f"Linkage run report - {run_label}").encode("utf-8")
