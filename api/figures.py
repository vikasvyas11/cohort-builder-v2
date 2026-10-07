"""Plotly figures for the web UI, returned as JSON (the page draws them with Plotly.js).

Building figures here, rather than in the browser, keeps the front end a thin layer and lets
the HTML report and the web app share the same drawing code.
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from modules.splink_runner import compute_blocking_waterfall, determine_cascade_order

BLUE, ORANGE, GREEN, PURPLE, RED = "#0E6B63", "#E0A21B", "#A3205B", "#5B6FD6", "#B3261E"   # teal, saffron, plum, indigo, red
PALETTE = [BLUE, ORANGE, GREEN, PURPLE, "#7FB069", "#C97B5A", "#8C97A3"]


def to_json(fig: Optional[go.Figure]) -> Optional[dict]:
    return None if fig is None else json.loads(fig.to_json())


def bar(df: pd.DataFrame, x: str, y: str, title: str, colour: str = BLUE, height: int = 320) -> dict:
    fig = px.bar(df, x=x, y=y, title=title, color_discrete_sequence=[colour], template="simple_white")
    fig.update_layout(title_font_size=14, xaxis_title=x.replace("_", " ").title(),
                      yaxis_title=y.replace("_", " ").title(), margin=dict(l=40, r=20, t=50, b=40), height=height)
    return to_json(fig)


def pie(df: pd.DataFrame, values: str, names: str, title: str, height: int = 320) -> dict:
    fig = px.pie(df, values=values, names=names, title=title, template="simple_white",
                 color_discrete_sequence=PALETTE)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=height)
    return to_json(fig)


def singleton_chart(frame: pd.DataFrame) -> dict:
    fig = px.bar(frame, x="cluster_type", y="n_clusters", color="cluster_type",
                 color_discrete_sequence=[BLUE, ORANGE], title="Singleton vs Multi-record", template="simple_white",
                 labels={"cluster_type": "", "n_clusters": "Number of clusters"})
    fig.update_layout(height=280, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    return to_json(fig)


def breakdown_figures(breakdowns: dict, height: int = 320) -> list:
    """One chart per demographic field: pies for coded fields, bars for the rest."""
    out = []
    for field, info in breakdowns.items():
        df, label, kind = info["df"], info["label"], info["kind"]
        if kind == "pie":
            fig = pie(df, "n_records", "label", f"{label} Distribution", height)
        else:
            x = "label" if kind == "bar_binned" else "value"
            shown = df if kind == "bar_binned" else df.head(10)
            title = label if kind == "bar_binned" else f"Top {min(10, len(df))} {label} values"
            fig = bar(shown, x, "n_records", title, PURPLE, height)
        out.append({"field": field, "label": label, "figure": fig})
    return out


def threshold_figures(curve: Optional[pd.DataFrame]) -> list:
    """Precision-recall and F* against threshold; empty when the run has no ground truth."""
    if curve is None or curve.empty:
        return []
    out = []
    pr = curve.dropna(subset=["precision", "recall"])
    if not pr.empty:
        fig = px.line(pr, x="recall", y="precision", title="Precision-Recall Curve", template="simple_white",
                      color_discrete_sequence=[BLUE])
        out.append(to_json(fig.update_layout(height=300, xaxis_range=[0, 1], yaxis_range=[0, 1.05])))
    fstar = curve.dropna(subset=["fstar", "match_probability"])
    if not fstar.empty:
        fig = px.line(fstar, x="match_probability", y="fstar", title="F* Score vs Threshold",
                      template="simple_white", color_discrete_sequence=[GREEN])
        out.append(to_json(fig.update_layout(height=300, xaxis_range=[0, 1], yaxis_range=[0, 1.05])))
    return out


def waterfall(coverage: pd.DataFrame, toggles: dict, titles: tuple,
              baseline_toggles: Optional[dict] = None) -> dict:
    """Left and right cascading waterfall charts of candidate pairs per blocking rule.

    ``coverage`` is a coverage matrix (one row per pair, from a finished run) or a pattern table
    (one row per combination of rules, from ``blocking_rule_patterns``); both work because
    ``compute_blocking_waterfall`` weights rows by ``n_pairs`` when present. Each pair is credited
    to the first enabled rule, in cascade order, that covers it, so nothing is counted twice and
    switching a rule off hands its pairs to later rules that also cover them.

    Left: the ``baseline_toggles`` rule set, or every rule on when there is no baseline.
    Right: ``toggles``, the live selection. Both share one y-axis so they compare directly.
    """
    order = determine_cascade_order(coverage, list(toggles))
    live = compute_blocking_waterfall(coverage, order, toggles)
    fields = live["fields"]
    if not fields:
        return {"fields": []}
    if baseline_toggles is None:
        base_count, base_total = live["all_active_count"], live["grand_total"]
    else:
        base = compute_blocking_waterfall(coverage, order, baseline_toggles)
        base_count, base_total = base["active_only_count"], base["active_total"]
    live_count, live_total = live["active_only_count"], live["active_total"]

    x_labels = [f"Rule {i + 1}: {f}" for i, f in enumerate(fields)]
    layout = dict(template="simple_white", height=420, showlegend=False,
                  xaxis=dict(title="", tickangle=-40, automargin=True),
                  yaxis=dict(title="Candidate pairs", range=[0, max(base_total, live_total, 1) * 1.12], automargin=True),
                  margin=dict(l=60, r=10, t=10, b=40))
    steps = dict(connector={"line": {"color": "rgba(150,150,150,0.4)"}},
                 increasing={"marker": {"color": BLUE}}, totals={"marker": {"color": BLUE}})

    left_values = [base_count.get(f, 0) for f in fields]
    left = go.Figure(go.Waterfall(x=x_labels + ["Total"], measure=["relative"] * len(fields) + ["total"],
                                  y=left_values + [None], text=[f"{v:,}" for v in left_values] + [f"{base_total:,}"],
                                  **steps)).update_layout(**layout)

    right_values = [live_count.get(f, 0) if toggles.get(f) else 0 for f in fields]
    right = go.Figure(go.Waterfall(x=x_labels + ["Total"], measure=["relative"] * len(fields) + ["total"],
                                   y=right_values + [None],
                                   text=[f"{v:,}" if v else "" for v in right_values] + [f"{live_total:,}"], **steps))
    off = [i for i, f in enumerate(fields) if not toggles.get(f) and base_count.get(f, 0) > 0]
    if off:
        right.add_trace(go.Bar(x=[x_labels[i] for i in off], y=[base_count[fields[i]] for i in off],
                               marker_color=RED, text=[f"{base_count[fields[i]]:,}" for i in off],
                               textposition="inside"))
    lost = max(base_total - live_total, 0) if baseline_toggles is None else 0
    if lost:
        right.add_trace(go.Bar(x=["Total"], y=[lost], base=[live_total], marker_color=RED,
                               text=[f"{lost:,}"], textposition="inside"))
    right.update_layout(**layout)

    rows = [{"rule": f, "before": base_count.get(f, 0), "after": live_count.get(f, 0) if toggles.get(f) else 0,
             "on": bool(toggles.get(f))} for f in fields]
    for row in rows:
        row["change"] = row["after"] - row["before"]
    return {
        "fields": fields, "titles": list(titles), "left": to_json(left), "right": to_json(right),
        "table": rows, "totals": {"before": base_total, "after": live_total, "change": live_total - base_total,
                                  "lost": lost, "all_rules_total": live["grand_total"]},
        "rules_on": {"before": len(fields) if baseline_toggles is None else sum(1 for f in fields if baseline_toggles.get(f)),
                     "after": sum(1 for f in fields if toggles.get(f))},
    }
