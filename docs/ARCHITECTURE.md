# Architecture

## Overview

```
web/                      static UI: index.html, style.css, js/*.js (ES modules, Plotly from a CDN)
  state.js  main.js       shared state, hash router, the left rail (progress thread)
  ui.js  hero.js          DOM and chart helpers (chips, switches, live charts); the first screen's linkage demo
  data.js  profile.js  configure.js  results.js  compare.js      one module per step

api/
  main.py                 FastAPI routes: parse input, map errors, serve web/
  service.py              the logic: load data, run linkage, summarise every tab, compare, export
  store.py                in-memory sessions (capped, expiring) and a one-at-a-time job queue
  figures.py              Plotly figures as JSON, incl. the cascading waterfall

modules/                  the engine, free of any web framework
  splink_runner.py        Splink settings, training, prediction, clustering, blocking counts, sandbox for model JSON
  metrics_engine.py       run metrics, ground-truth pairs, confusion matrix, precision-recall curve
  eda_engine.py  corruption.py  synthetic_data.py  voter_data.py  data_builder.py
  demographics.py  cohort_filter.py  report.py
utils/safe_io.py          guarded loading of uploaded files and URLs
```

The three ways in (Standard, Upload, Advanced) share one wizard: Data → Profile → Configure → Results → Compare →
Export. Only the first step and, for Advanced, the Configure step differ.

## A request's life

1. `POST /api/sessions` (demo data) or `/api/sessions/upload` builds a `Session` holding Dataset A, Dataset B and
   the detected field types, and returns an overview (columns, defaults, profile charts).
2. The configure step calls `POST …/estimate` as the user edits rules. `estimate_candidate_pairs` does one DuckDB hash
   join per rule, so even 50,000 records answer in well under a second. Over the budget, the run is refused.
3. `POST …/runs/{run1|run2}` queues a job. The single worker thread runs `service.execute_run`: validate →
   estimate → `run_linkage` (or `run_linkage_from_json`) → metrics → confusion matrix and threshold curve →
   coverage matrix. The page polls `GET /api/jobs/{id}` for the stage.
4. The tabs read from the stored `Run`: `GET …/runs/{slot}` (cards, edge metrics, cluster metrics, accuracy),
   `…/demographics`, `…/explorer`, `…/studio`, `…/raw`.
5. Downloads (`cohort.csv`, `report.html`, `model.json`) are built on request from the stored run.

## The live waterfall

Blocking rules are exact matches, so which rules cover a pair is a property of the data, not of a finished run.
`blocking_rule_patterns` (in `splink_runner.py`) counts, for the set of rules in play, how many pairs show each
combination of rules: one hash join per rule, a `UNION` of the pair ids, then a `GROUP BY` over the per-rule
agreement flags. The result has one row per *pattern* with an `n_pairs` count, so it stays small however many pairs
there are. Rules switched on get priority within the pair budget; a rule too big for the budget is reported, not run.

`compute_blocking_waterfall` then credits each pair to the first enabled rule, in cascade order (largest standalone
count first), weighted by `n_pairs`. Toggling a rule re-weights that small table; only changing the set of rules (a new
combined rule) re-counts, and the counts are cached in the session. The same function draws the Run 1 explorer from a
finished run's coverage matrix, where each row is one pair.

## Metrics design

A pair of records is identified by an order-independent **pair key** (each record's `"<source>|<id>"`, sorted and
joined). Edge sets from different runs, and ground-truth pairs derived from a `cluster` column, are compared with
set operations. Recall is measured against all true pairs, so a true pair that blocking never generated counts as a miss.

## Deterministic linkage

`deterministic_link()` accepts any pair satisfying one blocking rule, but only if it agrees exactly on at least two
of the fields whose blocking rules are on (one if only one is on), so one common shared value cannot chain strangers
into a giant cluster.

## Scale

The engine keeps candidate pairs in DuckDB until they are scored; `df_predict` holds only pairs above a match-weight
threshold. Pair-level work in pandas (coverage matrix, run comparison, explorer table) is bounded by the pair budget.
The explorer table shows 200 rows; raw tables 100. `tools/scale_check.py` measures a full session at 50,000 records.

## Security notes

Uploaded model JSON runs through a DuckDB sandbox: its SQL is screened, external access is disabled, memory and time
are capped. User-supplied URLs go through `utils/safe_io.py` (http/https only, public addresses only, redirects
re-validated, 100 MB cap). Cell values and column names reach the browser only through `textContent`. Known residual
risk: DNS rebinding between the address check and the connection. There is no authentication: a session id is an
unguessable token, but anyone who can reach the host can start sessions, so cap sessions and pairs on a public host.
