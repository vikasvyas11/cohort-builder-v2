# Contributing

Thanks for helping improve Cohort Builder. Bug reports, fixes, tests and documentation are all welcome.

## Getting set up

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn api.main:app --reload --port 7860
```

## Before you open a pull request

```bash
pytest          # all tests must pass
ruff check .    # no lint errors
```

- Add a test for any behaviour you change. Metrics tests in `tests/test_metrics_engine.py` use tiny hand-built cases
  whose answers can be worked out on paper; `tests/test_blocking_patterns.py` does the same for the waterfall. Follow
  that style. `tests/test_api.py` clicks through every flow over HTTP.
- Keep HTTP details in `api/main.py`, app logic in `api/service.py`, and anything that does not need a request in
  `modules/` so it can be unit tested alone. The UI in `web/` only draws what the API returns.
- Never read user-supplied URLs or paths directly. Go through `utils/safe_io.py`.
- In the browser code, set text with `textContent` (the `h()` helper does), never `innerHTML`: values come from user data.
- Do not commit data. The app ships synthetic data only; `.gitignore` excludes CSVs for that reason. Never add real
  personal records, even if they are public.

## Provenance and licensing

This project is MIT licensed. By contributing you confirm that you wrote your contribution (or have the right to
submit it under the MIT licence) and that it does not copy code, report templates or data from proprietary or
restrictively licensed sources, including code from an employer or research group you cannot publish. If your change
is inspired by a published method, cite the paper in the docstring rather than reproducing someone's implementation.

## Style

- Python 3.12, type hints on public functions, docstrings that say *why* as well as *what*.
- Prefer small pure functions; avoid hidden global state and swallowed exceptions. If you must catch a broad exception
  to keep the server alive, log it.
- The web UI has no build step: plain ES modules, no framework. Keep it that way unless a feature needs more.

## Reporting bugs

Include your Python and package versions (`pip freeze | grep -iE "fastapi|splink|duckdb|pandas"`), the steps you took,
and any traceback. Do not paste real personal data into an issue.
