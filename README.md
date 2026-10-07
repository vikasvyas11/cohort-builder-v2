# Cohort Builder

Record linkage and deduplication in the browser, built on [Splink](https://github.com/moj-analytical-services/splink)
and [DuckDB](https://duckdb.org). Load data, choose which fields to compare and which blocking rules generate
candidate pairs, run a deterministic or probabilistic (Fellegi–Sunter) linkage, check how accurate it is, compare a
second run, and export the cohort. Tested at **50,000 records** in Dataset A and 25,000 in Dataset B.

**Try it live: <https://vikasvyas-cohort-builder-app.hf.space>** ([Space page](https://huggingface.co/spaces/vikasvyas/cohort-builder-app)).
It finds the same person across messy records: link two files, or find duplicates inside one, then check how accurate
the match was.

It ships with **synthetic demo data** (no real people, no downloads, no API keys), so it is safe to host publicly.
`examples/` has two sample files to try the Upload flow.

```
 browser (static HTML + JS + Plotly)  ──HTTP/JSON──▶  FastAPI  ──▶  Splink + DuckDB + pandas
 web/                                                  api/         modules/
 Cloudflare Pages, or served by the API                Hugging Face Space / any Docker host
```

## Three ways in, one flow

| Way in | What happens |
|---|---|
| **Standard** | Pick the people dataset or the North Carolina voter dataset (1,200 / 10,000 / **50,000** voters in A) → profile → configure → run → results → compare → export |
| **Upload** | Upload a CSV/TSV (or give a public URL), optionally a Dataset B or a generated damaged sample of A → automatic cleaning → the same steps |
| **Advanced** | Upload a saved Splink model JSON, skip training, and predict on a dataset → the same results, compare and export steps |

## What you get

- **Probabilistic linkage** (Splink, expectation-maximisation) and **deterministic linkage** (exact rules).
- **Blocking control**: OR/AND, combined-field rules such as `first_name+last_name`, and a pair-count check that
  refuses a configuration that would create more candidate pairs than the host can score.
- **Live cascading waterfall** on the Compare step: toggle rules or add a combined rule and see how many candidate
  pairs each rule contributes and the net effect on the total, against Run 1. Pair counts are taken from the data
  itself (one DuckDB hash join per rule, cached), so rules Run 1 never used are included and a toggle redraws at once.
- **Seven result tabs for every run**: edge metrics, cluster metrics (with an overlapping Venn of clusters by source
  dataset), demographics, blocking explorer, cluster studio, confusion matrix and raw data.
- **Accuracy against ground truth** when the data has a `cluster` column: precision, recall, F1, F\*, confusion matrix,
  precision–recall curve.
- **Run 1 vs Run 2** comparison, a self-contained **HTML report** per run, a saveable **model JSON**, a **cohort CSV**
  with `cluster_id`, and a link to the separate **AI Assistant** that explains results and parameters.

## Run it locally

Python 3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Windows: py -3.12 -m venv .venv ; .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --port 7860
```

Open <http://127.0.0.1:7860>. The API's interactive docs are at `/api/docs`.

## Scale

Measured in-process on a 15 GB Windows laptop with the default voter blocking rules
(`dob`, `first_name+last_name`, `last_name+zip_code`), using `python tools/scale_check.py 50000`:

| Step (50,000 voters in A, 25,000 in B) | Time |
|---|---|
| Generate both datasets and the profile | about 3 s |
| Run 1, deduplicate A, probabilistic | about 3 s |
| Run 1, link A and B, probabilistic (precision 0.995, recall 0.993) | about 3–4 s |
| Live waterfall: first count of 4.4M `first_name` pairs / re-weighting after that | about 6 s / instant |
| Cohort export (10 MB), HTML report (5 MB) | under 1 s each |
| Peak memory for the whole session | under 1 GB |

Blocking decides cost, not record count: `first_name` alone gives about 4.3M pairs at 50,000 records, so the voter
defaults switch to `dob` plus two name/place combinations above 5,000 records. The app refuses a run over
`COHORT_BUILDER_MAX_CANDIDATE_PAIRS` (5,000,000 by default) and says why. These figures are from a laptop, not a
hosted container; re-run `tools/scale_check.py` on your host.

**Why 5,000,000, and can you go past it?** The limit guards memory, not time. Candidate pairs are scored in DuckDB,
but every pair the model keeps comes back into pandas as a wide row, and the metrics, explorer and comparison work on
those rows. Measured with `tools/pair_cost.py` on 50,000 records: 66,000 candidate pairs peak at 0.3 GB and 2 s;
4.4 million (adding `first_name`) take 20 s and peak at 2.7 GB, about 0.6 KB per candidate pair, and that run keeps a
million edges because `first_name` alone mostly links strangers. So memory grows with pairs kept, roughly 1 GB per
million candidates. You can go past 5 million by setting `COHORT_BUILDER_MAX_CANDIDATE_PAIRS` to about one million
per GB of RAM you can spare on the host (10,000,000 on a 16 GB Space). Raising DuckDB's own memory limit did not help
at this size: the cost is the kept pairs, not the join. Tighter blocking beats a higher limit for accuracy as well.

## Configuration

Environment variables, all optional:

| Variable | Default | Meaning |
|---|---|---|
| `COHORT_BUILDER_MAX_CANDIDATE_PAIRS` | 5,000,000 | Largest blocking configuration a run will score |
| `COHORT_BUILDER_MAX_TRUTH_PAIRS` | 5,000,000 | Largest ground-truth pair set built for accuracy |
| `COHORT_BUILDER_DUCKDB_MEMORY_LIMIT` | 2GB | Memory cap for DuckDB queries |
| `COHORT_BUILDER_MAX_SESSIONS` | 4 | Sessions kept in memory (oldest dropped first) |
| `COHORT_BUILDER_SESSION_TTL_S` | 3600 | Idle seconds before a session expires |
| `COHORT_BUILDER_CORS_ORIGINS` | `*` | Comma-separated sites allowed to call the API from a browser |

Uploaded model JSON is screened before it runs (SQL allow-list, no external file access, memory and time limits), and
URLs go through `utils/safe_io.py` (public http/https only, redirects re-checked, 100 MB cap).

## Deploying

See [docs/DEPLOY.md](docs/DEPLOY.md): the API on a free Hugging Face Space (no Docker needed: it runs python app.py), a free Oracle Cloud VM, or any Docker host, and optionally the web UI on Cloudflare Pages under your own domain.

## Development

```bash
pip install -r requirements-dev.txt
pytest          # all tests must pass
ruff check .
```

More in [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Limitations

Sessions live in one process's memory, so run a single worker and expect a restart to clear them. Linkage jobs run
one at a time. Exact-match blocking misses true pairs whose blocking fields are damaged, which is what the accuracy
tab shows. The demo datasets are synthetic; the voter profile is aggregate statistics only (`tools/build_voter_profile.py`).

## License

MIT. See [LICENSE](LICENSE).
