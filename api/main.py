"""HTTP API and static web UI for Cohort Builder.

    uvicorn api.main:app --port 7860

The routes are thin: parsing and error mapping only. The logic lives in :mod:`api.service`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.routing import Mount, Route

from api import service, store
from api.store import UserError
from utils import safe_io

MAX_UPLOAD_BYTES = safe_io.MAX_DOWNLOAD_BYTES
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
ASSISTANT_URL = "https://huggingface.co/spaces/vikasvyas/AI_Assistant_Cohort_Builder"
SLOTS = ("run1", "run2")

app = FastAPI(title="Cohort Builder", docs_url="/api/docs", openapi_url="/api/openapi.json")
origins = [o.strip() for o in os.environ.get("COHORT_BUILDER_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(UserError)
async def _user_error(_, exc: UserError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(safe_io.UnsafeSourceError)
async def _unsafe_source(_, exc):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(KeyError)
async def _missing(_, exc: KeyError):
    return JSONResponse({"detail": "That session or job has expired. Start again from the first step."}, status_code=404)


def _session(session_id: str) -> store.Session:
    return store.get_session(session_id)


def _slot(slot: str) -> str:
    if slot not in SLOTS:
        raise HTTPException(404, "Unknown run.")
    return slot


class _Upload:
    """The bits of a Streamlit upload that ``safe_io.read_uploaded_table`` reads."""

    def __init__(self, name: str, data: bytes):
        self.name, self._data = name, data

    def read(self) -> bytes:
        return self._data


async def _read_table(file: Optional[UploadFile], url: Optional[str], max_rows: int):
    if file is not None and file.filename:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise UserError(f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        try:
            return safe_io.read_uploaded_table(_Upload(file.filename, data))
        except ValueError as exc:
            raise UserError(str(exc)) from exc
    if url:
        try:
            return safe_io.read_remote_table(url.strip(), nrows=max_rows)
        except safe_io.UnsafeSourceError:
            raise
        except Exception as exc:                          # network and parse errors, shown to the user
            raise UserError(f"Could not load that URL: {exc}") from exc
    return None


# ── health and config ─────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    return {"assistant_url": ASSISTANT_URL, "voter_sizes": list(service.VOTER_SIZES),
            "max_candidate_pairs": service.MAX_CANDIDATE_PAIRS}


# ── datasets ──────────────────────────────────────────────────────────────────

@app.post("/api/sessions")
def create_demo(body: dict = Body(...)):
    session = service.load_demo(str(body.get("source", "")), int(body.get("n_rows") or 1200))
    return service.overview(session)


@app.post("/api/sessions/upload")
async def create_upload(
    file_a: Optional[UploadFile] = File(None), url_a: Optional[str] = Form(None),
    file_b: Optional[UploadFile] = File(None), url_b: Optional[str] = Form(None),
    id_col_a: Optional[str] = Form(None), id_col_b: Optional[str] = Form(None),
    mode: str = Form("dedupe_only"), max_rows: int = Form(100_000),
):
    if mode not in ("dedupe_only", "uploaded", "sample"):
        raise UserError("Unknown way of setting up Dataset B.")
    max_rows = min(max(int(max_rows), 1_000), 1_000_000)
    raw_a = await _read_table(file_a, url_a, max_rows)
    if raw_a is None:
        raise UserError("Upload Dataset A or give a URL to it.")
    raw_b = await _read_table(file_b, url_b, max_rows) if mode == "uploaded" else None
    session = service.load_upload(raw_a, raw_b, id_col_a or None, id_col_b or None, mode)
    return service.overview(session)


@app.get("/api/sessions/{sid}")
def get_overview(sid: str):
    return service.overview(_session(sid))


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    store.drop_session(sid)
    return {"deleted": True}


@app.post("/api/sessions/{sid}/dataset-b")
def make_dataset_b(sid: str, body: dict = Body(...)):
    session = _session(sid)
    result = service.generate_b(session, min(0.9, max(0.1, float(body.get("sample_frac", 0.5)))),
                                body.get("rates") or {}, int(body.get("letters", 1)), int(body.get("year_shift", 1)))
    return {**result, **service.overview(session)}


@app.post("/api/sessions/{sid}/cohort")
def set_cohort(sid: str, body: dict = Body(...)):
    session = _session(sid)
    result = service.apply_cohort(session, body)
    return {**result, **service.overview(session)}


@app.post("/api/sessions/{sid}/model")
async def upload_model(sid: str, file: UploadFile = File(...)):
    session = _session(sid)
    data = await file.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        raise UserError("A model file should be well under 10 MB.")
    parsed = service.read_model(data)
    session.model_json = parsed["model"]
    return parsed["summary"]


# ── running ───────────────────────────────────────────────────────────────────

@app.post("/api/sessions/{sid}/estimate")
def estimate(sid: str, body: dict = Body(...)):
    try:
        return service.estimate(_session(sid), body)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, UserError):
            raise
        raise UserError(f"The configuration is not valid: {exc}") from exc


@app.post("/api/sessions/{sid}/runs/{slot}")
def start_run(sid: str, slot: str, body: dict = Body(...)):
    session, slot = _session(sid), _slot(slot)
    if slot == "run2" and "run1" not in session.runs:
        raise UserError("Complete Run 1 before starting Run 2.")
    try:
        service.clean_config(body)                        # reject a malformed configuration before queueing
    except (TypeError, ValueError) as exc:
        raise UserError(f"The configuration is not valid: {exc}") from exc

    return {"job_id": store.submit(lambda stage: service.execute_run(session, slot, body, stage)).id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = store.get_job(job_id)
    return {"status": job.status, "stage": job.stage, "error": job.error,
            "elapsed_s": round((job.finished or time.time()) - job.started, 1)}


# ── results ───────────────────────────────────────────────────────────────────

@app.get("/api/sessions/{sid}/runs/{slot}")
def run_summary(sid: str, slot: str):
    return service.run_summary(_session(sid), _slot(slot))


@app.get("/api/sessions/{sid}/runs/{slot}/demographics")
def run_demographics(sid: str, slot: str):
    return service.run_demographics(_session(sid), _slot(slot))


@app.post("/api/sessions/{sid}/runs/{slot}/explorer")
def run_explorer(sid: str, slot: str, body: dict = Body(default={})):
    return service.explorer(_session(sid), _slot(slot), body.get("toggles"), float(body.get("threshold", 0.8)))


@app.post("/api/sessions/{sid}/runs/{slot}/recluster")
def run_recluster(sid: str, slot: str, body: dict = Body(...)):
    return service.recluster(_session(sid), _slot(slot), body.get("toggles") or {}, float(body.get("threshold", 0.8)))


@app.get("/api/sessions/{sid}/runs/{slot}/raw")
def run_raw(sid: str, slot: str, limit: int = Query(100, ge=1, le=1000)):
    return service.raw_tables(_session(sid), _slot(slot), limit)


@app.get("/api/sessions/{sid}/runs/{slot}/studio", response_class=HTMLResponse)
def run_studio(sid: str, slot: str):
    return HTMLResponse(service.studio_html(_session(sid), _slot(slot)) or "<p>No cluster studio for this run.</p>")


@app.post("/api/sessions/{sid}/waterfall")
def waterfall(sid: str, body: dict = Body(...)):
    return service.live_waterfall(_session(sid), body.get("toggles") or {})


@app.get("/api/sessions/{sid}/compare")
def compare(sid: str):
    return service.compare(_session(sid))


# ── downloads ─────────────────────────────────────────────────────────────────

def _download(content: bytes, name: str, media: str) -> Response:
    return Response(content, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/sessions/{sid}/runs/{slot}/cohort.csv")
def download_cohort(sid: str, slot: str):
    slot = _slot(slot)
    return _download(service.cohort_csv(_session(sid), slot), f"cohort_{slot}.csv", "text/csv")


@app.get("/api/sessions/{sid}/runs/{slot}/report.html")
def download_report(sid: str, slot: str):
    slot = _slot(slot)
    return _download(service.report_html(_session(sid), slot), f"linkage_report_{slot}.html", "text/html")


@app.get("/api/sessions/{sid}/runs/{slot}/model.json")
def download_model(sid: str, slot: str):
    slot = _slot(slot)
    return _download(json.dumps(service.model_json(_session(sid), slot), indent=2).encode("utf-8"),
                     f"splink_model_{slot}.json", "application/json")


def web_routes() -> list:
    """The static UI as explicit routes (the page, its stylesheet and config, the scripts).

    Explicit rather than one catch-all mount at "/" so that, when another framework owns the server (Gradio
    on a Hugging Face Space), its own paths such as /config are not swallowed by the UI.
    """
    def single(path: str, file: str):
        return Route(path, lambda request, f=WEB_DIR / file: FileResponse(f), name=file)

    return [single("/", "index.html"), single("/style.css", "style.css"), single("/config.js", "config.js"),
            Mount("/js", StaticFiles(directory=WEB_DIR / "js"), name="js")]


if WEB_DIR.is_dir():                                      # the UI, when served from the same host
    app.router.routes.extend(web_routes())
