"""In-memory sessions and a one-at-a-time job queue.

A session holds one user's datasets and runs. Everything lives in this process's memory, so
the number of sessions is capped and idle ones expire. Linkage jobs run one at a time on a
single worker thread: a free host has a few GB of RAM, and two 50,000-row runs side by side
would not fit.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

log = logging.getLogger("cohort_builder")
MAX_SESSIONS = int(os.environ.get("COHORT_BUILDER_MAX_SESSIONS", "4"))
SESSION_TTL_S = int(os.environ.get("COHORT_BUILDER_SESSION_TTL_S", "3600"))


class UserError(ValueError):
    """A problem the user can fix (bad configuration, too many pairs). Shown as-is, not as a crash."""


@dataclass
class Run:
    """Everything one linkage run produced."""
    results: dict
    metrics: dict
    cm: dict
    threshold_curve: Optional[pd.DataFrame]
    curve: dict
    coverage: pd.DataFrame


@dataclass
class Session:
    id: str
    source: str                                   # "people" | "voters" | "upload"
    a: pd.DataFrame
    b: Optional[pd.DataFrame] = None
    field_types: dict = field(default_factory=dict)
    eda: dict = field(default_factory=dict)
    model_json: Optional[dict] = None
    runs: dict = field(default_factory=dict)      # "run1" / "run2" -> Run
    cache: dict = field(default_factory=dict)     # small derived results (rule-pattern tables)
    touched: float = field(default_factory=time.time)
    # the unfiltered frames, so a cohort filter can be changed or cleared
    full_a: Optional[pd.DataFrame] = None


_sessions: "OrderedDict[str, Session]" = OrderedDict()
_lock = threading.Lock()


def create_session(**kwargs) -> Session:
    """Register a new session, evicting expired and then oldest sessions beyond the cap."""
    session = Session(id=uuid.uuid4().hex, **kwargs)
    session.full_a = session.a
    with _lock:
        now = time.time()
        for sid in [s for s, v in _sessions.items() if now - v.touched > SESSION_TTL_S]:
            del _sessions[sid]
        _sessions[session.id] = session
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
    return session


def get_session(session_id: str) -> Session:
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        session.touched = time.time()
        _sessions.move_to_end(session_id)
        return session


def drop_session(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)


# ── Jobs ──────────────────────────────────────────────────────────────────────

@dataclass
class Job:
    id: str
    status: str = "queued"          # queued | running | done | error
    stage: str = "Waiting for a free worker"
    error: Optional[str] = None
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None


_jobs: "OrderedDict[str, Job]" = OrderedDict()
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="linkage")


def submit(work: Callable[[Callable[[str], None]], None]) -> Job:
    """Queue ``work(set_stage)``; the job's status is polled with :func:`get_job`."""
    job = Job(id=uuid.uuid4().hex)
    with _lock:
        _jobs[job.id] = job
        while len(_jobs) > 50:
            _jobs.popitem(last=False)

    def runner() -> None:
        job.status, job.started = "running", time.time()
        try:
            work(lambda text: setattr(job, "stage", text))
            job.status, job.stage = "done", "Finished"
        except UserError as exc:
            job.status, job.error = "error", str(exc)
        except Exception as exc:                      # keep the worker alive whatever a run does
            job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            job.finished = time.time()

    _pool.submit(runner)
    return job


def get_job(job_id: str) -> Job:
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise KeyError(job_id)
    return job
