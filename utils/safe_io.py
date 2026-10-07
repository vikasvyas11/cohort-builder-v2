"""Guarded readers for user-supplied data locations.

The Upload flow lets a visitor point the app at a URL or a file path.  On a
public deployment both are dangerous if taken at face value:

* ``urllib`` happily opens ``file://`` URLs and internal addresses
  (``http://169.254.169.254/...``, ``http://localhost:8501/...``), turning the
  app into a file reader / request forger (SSRF).
* A "local path" box lets anyone read any file the server process can read -
  including the app's own secrets - and preview its contents.

So: URLs are limited to public http(s) hosts, size-capped, and every redirect
is re-validated; local paths are only offered when the operator opts in with
``COHORT_BUILDER_ALLOW_LOCAL_FILES=1`` (intended for running on your own
machine).
"""

from __future__ import annotations

import io
import ipaddress
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024    # 100 MB
_ALLOWED_SCHEMES = ("http", "https")
_READ_KWARGS = dict(dtype=str, encoding="latin-1", on_bad_lines="skip")


class UnsafeSourceError(ValueError):
    """The requested location is not allowed."""


def local_files_enabled() -> bool:
    """True when the operator has opted in to reading server-side file paths."""
    return os.environ.get("COHORT_BUILDER_ALLOW_LOCAL_FILES", "").strip().lower() in {"1", "true", "yes"}


def _check_public_host(url: str) -> None:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        raise UnsafeSourceError("Only http(s) URLs are supported.")
    host = parts.hostname
    if not host:
        raise UnsafeSourceError("The URL has no host name.")
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, parts.port or 443)}
    except socket.gaierror as exc:
        raise UnsafeSourceError(f"Could not resolve host '{host}'.") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if not ip.is_global:
            raise UnsafeSourceError("URLs that resolve to private or local addresses are not allowed.")


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    """Re-validate the target of every redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_public_host(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _default_separator(name: str) -> str:
    return "\t" if name.lower().endswith(".txt") else ","


def read_remote_table(url: str, nrows: Optional[int] = None, sep: Optional[str] = None) -> pd.DataFrame:
    """Download a delimited text file from a public http(s) URL into a DataFrame."""
    _check_public_host(url)
    opener = urllib.request.build_opener(_CheckedRedirects)
    with opener.open(url, timeout=30) as response:
        raw = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(raw) > MAX_DOWNLOAD_BYTES:
        raise UnsafeSourceError(f"The file is larger than {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB.")
    return pd.read_csv(io.BytesIO(raw), sep=sep or _default_separator(urllib.parse.urlsplit(url).path),
                       nrows=nrows or None, **_READ_KWARGS)


def read_local_table(path: str, nrows: Optional[int] = None, sep: Optional[str] = None) -> pd.DataFrame:
    """Read a delimited text file from the server's disk (only if opted in)."""
    if not local_files_enabled():
        raise UnsafeSourceError(
            "Reading local file paths is disabled. Set COHORT_BUILDER_ALLOW_LOCAL_FILES=1 "
            "when running the app on your own machine to enable it.")
    file = Path(path).expanduser()
    if not file.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(file, sep=sep or _default_separator(file.name), nrows=nrows or None, **_READ_KWARGS)


def read_uploaded_table(file_obj) -> pd.DataFrame:
    """Parse a Streamlit ``UploadedFile`` as CSV or TSV."""
    raw = file_obj.read()
    separators = ["\t", ","] if file_obj.name.lower().endswith(".txt") else [",", "\t"]
    for sep in separators:
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep, **_READ_KWARGS)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
            continue
        if df.shape[1] > 1:
            return df
    raise ValueError(f"Could not parse {file_obj.name} as CSV or TSV.")
