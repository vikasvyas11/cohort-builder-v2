import io
import socket

import pandas as pd
import pytest

from utils import safe_io


def _resolve_to(monkeypatch, address):
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port: [(None, None, None, None, (address, 0))])


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com/x.csv", "gopher://example.com", "//example.com/x.csv", "not a url",
])
def test_non_http_schemes_are_rejected(url):
    with pytest.raises(safe_io.UnsafeSourceError):
        safe_io.read_remote_table(url)


@pytest.mark.parametrize("address", [
    "127.0.0.1", "10.0.0.5", "192.168.1.20", "172.16.0.1", "169.254.169.254", "::1", "0.0.0.0", "100.64.0.1",
])
def test_private_and_local_addresses_are_rejected(monkeypatch, address):
    _resolve_to(monkeypatch, address)
    with pytest.raises(safe_io.UnsafeSourceError):
        safe_io.read_remote_table("http://innocent.example/data.csv")


def test_public_address_passes_host_check(monkeypatch):
    _resolve_to(monkeypatch, "93.184.216.34")
    safe_io._check_public_host("https://example.com/data.csv")        # does not raise


def test_redirect_to_private_host_is_blocked(monkeypatch):
    _resolve_to(monkeypatch, "127.0.0.1")
    handler = safe_io._CheckedRedirects()
    with pytest.raises(safe_io.UnsafeSourceError):
        handler.redirect_request(None, None, 302, "Found", {}, "http://localhost:8501/secret")


def test_local_paths_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("COHORT_BUILDER_ALLOW_LOCAL_FILES", raising=False)
    target = tmp_path / "x.csv"
    target.write_text("a,b\n1,2\n")
    assert not safe_io.local_files_enabled()
    with pytest.raises(safe_io.UnsafeSourceError):
        safe_io.read_local_table(str(target))


def test_local_paths_work_when_opted_in(monkeypatch, tmp_path):
    monkeypatch.setenv("COHORT_BUILDER_ALLOW_LOCAL_FILES", "1")
    target = tmp_path / "x.csv"
    target.write_text("a,b\n1,2\n3,4\n")
    assert len(safe_io.read_local_table(str(target))) == 2
    with pytest.raises(FileNotFoundError):
        safe_io.read_local_table(str(tmp_path / "missing.csv"))


class _Upload(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name


def test_uploaded_file_parsing_csv_and_tsv():
    assert safe_io.read_uploaded_table(_Upload(b"a,b\n1,2\n", "x.csv")).shape == (1, 2)
    assert safe_io.read_uploaded_table(_Upload(b"a\tb\n1\t2\n", "x.txt")).shape == (1, 2)
    assert isinstance(safe_io.read_uploaded_table(_Upload(b"a,b\n1,2\n", "x.csv")), pd.DataFrame)
    with pytest.raises(ValueError):
        safe_io.read_uploaded_table(_Upload(b"onlyonecolumn\n1\n", "x.csv"))
