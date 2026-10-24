"""Unit checks for analyzers.network.remote_fetch (no real network)."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzers.network import remote_fetch as rf  # noqa: E402


@pytest.mark.parametrize("url,ok", [
    ("", False), ("ftp://example.com/a", False), ("http://127.0.0.1/a", False),
    ("http://192.168.1.10/a", False), ("https://example.com/a.pdf", True),
])
def test_safe_url_for_fetch(url, ok):
    assert rf.safe_url_for_fetch(url)[0] is ok


class _Resp:
    def __init__(self, status=200, ctype="application/pdf", body=b"%PDF-1.4 data"):
        self.status_code, self.history, self.url = status, [], "https://example.com/files/a%20b.pdf"
        self.headers = {"Content-Type": ctype}
        self._body = body

    def iter_content(self, chunk_size=8192):
        yield self._body


def test_fetch_success_and_filters(monkeypatch):
    monkeypatch.setattr(rf.requests, "get", lambda *a, **k: _Resp())
    r = rf.fetch_remote_attachment("https://example.com/files/a.pdf")
    try:
        assert r["ok"] and r["status"] == "success" and r["file_name"] == "a b.pdf" and r["bytes"] == 13
    finally:
        os.remove(r["local_path"])
    monkeypatch.setattr(rf.requests, "get", lambda *a, **k: _Resp(status=404))
    assert rf.fetch_remote_attachment("https://example.com/x")["status"] == "not_found"
    monkeypatch.setattr(rf.requests, "get", lambda *a, **k: _Resp(ctype="text/html"))
    assert rf.fetch_remote_attachment("https://example.com/x")["status"] == "webpage"
    monkeypatch.setattr(rf.requests, "get", lambda *a, **k: _Resp(body=b"x" * 50))
    assert rf.fetch_remote_attachment("https://example.com/x", max_bytes=10)["status"] == "oversized_file"
