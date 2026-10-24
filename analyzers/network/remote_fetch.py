# analyzers/network/remote_fetch.py
import ipaddress
import os
import re
import tempfile
from urllib.parse import unquote, urlparse

import requests


def safe_url_for_fetch(u: str):
    """
    Validate URL for remote fetch flow.
    Returns (ok: bool, msg: str).
    """
    if not isinstance(u, str) or not u.strip():
        return False, "Empty URL."

    try:
        p = urlparse(u.strip())
    except Exception:
        return False, "Invalid URL format."

    if p.scheme not in ("http", "https"):
        return False, "Only http/https URLs are allowed."
    if not p.netloc:
        return False, "URL must include a valid host."

    host = (p.hostname or "").strip()
    if not host:
        return False, "URL host is missing."

    # Conservative local-network blocking to avoid SSRF-like misuse in demo tool.
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False, "Local/private IP targets are blocked for safety."
    except Exception:
        # hostname case; allow
        pass

    return True, "OK"


def _looks_like_webpage(content_type: str, content_disposition: str, url_path: str) -> bool:
    ct = (content_type or "").lower()
    cd = (content_disposition or "").lower()
    up = (url_path or "").lower()
    if "text/html" in ct and "attachment" not in cd:
        return True
    if up.endswith("/") and "attachment" not in cd:
        return True
    return False


def _is_supported_content_type(content_type: str, content_disposition: str, final_url: str) -> bool:
    ct = (content_type or "").lower()
    cd = (content_disposition or "").lower()
    path = (urlparse(final_url).path or "").lower() if final_url else ""
    if not ct:
        return True  # unclear, allow and rely on local analyzer
    if "application/" in ct or "image/" in ct or "audio/" in ct or "video/" in ct:
        return True
    if "text/plain" in ct:
        return True
    if "application/octet-stream" in ct:
        return True
    return False


def _response_kind(status_code: int, content_type: str, content_disposition: str, final_url: str):
    if status_code == 403:
        return "blocked", "Remote server blocked automated download access (HTTP 403)."
    if status_code == 404:
        return "not_found", "Remote file not found (HTTP 404)."
    if status_code >= 400:
        return "http_error", f"Remote server returned HTTP {status_code}."

    path = urlparse(final_url).path if final_url else ""
    if _looks_like_webpage(content_type, content_disposition, path):
        return "webpage", "Response appears to be a webpage, not a direct downloadable file URL."
    if not _is_supported_content_type(content_type, content_disposition, final_url):
        return "unsupported_content_type", f"Unsupported or unclear downloadable content type: {content_type or 'missing'}"
    return "ok", "Ready for attachment scan."


def fetch_remote_attachment(remote_url: str, max_bytes: int = 25 * 1024 * 1024):
    ok, msg = safe_url_for_fetch(remote_url)
    if not ok:
        return {"ok": False, "status": "invalid_url", "message": f"Invalid URL: {msg}", "debug": {"stage": "validation"}}

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/octet-stream,application/pdf,application/zip,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        r = requests.get(
            remote_url,
            headers=headers,
            timeout=(8, 20),
            allow_redirects=True,
            stream=True,
        )
    except requests.exceptions.Timeout:
        return {"ok": False, "status": "timeout", "message": "Download timed out. Try another URL or retry later.", "debug": {"stage": "request"}}
    except requests.exceptions.SSLError:
        return {"ok": False, "status": "tls_error", "message": "TLS/SSL certificate verification failed for this URL.", "debug": {"stage": "request"}}
    except requests.exceptions.TooManyRedirects:
        return {"ok": False, "status": "too_many_redirects", "message": "Too many redirects while resolving URL.", "debug": {"stage": "request"}}
    except requests.exceptions.ConnectionError as e:
        return {"ok": False, "status": "connection_error", "message": "Connection failed while fetching remote URL.", "debug": {"stage": "request", "error": str(e)}}
    except requests.exceptions.RequestException as e:
        return {"ok": False, "status": "network_error", "message": f"Network fetch failed: {e}", "debug": {"stage": "request", "error": str(e)}}

    if len(r.history) > 5:
        return {"ok": False, "status": "too_many_redirects", "message": "Too many redirects (>5). Refusing for safety.", "debug": {"stage": "redirect", "hops": len(r.history)}}

    final_url = str(r.url or remote_url)
    content_type = (r.headers.get("Content-Type") or "").strip()
    content_disposition = (r.headers.get("Content-Disposition") or "").strip()
    kind, kind_msg = _response_kind(r.status_code, content_type, content_disposition, final_url)
    if kind != "ok":
        return {
            "ok": False,
            "status": kind,
            "message": kind_msg,
            "http_status": int(r.status_code),
            "content_type": content_type,
            "final_url": final_url,
            "debug": {
                "stage": "response_filter",
                "content_disposition": content_disposition,
            },
        }

    parsed = urlparse(final_url)
    base_name = unquote(os.path.basename(parsed.path) or "")
    if not base_name:
        base_name = "downloaded_file.bin"
    safe_name = re.sub(r'[\\/*?:"<>|&=%]', "_", base_name)

    suffix = os.path.splitext(safe_name)[1] if "." in safe_name else ".bin"
    tmp = tempfile.NamedTemporaryFile(prefix="phis_remote_", suffix=suffix, delete=False)
    local_path = tmp.name
    tmp.close()

    try:
        total = 0
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("oversized_file")
                f.write(chunk)
    except ValueError as ve:
        if str(ve) == "oversized_file":
            try:
                if os.path.exists(local_path):
                    os.remove(local_path)
            except Exception:
                pass
            return {
                "ok": False,
                "status": "oversized_file",
                "message": "File too large (>25MB). Refusing download for safety.",
                "final_url": final_url,
                "content_type": content_type,
                "debug": {"stage": "download", "bytes_read": total, "limit": max_bytes},
            }
        raise

    # Heuristic: HTML body despite non-strict headers (non-direct page).
    if (content_type or "").lower().startswith("text/html"):
        try:
            with open(local_path, "rb") as f:
                peek = f.read(1024).lower()
            if b"<html" in peek or b"<!doctype html" in peek:
                os.remove(local_path)
                return {
                    "ok": False,
                    "status": "non_direct_url",
                    "message": "URL appears to be a webpage, not a direct file download link.",
                    "final_url": final_url,
                    "content_type": content_type,
                    "debug": {"stage": "content_probe"},
                }
        except Exception:
            pass

    return {
        "ok": True,
        "status": "success",
        "message": "File downloaded successfully.",
        "local_path": local_path,
        "file_name": safe_name,
        "bytes": total,
        "http_status": int(r.status_code),
        "final_url": final_url,
        "content_type": content_type,
        "content_disposition": content_disposition,
        "debug": {"stage": "completed"},
    }
