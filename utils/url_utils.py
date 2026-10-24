# utils/url_utils.py
import re
from urllib.parse import urlparse


def canonicalize_url(u: str) -> str:
    """
    Must match training:
    - lowercase
    - remove whitespace
    - strip scheme (http/https)
    - strip leading www.
    - strip trailing slashes
    """
    u = "" if u is None else str(u).strip().lower()
    u = re.sub(r"\s+", "", u)
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    u = u.strip("/")
    return u


def extract_host(u: str) -> str:
    """
    Extract host from canonical URL. If no scheme, prepend http:// for parsing.
    Must match training.
    """
    u = canonicalize_url(u)
    if not u:
        return ""
    try:
        parsed = urlparse("http://" + u)
        host = (parsed.netloc or "").lower()
        host = re.sub(r"^www\.", "", host)
        return host
    except Exception:
        return ""
