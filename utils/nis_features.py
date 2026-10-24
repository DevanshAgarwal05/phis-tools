# utils/nis_features.py
import re
import math
from urllib.parse import urlparse

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "update", "password", "account", "secure", "bank",
    "confirm", "signin", "sign-in", "unlock", "invoice", "payment", "kyc",
    "otp", "refund", "security", "alert"
]

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rebrand.ly"
}

SUSPICIOUS_TLDS = {
    "xyz", "top", "icu", "click", "tk", "ml", "ga", "cf", "gq", "work", "fit"
}

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    ent = 0.0
    n = len(s)
    for c in freq.values():
        p = c / n
        ent -= p * math.log2(p)
    return float(ent)

def _get_host(url: str) -> str:
    try:
        # handle urls without scheme
        if not re.match(r"^https?://", url, flags=re.I):
            url = "http://" + url
        p = urlparse(url)
        host = (p.netloc or "").lower()
        host = re.sub(r"^www\.", "", host)
        return host
    except Exception:
        return ""

def _get_path_query(url: str) -> str:
    try:
        if not re.match(r"^https?://", url, flags=re.I):
            url = "http://" + url
        p = urlparse(url)
        return (p.path or "") + ("?" + p.query if p.query else "")
    except Exception:
        return ""

def extract_url_nis_features(url: str) -> dict:
    url = "" if url is None else str(url).strip()
    u = url.lower()

    host = _get_host(u)
    pathq = _get_path_query(u)

    has_https = int(u.startswith("https://"))
    has_ip_host = int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host)))
    has_at = int("@" in u)
    has_double_slash = int("//" in pathq)  # suspicious in path
    num_dots = host.count(".")
    url_len = len(u)

    # keyword hits
    kw_hits = [k for k in SUSPICIOUS_KEYWORDS if k in u]
    kw_count = len(kw_hits)

    # entropy (host)
    ent = shannon_entropy(host)

    # shortener
    is_shortener = int(host in SHORTENER_DOMAINS)

    # suspicious tld
    tld = host.split(".")[-1] if "." in host else ""
    suspicious_tld = int(tld in SUSPICIOUS_TLDS)

    # lots of digits
    digit_ratio = (sum(ch.isdigit() for ch in u) / max(1, len(u)))
    high_digit_ratio = int(digit_ratio > 0.18)

    # many hyphens
    hyphen_count = host.count("-")
    many_hyphens = int(hyphen_count >= 3)

    return {
        "host": host,
        "tld": tld,
        "url_len": url_len,
        "has_https": has_https,
        "has_ip_host": has_ip_host,
        "has_at": has_at,
        "has_double_slash_in_path": has_double_slash,
        "num_subdomain_dots": num_dots,
        "keyword_count": kw_count,
        "keyword_hits": kw_hits[:8],
        "host_entropy": ent,
        "is_shortener": is_shortener,
        "suspicious_tld": suspicious_tld,
        "high_digit_ratio": high_digit_ratio,
        "many_hyphens": many_hyphens,
    }

def extract_email_nis_features(subject: str, body: str, headers: str) -> dict:
    subject = "" if subject is None else str(subject)
    body = "" if body is None else str(body)
    headers = "" if headers is None else str(headers)

    text = (subject + " " + body).lower()

    # urgency cues
    urgency_words = ["urgent", "immediately", "asap", "action required", "limited time", "final notice"]
    urgency_hits = [w for w in urgency_words if w in text]

    # credential harvesting cues
    cred_words = ["password", "otp", "verify", "login", "sign in", "account locked", "confirm identity"]
    cred_hits = [w for w in cred_words if w in text]

    # attachment cues
    attach_words = ["invoice", "receipt", "payment", "document", "attached", "pdf", "zip", "scan"]
    attach_hits = [w for w in attach_words if w in text]

    # header spoof cues (simple)
    from_match = re.search(r"^From:\s*(.+)$", headers, flags=re.I | re.M)
    reply_match = re.search(r"^Reply-To:\s*(.+)$", headers, flags=re.I | re.M)
    from_val = from_match.group(1).strip().lower() if from_match else ""
    reply_val = reply_match.group(1).strip().lower() if reply_match else ""
    reply_differs = int(bool(from_val and reply_val and from_val != reply_val))

    return {
        "urgency_hits": urgency_hits[:6],
        "cred_hits": cred_hits[:6],
        "attach_hits": attach_hits[:6],
        "reply_to_differs": reply_differs,
        "has_headers": int(bool(headers.strip())),
    }
