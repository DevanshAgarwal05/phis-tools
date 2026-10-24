# analyzers/url/selectors.py
import re
from urllib.parse import urlparse

import numpy as np


# ✅ Needed for loading the URL pipeline model (pickle expects these in module "main")
def sel_url(X):
    return X["url"].astype(str).values

def sel_host(X):
    return X["host"].astype(str).values


SUSPICIOUS_KEYWORDS_URL = [
    "login", "signin", "verify", "secure", "update", "account", "password",
    "confirm", "bank", "invoice", "payment", "otp", "reset", "webscr"
]
SUSPICIOUS_TLDS_URL = {
    "tk", "ml", "ga", "cf", "gq", "work", "top", "xyz", "click", "zip", "review"
}
SHORTENER_DOMAINS_URL = {
    "bit.ly", "t.co", "goo.gl", "tinyurl.com", "ow.ly", "is.gd", "cutt.ly", "rb.gy", "shorturl.at"
}


def sel_struct(X):
    """
    Compatibility selector for newer URL models with structural feature branch.
    """
    out = []
    for _, row in X.iterrows():
        url = str(row.get("url", "") or "")
        host = str(row.get("host", "") or "")
        parsed = urlparse("http://" + url) if url else None
        path = (parsed.path or "") if parsed else ""
        host_parts = [p for p in host.split(".") if p]
        tld = host_parts[-1] if host_parts else ""
        subdomain_depth = max(0, len(host_parts) - 2)
        digit_count = sum(ch.isdigit() for ch in url)
        hyphen_count = url.count("-")
        keyword_count = sum(1 for k in SUSPICIOUS_KEYWORDS_URL if k in url)
        ip_host = 1 if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host or "") else 0
        suspicious_tld = 1 if tld in SUSPICIOUS_TLDS_URL else 0
        punycode = 1 if "xn--" in host else 0
        shortener = 1 if host in SHORTENER_DOMAINS_URL else 0
        path_depth = len([p for p in path.split("/") if p])
        out.append([
            len(host), len(url), subdomain_depth, (digit_count / max(1, len(url))),
            hyphen_count, keyword_count, ip_host, suspicious_tld, punycode, path_depth, shortener
        ])
    return np.asarray(out, dtype=float)
