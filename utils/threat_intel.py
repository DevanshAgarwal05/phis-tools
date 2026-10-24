# utils/threat_intel.py
import re
from urllib.parse import urlparse

try:
    import requests
except Exception:
    requests = None

# Put real keys here or leave as "" (empty string)
VT_API_KEY = ""          # e.g. "abc123..."
ABUSEIPDB_API_KEY = ""   # e.g. "xyz789..."

# Offline intel (demo-proof)
KNOWN_BRAND_KEYWORDS = [
    "paypal", "google", "microsoft", "apple", "amazon", "netflix",
    "bank", "sbi", "hdfc", "icici", "axis", "upi"
]

SHORTENER_DOMAINS = {"bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "rebrand.ly"}
SUSPICIOUS_TLDS = {"xyz", "top", "icu", "click", "tk", "ml", "ga", "cf", "gq", "work", "fit"}

def _normalize_url(url: str) -> str:
    url = "" if url is None else str(url).strip()
    if url and not re.match(r"^https?://", url, flags=re.I):
        url = "http://" + url
    return url

def _get_host(url: str) -> str:
    try:
        url = _normalize_url(url)
        p = urlparse(url)
        host = (p.netloc or "").lower()
        host = re.sub(r"^www\.", "", host)
        return host
    except Exception:
        return ""

def offline_intel(url: str) -> dict:
    """
    Works without any API. Use this for stable demos.
    Returns signals you can show in NIS mode.
    """
    host = _get_host(url)
    tld = host.split(".")[-1] if "." in host else ""
    is_shortener = int(host in SHORTENER_DOMAINS)
    suspicious_tld = int(tld in SUSPICIOUS_TLDS)

    brand_hits = [b for b in KNOWN_BRAND_KEYWORDS if b in host]
    many_hyphens = int(host.count("-") >= 3)
    many_subdomains = int(host.count(".") >= 3)

    brand_impersonation = int(bool(brand_hits) and (many_hyphens or many_subdomains or len(url) > 60))

    return {
        "host": host,
        "tld": tld,
        "is_shortener": is_shortener,
        "suspicious_tld": suspicious_tld,
        "brand_hits": brand_hits[:6],
        "brand_impersonation": brand_impersonation,
        "source": "offline"
    }

def check_virustotal(url: str) -> dict:
    """VirusTotal URL check (safe + timeout)."""
    if not VT_API_KEY or VT_API_KEY.strip().upper().startswith("YOUR_"):
        return {"enabled": False, "source": "virustotal", "result": "API key not provided"}

    if requests is None:
        return {"enabled": False, "source": "virustotal", "result": "requests not available"}

    try:
        api_url = "https://www.virustotal.com/api/v3/urls"
        resp = requests.post(
            api_url,
            data={"url": url},
            headers={"x-apikey": VT_API_KEY},
            timeout=6
        )

        if resp.status_code == 200:
            result = resp.json()
            stats = result.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
            malicious = int(stats.get("malicious", 0))
            suspicious = int(stats.get("suspicious", 0))
            return {
                "enabled": True,
                "source": "virustotal",
                "malicious": malicious,
                "suspicious": suspicious,
                "result": f"Malicious={malicious}, Suspicious={suspicious}"
            }

        return {"enabled": True, "source": "virustotal", "result": f"HTTP {resp.status_code}"}

    except Exception as e:
        return {"enabled": True, "source": "virustotal", "result": f"Error: {e}"}

def check_abuseipdb(ip: str) -> dict:
    """AbuseIPDB check (safe + timeout)."""
    if not ABUSEIPDB_API_KEY or ABUSEIPDB_API_KEY.strip().upper().startswith("YOUR_"):
        return {"enabled": False, "source": "abuseipdb", "result": "API key not provided"}

    if requests is None:
        return {"enabled": False, "source": "abuseipdb", "result": "requests not available"}

    try:
        url = "https://api.abuseipdb.com/api/v2/check"
        querystring = {"ipAddress": ip, "maxAgeInDays": "90"}
        headers = {"Key": ABUSEIPDB_API_KEY, "Accept": "application/json"}

        response = requests.get(url, headers=headers, params=querystring, timeout=6)
        data = response.json()

        if "data" in data:
            reports = int(data["data"].get("totalReports", 0))
            confidence = int(data["data"].get("abuseConfidenceScore", 0))
            return {
                "enabled": True,
                "source": "abuseipdb",
                "reports": reports,
                "confidence": confidence,
                "result": f"Reports={reports}, Confidence={confidence}"
            }

        return {"enabled": True, "source": "abuseipdb", "result": "No data found"}

    except Exception as e:
        return {"enabled": True, "source": "abuseipdb", "result": f"Error: {e}"}
