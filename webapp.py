import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import re
import json
import ipaddress
import tempfile
import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from urllib.parse import urlparse, unquote

# ✅ NIS modules (you created these)
from utils.nis_features import extract_url_nis_features, extract_email_nis_features
from utils.risk_scoring import compute_risk_score, build_reasons
from utils.scan_logger import append_log, now_iso

# ✅ threat_intel.py (safe version)
from utils.threat_intel import offline_intel, check_virustotal, check_abuseipdb


try:
    from keras.saving import register_keras_serializable
except Exception:
    from tensorflow.keras.utils import register_keras_serializable

try:
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
except Exception:
    tokenizer_from_json = None

try:
    from utils.attachment_file_analyzer import analyze_attachment_file
    ATTACHMENT_ANALYZER_AVAILABLE = True
except Exception as e:
    ATTACHMENT_ANALYZER_AVAILABLE = False
    ATTACHMENT_IMPORT_ERROR = str(e)


# ----------------------------
# Keras compatibility patches
# ----------------------------
@register_keras_serializable(package="Custom")
class PatchedInputLayer(tf.keras.layers.InputLayer):
    @classmethod
    def from_config(cls, config):
        if "batch_shape" in config and "batch_input_shape" not in config:
            config["batch_input_shape"] = config.pop("batch_shape")
        return super().from_config(config)


@register_keras_serializable(package="Custom")
def DTypePolicy(name="float32"):
    return tf.keras.mixed_precision.Policy(name)


# ----------------------------
# Helpers
# ----------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_tfidf_fitted(vec):
    if vec is None:
        return False
    if hasattr(vec, "idf_"):
        return True
    tfidf = getattr(vec, "_tfidf", None)
    if tfidf is not None and hasattr(tfidf, "idf_"):
        return True
    return False


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


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


def policy_thresholds(policy_name: str):
    """
    Organization Policy Mode
    - Strict: flags earlier (banking/enterprise)
    - Normal: balanced (default)
    - Relaxed: fewer false positives (personal)
    """
    policy_name = (policy_name or "").lower()
    if "strict" in policy_name:
        return {"suspicious": 50.0, "phishing": 70.0}
    if "relaxed" in policy_name:
        return {"suspicious": 70.0, "phishing": 85.0}
    # Normal
    return {"suspicious": 60.0, "phishing": 80.0}


def policy_threat_level(score: float, suspicious_th: float, phishing_th: float) -> str:
    """
    Threat level aligned with the selected policy (demo-friendly).
    """
    if score >= phishing_th + 10:
        return "CRITICAL"
    if score >= phishing_th:
        return "HIGH"
    if score >= suspicious_th:
        return "MEDIUM"
    return "LOW"


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


def grouped_explainability(email_prob, url_prob_mean, header_info, url_feat, email_feat, intel, risk_score, policy_name):
    groups = {
        "Email Indicators": [],
        "Header/Spoofing Indicators": [],
        "URL/Domain Indicators": [],
        "Threat Intel Indicators": [],
        "Final Risk Reasoning": [],
    }

    if email_prob is not None:
        groups["Email Indicators"].append(f"Email ML probability used: {email_prob:.2f}")
    if email_feat.get("urgency_hits"):
        groups["Email Indicators"].append("Urgency cues: " + ", ".join(email_feat.get("urgency_hits", [])))
    if email_feat.get("cred_hits"):
        groups["Email Indicators"].append("Credential cues: " + ", ".join(email_feat.get("cred_hits", [])))
    if email_feat.get("attach_hits"):
        groups["Email Indicators"].append("Attachment cues: " + ", ".join(email_feat.get("attach_hits", [])))

    groups["Header/Spoofing Indicators"].append(
        f"Header risk: {header_info.get('risk', 'Low Risk')} (confidence: {header_info.get('confidence', 'low')})"
    )
    for i in header_info.get("issues", []):
        groups["Header/Spoofing Indicators"].append(i)
    if header_info.get("manual_or_incomplete"):
        groups["Header/Spoofing Indicators"].append("Header appears manually entered/incomplete; spoofing confidence reduced.")

    if url_prob_mean is not None:
        groups["URL/Domain Indicators"].append(f"Average URL ML probability: {url_prob_mean:.2f}")
    if url_feat.get("host"):
        groups["URL/Domain Indicators"].append(f"Top URL host: {url_feat.get('host')}")
    if url_feat.get("has_ip_host") == 1:
        groups["URL/Domain Indicators"].append("Top URL uses IP address as host.")
    if url_feat.get("is_shortener") == 1:
        groups["URL/Domain Indicators"].append("Top URL host is a known shortener.")
    if url_feat.get("suspicious_tld") == 1:
        groups["URL/Domain Indicators"].append(f"Suspicious TLD detected: .{url_feat.get('tld', '')}")
    if url_feat.get("keyword_hits"):
        groups["URL/Domain Indicators"].append("Suspicious URL keywords: " + ", ".join(url_feat.get("keyword_hits", [])))

    if intel.get("source"):
        groups["Threat Intel Indicators"].append(f"Threat intel source: {intel.get('source')}")
    if intel.get("brand_hits"):
        groups["Threat Intel Indicators"].append("Brand hits: " + ", ".join(intel.get("brand_hits", [])))
    if intel.get("brand_impersonation") == 1:
        groups["Threat Intel Indicators"].append("Brand impersonation heuristic triggered.")

    groups["Final Risk Reasoning"].append(f"Combined risk score: {risk_score:.1f}/100")
    groups["Final Risk Reasoning"].append(f"Policy mode: {policy_name}")
    groups["Final Risk Reasoning"].append(
        "Final score blends ML probabilities with NIS heuristic bumps (headers, URL/domain signals, intel)."
    )
    return groups


def classify_attack_types(url_feat: dict, email_feat: dict, intel: dict, header_risk: str) -> list:
    """
    Attack Type Classification (NIS)
    This is rule-based to stay stable and demo-proof.
    """
    types = []

    # Credential harvesting
    if (email_feat.get("cred_hits") and len(email_feat.get("cred_hits", [])) > 0) or (url_feat.get("keyword_count", 0) > 0):
        # tighten: only if keywords include login/verify/otp/password/account
        kw = set(url_feat.get("keyword_hits", []))
        if any(x in kw for x in ["login", "verify", "password", "account", "otp"]) or len(email_feat.get("cred_hits", [])) > 0:
            types.append("Credential Harvesting")

    # Brand impersonation
    if intel.get("brand_impersonation") == 1 or (intel.get("brand_hits") and len(intel.get("brand_hits", [])) > 0):
        types.append("Brand Impersonation")

    # Spoofing / header manipulation
    if str(header_risk).lower().startswith("high"):
        types.append("Spoofing / Header Manipulation")

    # Link shortener / redirect abuse
    if url_feat.get("is_shortener") == 1:
        types.append("Link Shortener / Redirect Abuse")

    # Suspicious infrastructure
    if url_feat.get("has_ip_host") == 1:
        types.append("IP-based URL / Suspicious Hosting")

    # Attachment delivery (heuristic: if email text references invoice/attachment keywords)
    if email_feat.get("attach_hits") and len(email_feat.get("attach_hits", [])) > 0:
        types.append("Attachment Delivery (Possible Malware)")

    # If nothing triggered, keep meaningful label
    if not types:
        types.append("Generic Phishing / Social Engineering")

    # de-duplicate while preserving order
    seen = set()
    out = []
    for t in types:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def recommended_actions(level: str, attack_types: list) -> list:
    """
    Defensive recommendations: what to do next.
    """
    level = (level or "").upper()
    attack_types = attack_types or []

    base = [
        "Do not click links or open attachments from suspicious emails.",
        "Verify the request using an official channel (company website/app or known phone number).",
    ]

    if level in ("CRITICAL", "HIGH"):
        base += [
            "Report the email to IT/SOC immediately (phishing incident).",
            "Block the sender domain and URL at email gateway / DNS filtering (if available).",
            "If you already clicked: change password immediately and enable MFA/2FA.",
            "Check for similar emails in inboxes (potential campaign) and warn users.",
        ]

    if "Spoofing / Header Manipulation" in attack_types:
        base += [
            "Review SPF/DKIM/DMARC alignment on the sending domain (spoofing control).",
        ]

    if "Credential Harvesting" in attack_types:
        base += [
            "Reset credentials for impacted accounts and monitor login attempts for anomalies.",
        ]

    if "Attachment Delivery (Possible Malware)" in attack_types:
        base += [
            "Quarantine the attachment and scan in a sandbox / AV before any access.",
        ]

    if "Link Shortener / Redirect Abuse" in attack_types:
        base += [
            "Expand shortened URLs in a safe environment and block redirecting domains if malicious.",
        ]

    # keep it tight for UI
    return base[:10]


# ----------------------------
# Model loading
# ----------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    models = {
        "logistic": None,
        "vectorizer": None,
        "hybrid_meta": None,
        "tokenizer": None,
        "lstm": None,

        # URL pipeline model (host+url FeatureUnion)
        "url_text_model": None,

        "errors": {
            "logistic": None,
            "vectorizer": None,
            "hybrid_meta": None,
            "tokenizer": None,
            "lstm": None,
            "url_text_model": None,
        },
        "config": {
            "max_len": 200
        }
    }

    # ----------------------------
    # Load URL Host+URL Pipeline model
    # ----------------------------
    try:
        if os.path.exists("models/url_tfidf_logistic_model.pkl"):
            # Compatibility: old pickles may reference sel_url/sel_host in __main__.
            try:
                import __main__ as _main
                setattr(_main, "sel_url", sel_url)
                setattr(_main, "sel_host", sel_host)
                setattr(_main, "sel_struct", sel_struct)
            except Exception:
                pass
            models["url_text_model"] = joblib.load("models/url_tfidf_logistic_model.pkl")
        else:
            models["errors"]["url_text_model"] = (
                "models/url_tfidf_logistic_model.pkl not found. Retrain URL model."
            )
    except Exception as e:
        models["errors"]["url_text_model"] = f"URL pipeline load failed ({type(e).__name__}): {e}"

    # ----------------------------
    # TF-IDF vectorizer for EMAIL logistic (try multiple names)
    # ----------------------------
    vec_candidates = [
        "models/phish_tfidf_vectorizer.pkl",
        "models/vectorizer.pkl",
        "models/tfidf_vectorizer.pkl",
        "phish_tfidf_vectorizer.pkl",
        "vectorizer.pkl",
        "tfidf_vectorizer.pkl",
    ]
    vec_loaded = None
    vec_errs = []

    for p in vec_candidates:
        if os.path.exists(p):
            try:
                v = joblib.load(p)
                if _is_tfidf_fitted(v):
                    vec_loaded = v
                    break
                else:
                    vec_errs.append(f"{p}: not fitted")
            except Exception as e:
                vec_errs.append(f"{p}: {e}")

    if vec_loaded is None:
        models["errors"]["vectorizer"] = "No fitted TF-IDF vectorizer found. " + " | ".join(vec_errs)
    else:
        models["vectorizer"] = vec_loaded

    # EMAIL Logistic model
    try:
        models["logistic"] = joblib.load("models/phish_logistic_model.pkl")
    except Exception as e:
        models["errors"]["logistic"] = f"Logistic model load failed: {e}"

    # Hybrid meta model
    try:
        if os.path.exists("models/hybrid_model.pkl"):
            models["hybrid_meta"] = joblib.load("models/hybrid_model.pkl")
        else:
            models["errors"]["hybrid_meta"] = "models/hybrid_model.pkl not found (run train_hybrid_model.py)."
    except Exception as e:
        models["errors"]["hybrid_meta"] = f"Hybrid meta load failed: {e}"

    # Tokenizer: prefer tokenizer.json (portable), fallback to tokenizer.pkl
    tok = None
    if os.path.exists("models/tokenizer.json") and tokenizer_from_json is not None:
        try:
            with open("models/tokenizer.json", "r", encoding="utf-8") as f:
                tok_json = f.read()
            tok = tokenizer_from_json(tok_json)
        except Exception as e:
            models["errors"]["tokenizer"] = f"Tokenizer JSON load failed: {e}"

    if tok is None:
        try:
            tok = joblib.load("models/tokenizer.pkl")
        except Exception as e:
            models["errors"]["tokenizer"] = models["errors"]["tokenizer"] or f"Tokenizer PKL load failed: {e}"

    models["tokenizer"] = tok

    # LSTM model
    try:
        if os.path.exists("models/phish_lstm_model.h5"):
            models["lstm"] = tf.keras.models.load_model(
                "models/phish_lstm_model.h5",
                compile=False,
                custom_objects={
                    "PatchedInputLayer": PatchedInputLayer,
                    "InputLayer": PatchedInputLayer,
                    "DTypePolicy": DTypePolicy,
                },
            )
            meta = _load_json("models/email_meta.json")
            if isinstance(meta, dict) and "max_len" in meta:
                models["config"]["max_len"] = int(meta["max_len"])
        else:
            models["errors"]["lstm"] = "models/phish_lstm_model.h5 not found (run train_deep_model.py)."
    except Exception as e:
        models["errors"]["lstm"] = f"LSTM load failed: {e}"

    return models


models = load_models()


# ----------------------------
# Header analysis
# ----------------------------
def analyze_headers(header_text):
    details = {
        "risk": "Low Risk",
        "confidence": "low",
        "manual_or_incomplete": False,
        "score": 0.0,
        "issues": [],
    }
    if not header_text.strip():
        details["issues"] = ["No headers provided."]
        return details["risk"], details["issues"], details

    txt = str(header_text)
    fields_found = 0
    for fld in ["From:", "To:", "Subject:", "Received:", "Return-Path:", "Reply-To:", "Message-ID:", "Date:"]:
        if fld.lower() in txt.lower():
            fields_found += 1
    details["manual_or_incomplete"] = fields_found < 4

    score = 0.0
    issues = []
    if details["manual_or_incomplete"]:
        score += 0.5
        issues.append("Headers appear partial/manual; spoofing confidence is limited.")

    if "Return-Path:" not in txt and "From:" in txt:
        score += 1.2 if not details["manual_or_incomplete"] else 0.6
        issues.append("Return-Path missing despite sender header.")

    if "Received:" not in txt:
        score += 1.0 if not details["manual_or_incomplete"] else 0.2
        issues.append("Received chain missing or incomplete.")

    from_line = re.search(r"From:\s*(.+)", txt, flags=re.I)
    reply_line = re.search(r"Reply-To:\s*(.+)", txt, flags=re.I)
    if from_line and reply_line:
        from_v = from_line.group(1).strip().lower()
        reply_v = reply_line.group(1).strip().lower()
        if reply_v != from_v:
            score += 3.0
            issues.append("Reply-To differs from From — suspicious redirection.")

    if score >= 3.0:
        details["risk"] = "High Risk"
        details["confidence"] = "high" if not details["manual_or_incomplete"] else "medium"
    elif score >= 1.0:
        details["risk"] = "Medium Risk"
        details["confidence"] = "medium"
    else:
        details["risk"] = "Low Risk"
        details["confidence"] = "low"

    details["score"] = score
    details["issues"] = issues if issues else ["Header seems normal."]
    return details["risk"], details["issues"], details


def calibrate_final_score(raw_score, email_prob, url_prob, header_info, url_feat):
    score = float(raw_score)
    manual = bool(header_info.get("manual_or_incomplete"))
    header_risk = str(header_info.get("risk", "Low Risk")).lower()

    # Reduce inflation from incomplete/manual headers.
    if manual and "high" in header_risk:
        score -= 6.0
    elif manual and "medium" in header_risk:
        score -= 3.0

    # Lift when URL + spoofing both present so these do not fall through too easily.
    if (url_prob or 0.0) >= 0.75 and header_info.get("score", 0.0) >= 2.5:
        score += 6.0

    # Lift for strong infrastructure-style URL indicators even if URL ML is weak.
    infra_hits = 0
    infra_hits += 1 if url_feat.get("has_ip_host") == 1 else 0
    infra_hits += 1 if url_feat.get("is_shortener") == 1 else 0
    infra_hits += 1 if url_feat.get("suspicious_tld") == 1 else 0
    infra_hits += 1 if url_feat.get("keyword_count", 0) >= 2 else 0
    if infra_hits >= 2 and (email_prob or 0.0) >= 0.60:
        score += 4.0

    return max(0.0, min(100.0, score))


# ----------------------------
# Email prediction (Logistic, LSTM, Hybrid meta)
# ----------------------------
def predict_email_models(subject, body):
    raw_text = f"{subject} {body}".strip()
    text = clean_text(raw_text)

    log_prob = None
    lstm_prob = None
    hybrid_prob = None
    email_note = None

    tokenizer = models.get("tokenizer")
    lstm_model = models.get("lstm")
    logistic_model = models.get("logistic")
    vectorizer = models.get("vectorizer")
    hybrid_meta = models.get("hybrid_meta")

    # LSTM prob
    if lstm_model is not None and tokenizer is not None:
        vocab_size = 5000
        try:
            vocab_size = int(lstm_model.layers[0].input_dim)
        except Exception:
            pass

        seq = tokenizer.texts_to_sequences([text])
        seq = [[tok if tok < vocab_size else 1 for tok in s] for s in seq]
        max_len = int(models["config"].get("max_len", 200))
        seq_padded = tf.keras.preprocessing.sequence.pad_sequences(
            seq, maxlen=max_len, padding="post", truncating="post"
        )
        lstm_prob = float(lstm_model.predict(seq_padded, verbose=0)[0][0])

    # Logistic prob
    if logistic_model is not None and vectorizer is not None:
        try:
            X_vec = vectorizer.transform([text])
            log_prob = float(logistic_model.predict_proba(X_vec)[0][1])
        except Exception as e:
            email_note = f"Logistic disabled (vectorizer issue): {e}"
            log_prob = None

    # Hybrid meta (preferred)
    if hybrid_meta is not None and (log_prob is not None) and (lstm_prob is not None):
        try:
            X_meta = np.array([[log_prob, lstm_prob]], dtype=float)
            hybrid_prob = float(hybrid_meta.predict_proba(X_meta)[0][1])
            email_note = "Using trained Hybrid Meta model (logistic + LSTM)."
        except Exception as e:
            hybrid_prob = (log_prob + lstm_prob) / 2.0
            email_note = f"Hybrid meta failed ({e}); using average of Logistic+LSTM."

    # Fallbacks
    if hybrid_prob is None:
        if (log_prob is not None) and (lstm_prob is not None):
            hybrid_prob = (log_prob + lstm_prob) / 2.0
            email_note = email_note or "Using average of Logistic+LSTM (meta model not available)."
        elif lstm_prob is not None:
            hybrid_prob = lstm_prob
            email_note = email_note or "Using LSTM only (Logistic/TF-IDF unavailable)."
        elif log_prob is not None:
            hybrid_prob = log_prob
            email_note = email_note or "Using Logistic only (LSTM unavailable)."
        else:
            hybrid_prob = 0.5
            email_note = email_note or "Email ML unavailable; using neutral score."

    return log_prob, lstm_prob, hybrid_prob, email_note


# ----------------------------
# URL prediction (Pipeline model expects DataFrame with url+host)
# ----------------------------
def predict_url_ml(url: str):
    model = models.get("url_text_model")
    if model is None:
        return 0.5

    url_c = canonicalize_url(url)
    if not url_c:
        return 0.5

    host = extract_host(url_c)
    if not host:
        return 0.5

    X = pd.DataFrame([{"url": url_c, "host": host}])

    try:
        probs = model.predict_proba(X)[0]
        classes = [str(c) for c in getattr(model.named_steps["clf"], "classes_", [])]
        if "1" in classes:
            return float(probs[classes.index("1")])
        return float(probs[1]) if len(probs) > 1 else float(probs[0])
    except Exception:
        return 0.5


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Phishing Detection & Prevention Tool", layout="centered")
st.title("Phishing Detection and Prevention Tool")
st.caption("Detect phishing based on email content, headers, URLs, and remote attachments.")

# ✅ Policy mode (A upgrade)
policy = st.selectbox(
    "Organization Policy Mode",
    [
        "Normal Policy (University / General)",
        "Strict Policy (Banking / Enterprise)",
        "Relaxed Policy (Personal Use)"
    ],
    index=0
)

ths = policy_thresholds(policy)
susp_th = float(ths["suspicious"])
phish_th = float(ths["phishing"])

st.info(f"Policy thresholds → Suspicious ≥ {susp_th:.0f}, Phishing ≥ {phish_th:.0f} (Risk Score out of 100)")

# ✅ NIS toggles
security_mode = st.checkbox("Security Analysis Mode (NIS)", value=True)
show_history = st.checkbox("Show Scan History (last 20)", value=True)

email_ok = (models.get("lstm") is not None and models.get("tokenizer") is not None) or (
    models.get("logistic") is not None and models.get("vectorizer") is not None
)
url_ok = (models.get("url_text_model") is not None)

if email_ok:
    st.success("Email ML pipeline is available (LSTM and/or Logistic).")
else:
    st.warning("Email ML is disabled. Check loaded models in errors below.")

if models.get("hybrid_meta") is not None:
    st.success("Hybrid Meta model loaded (trained combiner).")
else:
    st.info("Hybrid Meta model not loaded - app will fallback to averaging if both models exist.")

if url_ok:
    st.success("URL ML model loaded (Host+URL TF-IDF pipeline).")
else:
    st.warning("URL ML model not loaded. URL scoring will be neutral (0.5).")

if not ATTACHMENT_ANALYZER_AVAILABLE:
    st.info(
        "Attachment scanning module is not available on this setup.\n\n"
        f"Import error: {ATTACHMENT_IMPORT_ERROR}"
    )

with st.expander("Debug: Loaded model status"):
    for k, v in models["errors"].items():
        if v:
            st.write(f"{k}: {v}")
        else:
            st.write(f"{k}: OK")

st.markdown("### Email Content Analysis")
subject = st.text_input("Subject:")
body = st.text_area("Email Body:")
header_text = st.text_area("Email Headers (optional):")
urls_input = st.text_area("URLs Found (optional, one per line):")

if st.button("Analyze Email"):
    # Email side
    if subject.strip() or body.strip():
        log_prob, lstm_prob, hybrid_prob, email_note = predict_email_models(subject, body)
        st.subheader("Email Model Predictions")

        st.write(
            f"**Logistic Regression Probability:** {log_prob:.2f}"
            if log_prob is not None
            else "**Logistic Regression Probability:** (not available)"
        )
        st.write(
            f"**LSTM Model Probability:** {lstm_prob:.2f}"
            if lstm_prob is not None
            else "**LSTM Model Probability:** (not available)"
        )
        st.write(f"**Email Probability Used:** {hybrid_prob:.2f}")
        if email_note:
            st.info(email_note)
    else:
        hybrid_prob = 0.5
        log_prob, lstm_prob = None, None
        st.info("No email content provided - skipping email model prediction.")

    # Header side
    header_risk, header_issues, header_details = analyze_headers(header_text)
    st.subheader("Header Analysis")
    st.write(f"**Header Risk:** {header_risk} *(confidence: {header_details.get('confidence', 'low')})*")
    for issue in header_issues:
        st.write("-", issue)

    # URL side
    url_probs = []
    urls = []
    if urls_input.strip():
        st.subheader("URL Analysis")
        urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
        for u in urls:
            try:
                p = predict_url_ml(u)
                url_probs.append(p)
                st.write(f"{u} → Probability: **{p:.2f}**")
            except Exception as e:
                st.error(f"Error analyzing {u}: {e}")
        url_mean = float(np.mean(url_probs)) if url_probs else 0.5
        st.write(f"**Average URL Risk Probability:** {url_mean:.2f}")
    else:
        url_mean = 0.5

    # ----------------------------
    # NIS Risk Score
    # ----------------------------
    # Choose top URL for explanation (highest URL risk)
    top_url = None
    if urls and url_probs:
        idx = int(np.argmax(url_probs))
        top_url = urls[idx]

    # Extract NIS features (does NOT affect ML)
    url_feat = extract_url_nis_features(top_url) if top_url else {
        "host": "", "tld": "", "url_len": 0, "has_https": 0, "has_ip_host": 0, "has_at": 0,
        "has_double_slash_in_path": 0, "num_subdomain_dots": 0, "keyword_count": 0,
        "keyword_hits": [], "host_entropy": 0.0, "is_shortener": 0, "suspicious_tld": 0,
        "high_digit_ratio": 0, "many_hyphens": 0
    }
    email_feat = extract_email_nis_features(subject, body, header_text)

    # Threat intel (offline = demo-proof)
    intel = offline_intel(top_url) if top_url else {"source": "offline"}

    # Risk score (0..100)
    risk_score_raw = compute_risk_score(
        email_prob=float(hybrid_prob) if hybrid_prob is not None else 0.5,
        url_prob=float(url_mean) if url_mean is not None else 0.5,
        header_risk=header_risk,
        url_feat=url_feat,
        email_feat=email_feat,
        intel=intel
    )
    risk_score = calibrate_final_score(
        raw_score=risk_score_raw,
        email_prob=float(hybrid_prob) if hybrid_prob is not None else 0.5,
        url_prob=float(url_mean) if url_mean is not None else 0.5,
        header_info=header_details,
        url_feat=url_feat,
    )

    # Policy-aligned threat level
    level = policy_threat_level(risk_score, susp_th, phish_th)

    st.markdown("### Final Combined Verdict (NIS Risk Score)")
    st.write(f"**Risk Score:** {risk_score:.1f} / 100")
    st.write(f"**Threat Level:** {level}  *(Policy Mode: {policy})*")

    # Policy-based verdict thresholds
    if risk_score >= phish_th:
        verdict = "Phishing"
        st.error(f"Phishing Detected (Risk: {risk_score:.1f})")
    elif risk_score >= susp_th:
        verdict = "Suspicious"
        st.warning(f"Suspicious Email (Risk: {risk_score:.1f})")
    else:
        verdict = "Safe"
        st.success(f"Safe Email (Risk: {risk_score:.1f})")

    # ----------------------------
    # ✅ A Upgrade #1: Attack Type Classification
    # ----------------------------
    attack_types = classify_attack_types(url_feat, email_feat, intel, header_risk)
    st.markdown("### Attack Type Classification (NIS)")
    st.write("This helps categorize the incident like a real SOC / security tool would:")
    for t in attack_types:
        st.write("•", t)

    # ----------------------------
    # ✅ A Upgrade #2: Recommended Actions (Incident Response)
    # ----------------------------
    st.markdown("### Recommended Actions (Prevention / Response)")
    actions = recommended_actions(level, attack_types)
    for a in actions:
        st.write("•", a)

    # Reasons + details only in NIS mode
    if security_mode:
        reasons = build_reasons(
            email_prob=float(hybrid_prob) if hybrid_prob is not None else None,
            url_prob_mean=float(url_mean) if url_mean is not None else None,
            header_risk=header_risk,
            url_feat=url_feat,
            email_feat=email_feat,
            intel=intel
        )

        st.markdown("### Why this was flagged (NIS Indicators)")
        grouped = grouped_explainability(
            email_prob=float(hybrid_prob) if hybrid_prob is not None else None,
            url_prob_mean=float(url_mean) if url_mean is not None else None,
            header_info=header_details,
            url_feat=url_feat,
            email_feat=email_feat,
            intel=intel,
            risk_score=float(risk_score),
            policy_name=policy,
        )
        for section_name, items in grouped.items():
            st.markdown(f"**{section_name}**")
            if items:
                for i in items[:6]:
                    st.write("-", i)
            else:
                st.write("- No strong signal.")

        with st.expander("Raw compact reasons"):
            for r in reasons:
                st.write("-", r)

        with st.expander("Technical Details (NIS Features)"):
            st.write("**Top URL used for explanation:**", top_url if top_url else "(none)")
            st.json({
                "url_features": url_feat,
                "email_features": email_feat,
                "threat_intel": intel,
                "policy": policy,
                "policy_thresholds": {"suspicious": susp_th, "phishing": phish_th},
                "attack_types": attack_types,
                "header_details": header_details,
                "risk_score_raw": float(risk_score_raw),
                "risk_score_calibrated": float(risk_score),
            })

        # Optional online intel display (won't break without keys)
        if top_url:
            with st.expander("Online Threat Intel (optional)"):
                vt = check_virustotal(top_url)
                st.write("**VirusTotal:**", vt.get("result"))

                # AbuseIPDB only if host is an IP
                if url_feat.get("has_ip_host") == 1:
                    ab = check_abuseipdb(url_feat.get("host", ""))
                    st.write("**AbuseIPDB:**", ab.get("result"))
                else:
                    st.write("**AbuseIPDB:** (host is not an IP)")

    # ✅ Scan logging (safe)
    try:
        append_log({
            "timestamp": now_iso(),
            "risk_score": float(risk_score),
            "threat_level": level,
            "verdict": verdict,
            "email_prob": float(hybrid_prob) if hybrid_prob is not None else 0.5,
            "url_prob_mean": float(url_mean) if url_mean is not None else 0.5,
            "header_risk": str(header_risk),
            "urls": " | ".join(urls) if urls else "",
            "subject_preview": (subject or "")[:80],
        })
    except Exception:
        pass

    # ----------------------------
    # Feedback (kept compatible)
    # ----------------------------
    st.markdown("---")
    st.subheader("Feedback")
    st.write("Was this classification correct?")
    feedback = st.radio("Choose an option:", ["Yes", "No"], horizontal=True, key="feedback_radio")

    if st.button("Submit Feedback"):
        feedback_data = {
            "subject": subject,
            "body": body,
            "urls": urls_input,
            "verdict": verdict,
            "final_score": float(risk_score) / 100.0,
            "email_prob": float(hybrid_prob),
            "url_prob_mean": float(url_mean),
            "user_feedback": feedback,
        }
        feedback_df = pd.DataFrame([feedback_data])

        if os.path.exists("feedback_log.csv"):
            feedback_df.to_csv("feedback_log.csv", mode="a", header=False, index=False)
        else:
            feedback_df.to_csv("feedback_log.csv", index=False)

        st.success("Thank you for your feedback. It has been recorded.")

    # ✅ Scan History Viewer
    if show_history:
        try:
            if os.path.exists("logs/scan_logs.csv"):
                hist = pd.read_csv("logs/scan_logs.csv").tail(20)
                st.markdown("### Scan History (last 20)")
                st.dataframe(hist, use_container_width=True)
            else:
                st.info("No scan history yet.")
        except Exception:
            st.info("Scan history unavailable on this run.")


# ----------------------------
# Remote attachment scan
# ----------------------------
st.markdown("---")
st.markdown("### Remote Attachment URL Scan (Optional)")

remote_url = st.text_input("Enter an attachment URL (e.g., https://example.com/invoice.pdf)")

if st.button("Scan Remote File"):
    if not remote_url.strip():
        st.warning("Please enter a valid URL.")
    elif not ATTACHMENT_ANALYZER_AVAILABLE:
        st.error("Attachment analyzer is unavailable, cannot scan remote file on this setup.")
    else:
        local_path = None
        try:
            result = fetch_remote_attachment(remote_url)
            st.write("**Remote Fetch Status:**", result.get("status", "unknown"))
            st.write("**Message:**", result.get("message", "No details."))
            if result.get("final_url"):
                st.write("**Final URL:**", result.get("final_url"))
            if result.get("content_type"):
                st.write("**Content-Type:**", result.get("content_type"))
            else:
                st.info("Content-Type was missing; proceeding with cautious local file checks.")

            if not result.get("ok"):
                status = result.get("status")
                if status == "blocked":
                    st.warning("Access blocked by remote host (HTTP 403). Try another host/link or manual download.")
                elif status == "not_found":
                    st.warning("File not found on remote host (HTTP 404).")
                elif status in ("webpage", "non_direct_url"):
                    st.info("This looks like a webpage link. Use a direct file download URL.")
                elif status == "timeout":
                    st.warning("Download timed out before completion.")
                elif status == "invalid_url":
                    st.warning("Invalid URL format for remote scan.")
                elif status == "tls_error":
                    st.warning("TLS/certificate validation failed for this URL.")
                elif status == "connection_error":
                    st.warning("Could not connect to remote host.")
                elif status == "too_many_redirects":
                    st.warning("Too many redirects for safe scanning.")
                elif status == "oversized_file":
                    st.warning("Remote file exceeded safe size limit.")
                elif status == "unsupported_content_type":
                    st.warning("Link returned unsupported/unclear file type for attachment scanning.")
                else:
                    st.warning("Remote fetch failed. Please try a direct downloadable file URL.")
                dbg = result.get("debug") or {}
                if dbg:
                    with st.expander("Debug details (remote fetch)"):
                        st.json(dbg)
            else:
                local_path = result.get("local_path")
                st.success(f"File downloaded successfully: {result.get('file_name')} ({result.get('bytes', 0)} bytes)")
                file_risk, file_reasons = analyze_attachment_file(local_path)
                st.write(f"**File Risk Level:** {file_risk}")
                with st.expander("View File Analysis Details"):
                    for reason in file_reasons:
                        st.write("-", reason)

                if "HIGH" in str(file_risk).upper():
                    st.error("File appears malicious.")
                elif "MEDIUM" in str(file_risk).upper():
                    st.warning("File may be suspicious.")
                else:
                    st.success("File appears safe.")

        except Exception as e:
            st.error(f"Could not complete remote file scan safely: {e}")

        finally:
            try:
                if local_path and os.path.exists(local_path):
                    os.remove(local_path)
                    st.info("Temporary file deleted after scan.")
            except Exception:
                st.info("Temporary file cleanup could not be confirmed.")
