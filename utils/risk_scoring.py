# utils/risk_scoring.py
def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))

def threat_level(score: float) -> str:
    if score >= 85:
        return "CRITICAL"
    if score >= 70:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    return "LOW"

def build_reasons(email_prob, url_prob_mean, header_risk, url_feat, email_feat, intel) -> list:
    reasons = []

    # ML based
    if email_prob is not None and email_prob >= 0.75:
        reasons.append(f"Email ML probability is high ({email_prob:.2f}).")
    if url_prob_mean is not None and url_prob_mean >= 0.75:
        reasons.append(f"URL ML probability is high ({url_prob_mean:.2f}).")

    # header
    if str(header_risk).lower().startswith("high"):
        reasons.append("Header analysis indicates spoofing risk.")

    # URL heuristics
    if url_feat.get("has_ip_host") == 1:
        reasons.append("URL host is an IP address (common phishing indicator).")
    if url_feat.get("has_at") == 1:
        reasons.append("URL contains '@' which can hide the real destination.")
    if url_feat.get("is_shortener") == 1:
        reasons.append("URL is a known link shortener (can hide destination).")
    if url_feat.get("suspicious_tld") == 1:
        reasons.append(f"Suspicious TLD detected (.{url_feat.get('tld','')}).")
    if url_feat.get("host_entropy", 0) >= 3.8:
        reasons.append("High domain randomness/entropy detected (possible generated domain).")
    if url_feat.get("keyword_count", 0) >= 2:
        reasons.append(f"Multiple suspicious keywords found in URL: {', '.join(url_feat.get('keyword_hits', []))}.")
    if intel.get("brand_impersonation") == 1:
        bh = ", ".join(intel.get("brand_hits", []))
        reasons.append(f"Possible brand impersonation detected ({bh}).")

    # Email heuristics
    if email_feat.get("reply_to_differs") == 1:
        reasons.append("Reply-To differs from From (redirection tactic).")
    if len(email_feat.get("urgency_hits", [])) > 0:
        reasons.append(f"Urgency language detected: {', '.join(email_feat.get('urgency_hits', []))}.")
    if len(email_feat.get("cred_hits", [])) > 0:
        reasons.append(f"Credential-harvesting cues detected: {', '.join(email_feat.get('cred_hits', []))}.")

    # keep it tight
    return reasons[:8] if reasons else ["No strong indicators beyond model output."]

def compute_risk_score(email_prob: float, url_prob: float, header_risk: str, url_feat: dict, email_feat: dict, intel: dict) -> float:
    # Start from your existing combined ML signal
    base = ((email_prob or 0.5) * 60.0) + ((url_prob or 0.5) * 40.0)  # 0..100

    # Add NIS heuristic bumps (small, so we don't break your existing behavior)
    bump = 0.0
    if str(header_risk).lower().startswith("high"):
        bump += 8.0
    if url_feat.get("has_ip_host") == 1:
        bump += 10.0
    if url_feat.get("has_at") == 1:
        bump += 8.0
    if url_feat.get("is_shortener") == 1:
        bump += 6.0
    if url_feat.get("suspicious_tld") == 1:
        bump += 6.0
    if url_feat.get("host_entropy", 0) >= 3.8:
        bump += 5.0
    if url_feat.get("keyword_count", 0) >= 2:
        bump += 4.0
    if intel.get("brand_impersonation") == 1:
        bump += 7.0
    if email_feat.get("reply_to_differs") == 1:
        bump += 6.0
    if len(email_feat.get("urgency_hits", [])) > 0:
        bump += 4.0
    if len(email_feat.get("cred_hits", [])) > 0:
        bump += 4.0

    return clamp(base + bump, 0.0, 100.0)
