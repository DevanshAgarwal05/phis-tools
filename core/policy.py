# core/policy.py


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
