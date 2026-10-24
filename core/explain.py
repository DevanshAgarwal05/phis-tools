# core/explain.py


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
