# utils/header_analysis.py
import re

def analyze_headers(headers_text: str):
    """
    Analyze raw email headers to detect phishing indicators like
    mismatched From/Reply-To domains, failed SPF/DKIM, or suspicious TLDs.
    Returns a tuple: (risk_level, list_of_reasons)
    """

    score = 0
    reasons = []

    # --- Check for mismatched From and Reply-To domains ---
    from_match = re.search(r"From:\s*(.*)", headers_text, re.IGNORECASE)
    reply_match = re.search(r"Reply-To:\s*(.*)", headers_text, re.IGNORECASE)
    if from_match and reply_match:
        from_addr = from_match.group(1).strip()
        reply_addr = reply_match.group(1).strip()
        if '@' in from_addr and '@' in reply_addr:
            from_domain = from_addr.split('@')[-1].split('>')[0]
            reply_domain = reply_addr.split('@')[-1].split('>')[0]
            if from_domain.lower() != reply_domain.lower():
                score += 1
                reasons.append(f"From and Reply-To domains differ ({from_domain} ≠ {reply_domain})")

    # --- Check SPF/DKIM/DMARC authentication results ---
    if re.search(r"spf=fail", headers_text, re.IGNORECASE):
        score += 1
        reasons.append("SPF authentication failed")
    if re.search(r"dkim=fail", headers_text, re.IGNORECASE):
        score += 1
        reasons.append("DKIM authentication failed")
    if re.search(r"dmarc=fail", headers_text, re.IGNORECASE):
        score += 1
        reasons.append("DMARC authentication failed")

    # --- Check for suspicious sender domains ---
    if re.search(r"@.*\.(ru|xyz|top|click|work|tk|ml|ga)", headers_text, re.IGNORECASE):
        score += 1
        reasons.append("Suspicious sender domain extension detected")

    # --- Check Received headers for anomalies ---
    received_lines = re.findall(r"Received:.*", headers_text, re.IGNORECASE)
    if len(received_lines) < 2:
        score += 1
        reasons.append("Abnormally few Received headers (possible direct spoof)")

    # --- Assign risk level based on score ---
    if score >= 3:
        risk_level = "🚨 HIGH RISK"
    elif score == 2:
        risk_level = "⚠️ MEDIUM RISK"
    else:
        risk_level = "✅ LOW RISK"

    return risk_level, reasons
