# utils/content_analysis.py
import re
from bs4 import BeautifulSoup

def analyze_content(body_text):
    """
    Analyze email body for phishing-like language, HTML manipulation,
    or dangerous attachments.
    Returns: (risk_level, reasons)
    """

    score = 0
    reasons = []

    # ---------- 1️⃣ NLP Keyword Check ----------
    suspicious_phrases = [
        "verify your account", "update your password", "confirm your identity",
        "urgent action required", "login immediately", "suspended", "security alert",
        "click below", "bank account", "reset password", "invoice attached",
        "payment failure", "unauthorized access"
    ]
    for phrase in suspicious_phrases:
        if phrase in body_text.lower():
            score += 1
            reasons.append(f"Suspicious phrase detected: '{phrase}'")

    # ---------- 2️⃣ HTML Link Mismatch ----------
    if "<a" in body_text.lower():
        soup = BeautifulSoup(body_text, "html.parser")
        for link in soup.find_all("a", href=True):
            visible_text = link.get_text().strip()
            actual_link = link["href"]
            if visible_text and visible_text not in actual_link and not actual_link.startswith("#"):
                score += 1
                reasons.append(f"Link mismatch: text '{visible_text}' → URL '{actual_link}'")

    # ---------- 3️⃣ Attachment Check ----------
    attachment_patterns = [r"\.exe", r"\.zip", r"\.rar", r"\.js", r"\.bat", r"\.scr", r"\.vbs", r"\.docm", r"\.xlsm"]
    for pattern in attachment_patterns:
        if re.search(pattern, body_text.lower()):
            score += 1
            reasons.append(f"Potentially dangerous attachment type found: '{pattern}'")

    # ---------- 4️⃣ Encoding Tricks ----------
    if "xn--" in body_text or re.search(r"%[0-9a-f]{2}", body_text.lower()):
        score += 1
        reasons.append("Encoding or Punycode trick detected")

    # ---------- 5️⃣ Final Risk Level ----------
    if score >= 3:
        risk = "🚨 HIGH RISK"
    elif score == 2:
        risk = "⚠️ MEDIUM RISK"
    else:
        risk = "✅ LOW RISK"

    return risk, reasons
