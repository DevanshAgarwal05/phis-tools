# analyzers/email/header_analysis.py
import re


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
