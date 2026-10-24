import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzers.email.header_analysis import analyze_headers  # noqa: E402


def test_empty_headers():
    risk, issues, d = analyze_headers("  ")
    assert (risk, issues) == ("Low Risk", ["No headers provided."]) and d["score"] == 0.0


def test_reply_to_mismatch_is_high_risk():
    risk, issues, d = analyze_headers("From: a@x.com\nReply-To: b@y.com\nReturn-Path: c@z.com")
    assert risk == "High Risk" and d["score"] == 4.0 and d["confidence"] == "high"
    assert "Reply-To differs from From — suspicious redirection." in issues


def test_matching_headers_missing_received_is_medium_risk():
    risk, issues, d = analyze_headers("From: a@x.com\nReply-To: a@x.com\nReturn-Path: a@x.com")
    assert risk == "Medium Risk" and d["score"] == 1.0 and d["manual_or_incomplete"] is False
