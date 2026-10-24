# utils/scan_logger.py
import os
import csv
from datetime import datetime

from config.settings import SCAN_LOG_PATH

LOG_PATH_DEFAULT = str(SCAN_LOG_PATH)

def ensure_log_file(path: str = LOG_PATH_DEFAULT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp",
                "risk_score",
                "threat_level",
                "verdict",
                "email_prob",
                "url_prob_mean",
                "header_risk",
                "urls",
                "subject_preview"
            ])

def append_log(row: dict, path: str = LOG_PATH_DEFAULT):
    ensure_log_file(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            row.get("timestamp"),
            row.get("risk_score"),
            row.get("threat_level"),
            row.get("verdict"),
            row.get("email_prob"),
            row.get("url_prob_mean"),
            row.get("header_risk"),
            row.get("urls"),
            row.get("subject_preview"),
        ])

def now_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
