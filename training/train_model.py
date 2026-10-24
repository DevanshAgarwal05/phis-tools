# train_model.py (UPDATED) - toy URL classifier, standalone + demo-friendly
import os
import re
import json
import urllib.parse
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression

OUT_DIR = "models"
MODEL_PATH = os.path.join(OUT_DIR, "toy_url_logistic.pkl")
FEAT_PATH = os.path.join(OUT_DIR, "toy_url_features.json")

def extract_url_features(url: str):
    url = "" if url is None else str(url)
    try:
        parsed = urllib.parse.urlparse(url)
        domain = (parsed.netloc or "").lower()
    except Exception:
        domain = ""

    has_ip = int(bool(re.match(r"\d+\.\d+\.\d+\.\d+$", domain)))

    return {
        "url_length": len(url),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special": len(re.findall(r"[^A-Za-z0-9]", url)),
        "num_subdomains": domain.count("."),
        "has_https": int(url.startswith("https")),
        "has_ip": has_ip,
        "contains_at": int("@" in url),
        "contains_login": int("login" in url.lower()),
        "contains_verify": int("verify" in url.lower()),
    }

def generate_toy_dataset():
    urls = [
        ("http://192.168.1.10/login", 1),
        ("https://accounts.google.com/ServiceLogin", 0),
        ("http://xn--ph1sh-example.com/secure", 1),
        ("https://bank.example.com/verify", 1),
        ("https://example.com/home", 0),
        ("http://example.co.uk", 0),
        ("http://very-long-subdomain.a.b.c.d.e.f.g.h.example.com/path?x=1", 1),
        ("https://login.microsoftonline.com/common/oauth2/v2.0/authorize", 0),
        ("http://example.com@malicious.com/login", 1),
    ]
    rows = []
    for u, label in urls:
        f = extract_url_features(u)
        f["label"] = label
        rows.append(f)
    return pd.DataFrame(rows)

def train_and_save():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = generate_toy_dataset()
    X = df.drop(columns=["label"])
    y = df["label"].values

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)

    with open(FEAT_PATH, "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    print("✅ Toy model trained.")
    print("💾 Saved:", MODEL_PATH)
    print("💾 Saved:", FEAT_PATH)

if __name__ == "__main__":
    train_and_save()
