# train_url_model.py (UPDATED) - toy URL trainer, consistent artifacts
import os
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import re
import urllib.parse

RANDOM_STATE = 42

def extract_features(url: str):
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
        "contains_secure": int("secure" in url.lower()),
    }

def build_toy_dataset():
    examples = [
        ("http://192.168.1.10/login", 1),
        ("http://example.com", 0),
        ("https://accounts.google.com/ServiceLogin", 0),
        ("http://xn--ph1sh-example.com/secure", 1),
        ("https://bank.example.com/verify", 1),
        ("https://github.com/settings/security", 0),
        ("http://very-long-subdomain.a.b.c.d.e.f.example.com/login?user=1", 1),
        ("http://short.co", 0),
        ("https://login.microsoftonline.com/common/oauth2/v2.0/authorize", 0),
        ("http://example.com@malicious.com/login", 1),
        ("https://secure-paypal.com.example.com/verify", 1),
        ("https://shop.amazon.com/product/123", 0),
    ]

    rows = []
    for url, label in examples:
        feats = extract_features(url)
        feats["label"] = label
        rows.append(feats)
    return pd.DataFrame(rows)

def train_and_evaluate(model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "toy_url_model.pkl")
    feat_path = os.path.join(model_dir, "toy_url_model_features.json")

    df = build_toy_dataset()
    X = df.drop(columns=["label"])
    y = df["label"].values

    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n✅ Toy URL Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(model, model_path)
    print("\n💾 Saved:", model_path)
    print("💾 Saved:", feat_path)

if __name__ == "__main__":
    train_and_evaluate()
