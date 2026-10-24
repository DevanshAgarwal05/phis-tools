# train_url_visual.py (UPDATED)
import os
import re
import json
import argparse
import urllib.parse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

RANDOM_STATE = 42

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def detect_columns(df):
    cols = {c.strip().lower(): c for c in df.columns}

    url_col = None
    for k in ["url", "urls", "link", "uri"]:
        if k in cols:
            url_col = cols[k]
            break
    if url_col is None and "URL" in df.columns:
        url_col = "URL"

    label_col = None
    for k in ["label", "type", "category", "class"]:
        if k in cols:
            label_col = cols[k]
            break
    if label_col is None and "Label" in df.columns:
        label_col = "Label"

    if url_col is None or label_col is None:
        raise ValueError(f"Could not detect URL/label columns. Found: {df.columns.tolist()}")
    return url_col, label_col

def normalize_label(x):
    s = str(x).strip().lower()
    if s in ["1", "phishing", "malware", "defacement", "malicious", "bad"]:
        return 1
    if s in ["0", "benign", "legitimate", "good", "safe", "normal"]:
        return 0
    return None

def extract_url_features(url: str):
    url = "" if url is None else str(url)
    try:
        parsed = urllib.parse.urlparse(url)
        domain = (parsed.netloc or "").lower()
    except Exception:
        domain = ""

    has_ip = int(bool(re.match(r"\d+\.\d+\.\d+\.\d+$", domain)))
    suspicious_tld = int(re.search(r"\.(ru|cn|xyz|top|tk|ml|ga|cf|gq|zip|click)$", domain) is not None)

    return {
        "url_length": len(url),
        "num_digits": sum(c.isdigit() for c in url),
        "num_letters": sum(c.isalpha() for c in url),
        "num_special": len(re.findall(r"[^A-Za-z0-9]", url)),
        "num_subdomains": domain.count("."),
        "has_https": int(url.startswith("https")),
        "has_ip": has_ip,
        "contains_at": int("@" in url),
        "contains_dash": int("-" in domain),
        "contains_equal": int("=" in url),
        "contains_question": int("?" in url),
        "contains_login": int("login" in url.lower()),
        "contains_verify": int("verify" in url.lower()),
        "contains_secure": int("secure" in url.lower()),
        "suspicious_tld": suspicious_tld
    }

def main(data_path: str, out_dir: str, graph_dir: str):
    ensure_dir(out_dir)
    ensure_dir(graph_dir)

    model_out = os.path.join(out_dir, "url_ml_model.pkl")
    feat_out = os.path.join(out_dir, "url_features.json")

    df = pd.read_csv(data_path) if data_path.lower().endswith(".csv") else pd.read_excel(data_path)
    url_col, label_col = detect_columns(df)

    df = df[[url_col, label_col]].dropna().copy()
    df["label"] = df[label_col].apply(normalize_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["url"] = df[url_col].astype(str)

    feats = df["url"].apply(extract_url_features)
    X = pd.DataFrame(feats.tolist())
    y = df["label"].values

    if len(np.unique(y)) < 2:
        raise ValueError("URL dataset has only one class after normalization. Cannot train a classifier.")

    with open(feat_out, "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    # Distribution plot
    fig, ax = plt.subplots(figsize=(5, 3))
    pd.Series(y).value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("URL Class Distribution (0=Benign,1=Malicious)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    fig.savefig(os.path.join(graph_dir, "url_class_distribution.png"), bbox_inches="tight")
    plt.close(fig)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    base = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2
    )
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n✅ URL Visual Model Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["Benign", "Malicious"],
                yticklabels=["Benign", "Malicious"],
                ax=ax)
    ax.set_title("Confusion Matrix - URL Model")
    fig.savefig(os.path.join(graph_dir, "url_confusion_matrix.png"), bbox_inches="tight")
    plt.close(fig)

    joblib.dump(model, model_out)
    print("\n💾 Saved:")
    print(" -", model_out)
    print(" -", feat_out)
    print("📊 Graphs saved in:", graph_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Merged URL dataset path (csv/xlsx).")
    parser.add_argument("--out_dir", default="models", help="Output directory for models.")
    parser.add_argument("--graph_dir", default="graphs", help="Output directory for graphs.")
    args = parser.parse_args()
    main(args.data, args.out_dir, args.graph_dir)
