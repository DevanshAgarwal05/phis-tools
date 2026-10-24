# train_logistic_model.py (UPDATED)
import os
import re
import json
import argparse
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler

OUT_DIR = "models"
MODEL_PATH = os.path.join(OUT_DIR, "phish_logistic_model.pkl")
VEC_PATH = os.path.join(OUT_DIR, "phish_tfidf_vectorizer.pkl")
META_PATH = os.path.join(OUT_DIR, "email_logistic_meta.json")

MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
RANDOM_STATE = 42

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_label(x):
    s = str(x).strip().lower()
    if s in ["1", "phishing", "spam", "malicious", "true", "yes", "bad"]:
        return 1
    if s in ["0", "ham", "legit", "legitimate", "benign", "false", "no", "good", "safe"]:
        return 0
    try:
        v = int(float(s))
        return 1 if v == 1 else 0
    except Exception:
        return None

def read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)

def detect_email_columns(df: pd.DataFrame):
    cols = {c.strip().lower(): c for c in df.columns}

    label_col = None
    for c in ["label", "class", "target", "phishing"]:
        if c in cols:
            label_col = cols[c]
            break
    if label_col is None:
        raise ValueError(f"No label column found. Columns: {df.columns.tolist()}")

    text_col = cols.get("text")
    subject_col = cols.get("subject")
    body_col = cols.get("body")
    urls_col = cols.get("urls")

    if text_col is None and subject_col is None and body_col is None:
        raise ValueError("Need either a 'text' column OR 'subject'/'body' columns in the email dataset.")

    return label_col, text_col, subject_col, body_col, urls_col

def load_email_dataset(path: str) -> pd.DataFrame:
    df = read_any(path)
    label_col, text_col, subject_col, body_col, urls_col = detect_email_columns(df)

    if text_col:
        df["text"] = df[text_col].fillna("").astype(str)
    else:
        df["text"] = ""
        if subject_col:
            df["text"] += df[subject_col].fillna("").astype(str) + " "
        if body_col:
            df["text"] += df[body_col].fillna("").astype(str) + " "
        if urls_col:
            df["text"] += df[urls_col].fillna("").astype(str) + " "

    df["text"] = df["text"].apply(clean_text)
    df["label"] = df[label_col].apply(normalize_label)

    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(int)

    print("✅ Loaded email dataset:", df.shape)
    print("Label distribution:\n", df["label"].value_counts())
    return df[["text", "label"]]

def main(data_path: str):
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_email_dataset(data_path)
    X_text = df["text"].values
    y = df["label"].values

    # Oversample for stable demo metrics
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_res, y_res = ros.fit_resample(X_text.reshape(-1, 1), y)
    X_res = X_res.ravel()

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words="english",
        min_df=2
    )
    X_vec = vectorizer.fit_transform(X_res)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y_res, test_size=0.25, random_state=RANDOM_STATE, stratify=y_res
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Logistic Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred))
    try:
        print("✅ ROC-AUC:", roc_auc_score(y_test, y_prob))
    except Exception:
        pass

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)

    meta = {
        "max_features": MAX_FEATURES,
        "ngram_range": list(NGRAM_RANGE),
        "random_state": RANDOM_STATE,
        "trained_on": os.path.basename(data_path)
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n💾 Saved:")
    print(" -", MODEL_PATH)
    print(" -", VEC_PATH)
    print(" -", META_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Email dataset path (csv/xlsx).")
    args = parser.parse_args()
    main(args.data)
