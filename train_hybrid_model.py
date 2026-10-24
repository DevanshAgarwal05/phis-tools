# train_hybrid_model.py (UPDATED)
import os
import re
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

RANDOM_STATE = 42
MAX_LEN = 200

MODEL_DIR = "models"
TFIDF_PATH = os.path.join(MODEL_DIR, "phish_tfidf_vectorizer.pkl")
LOG_MODEL_PATH = os.path.join(MODEL_DIR, "phish_logistic_model.pkl")
TOKENIZER_JSON_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "phish_lstm_model.h5")
HYBRID_META_PATH = os.path.join(MODEL_DIR, "hybrid_model.pkl")

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

def load_email_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]

    label_col = None
    for c in ["label", "class", "target", "phishing"]:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError("No label column found.")

    subj = "subject" if "subject" in df.columns else None
    body = "body" if "body" in df.columns else None
    urls = "urls" if "urls" in df.columns else None
    text_col = "text" if "text" in df.columns else None

    if text_col:
        df["text"] = df["text"].fillna("").astype(str)
    else:
        if subj is None and body is None:
            raise ValueError("Need either 'text' OR 'subject'/'body'.")
        df["text"] = ""
        if subj: df["text"] += df[subj].fillna("").astype(str) + " "
        if body: df["text"] += df[body].fillna("").astype(str) + " "
        if urls: df["text"] += df[urls].fillna("").astype(str) + " "

    df["text"] = df["text"].apply(clean_text)
    df["label"] = df[label_col].apply(normalize_label)
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]]

def load_tokenizer_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        tok_json = f.read()
    return tokenizer_from_json(tok_json)

def main(data_path: str):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Require the already-trained base models
    if not (os.path.exists(TFIDF_PATH) and os.path.exists(LOG_MODEL_PATH) and os.path.exists(TOKENIZER_JSON_PATH) and os.path.exists(LSTM_MODEL_PATH)):
        raise FileNotFoundError(
            "Missing base models. Train them first (recommended: run train_all.py). "
            f"Expected:\n- {TFIDF_PATH}\n- {LOG_MODEL_PATH}\n- {TOKENIZER_JSON_PATH}\n- {LSTM_MODEL_PATH}"
        )

    df = load_email_dataset(data_path)
    texts = df["text"].values
    labels = df["label"].values

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )

    vectorizer = joblib.load(TFIDF_PATH)
    log_model = joblib.load(LOG_MODEL_PATH)
    lstm_model = load_model(LSTM_MODEL_PATH)
    tokenizer = load_tokenizer_from_json(TOKENIZER_JSON_PATH)

    # Probabilities from Logistic
    X_train_tfidf = vectorizer.transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    log_train = log_model.predict_proba(X_train_tfidf)[:, 1]
    log_test = log_model.predict_proba(X_test_tfidf)[:, 1]

    # Probabilities from LSTM (using SAME tokenizer)
    seq_train = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_LEN, padding="post", truncating="post")
    seq_test = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_LEN, padding="post", truncating="post")
    lstm_train = lstm_model.predict(seq_train, verbose=0).ravel()
    lstm_test = lstm_model.predict(seq_test, verbose=0).ravel()

    # Meta model: logistic regression over [log_prob, lstm_prob]
    X_meta_train = np.vstack([log_train, lstm_train]).T
    X_meta_test = np.vstack([log_test, lstm_test]).T

    meta = LogisticRegression(max_iter=1000)
    meta.fit(X_meta_train, y_train)

    pred = meta.predict(X_meta_test)
    acc = accuracy_score(y_test, pred)

    print(f"\n✅ Hybrid Meta Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, pred))

    joblib.dump(meta, HYBRID_META_PATH)
    print("💾 Saved:", HYBRID_META_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to email dataset (csv/xlsx).")
    args = parser.parse_args()
    main(args.data)
