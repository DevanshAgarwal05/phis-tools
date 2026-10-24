# train_deep_model.py (UPDATED)
import os
import re
import json
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import RandomOverSampler

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "phish_lstm_model.h5")
TOK_PKL_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
TOK_JSON_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
META_PATH = os.path.join(MODEL_DIR, "email_meta.json")

MAX_WORDS = 5000
MAX_LEN = 200
EMBED_DIM = 64
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
    if s in ["0", "ham", "benign", "legit", "legitimate", "false", "no", "good", "safe"]:
        return 0
    try:
        v = int(float(s))
        return 1 if v == 1 else 0
    except Exception:
        return None

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]

    label_col = None
    for c in ["label", "class", "target", "phishing"]:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"No label column found. Columns: {df.columns.tolist()}")

    subj = "subject" if "subject" in df.columns else None
    body = "body" if "body" in df.columns else None
    urls = "urls" if "urls" in df.columns else None
    text_col = "text" if "text" in df.columns else None

    if text_col:
        df["text"] = df["text"].fillna("").astype(str)
    else:
        if subj is None and body is None:
            raise ValueError("Need either 'text' OR 'subject'/'body' columns.")
        df["text"] = ""
        if subj: df["text"] += df[subj].fillna("").astype(str) + " "
        if body: df["text"] += df[body].fillna("").astype(str) + " "
        if urls: df["text"] += df[urls].fillna("").astype(str) + " "

    df["text"] = df["text"].apply(clean_text)
    df["label"] = df[label_col].apply(normalize_label)
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(int)

    print("✅ Loaded:", df.shape)
    print(df["label"].value_counts())
    return df[["text", "label"]]

def main(data_path: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_dataset(data_path)

    texts = df["text"].values
    labels = df["label"].values

    # Balance for better demo stability
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_res, y_res = ros.fit_resample(texts.reshape(-1, 1), labels)
    texts_res = X_res.ravel()
    labels_res = y_res

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts_res)

    X_seq = tokenizer.texts_to_sequences(texts_res)
    X_pad = pad_sequences(X_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    X_train, X_test, y_train, y_test = train_test_split(
        X_pad, labels_res, test_size=0.2, random_state=RANDOM_STATE, stratify=labels_res
    )

    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64)),
        Dropout(0.35),
        Dense(32, activation="relu"),
        Dropout(0.25),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5)
    ]

    model.fit(
        X_train, y_train,
        epochs=8,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    print("✅ LSTM Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    try:
        print("ROC AUC:", roc_auc_score(y_test, y_prob))
    except Exception:
        pass

    model.save(MODEL_PATH)
    joblib.dump(tokenizer, TOK_PKL_PATH)
    with open(TOK_JSON_PATH, "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"max_words": MAX_WORDS, "max_len": MAX_LEN, "embed_dim": EMBED_DIM}, f, indent=2)

    print("💾 Saved:", MODEL_PATH)
    print("💾 Saved:", TOK_PKL_PATH)
    print("💾 Saved:", TOK_JSON_PATH)
    print("💾 Saved:", META_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to email dataset (csv/xlsx).")
    args = parser.parse_args()
    main(args.data)
