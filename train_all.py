# train_all.py
import os
import re
import json
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import tokenizer_from_json, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TFIDF_PATH = os.path.join(MODEL_DIR, "phish_tfidf_vectorizer.pkl")
LOG_MODEL_PATH = os.path.join(MODEL_DIR, "phish_logistic_model.pkl")
TOKENIZER_JSON_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
TOKENIZER_PKL_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")  # optional legacy
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "phish_lstm_model.h5")
HYBRID_META_PATH = os.path.join(MODEL_DIR, "hybrid_model.pkl")
EMAIL_META_PATH = os.path.join(MODEL_DIR, "email_meta.json")

RANDOM_STATE = 42
VOCAB_SIZE = 5000
MAX_LEN = 200
EMBED_DIM = 64


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = [c.strip().lower() for c in df.columns]

    # Detect label column
    label_col = None
    for c in df.columns:
        if c in ["label", "class", "target", "phishing"]:
            label_col = c
            break
    if label_col is None:
        raise ValueError("No label column found. Expected one of: label/class/target/phishing")

    # Detect text columns
    subj = "subject" if "subject" in df.columns else None
    body = "body" if "body" in df.columns else None
    urls = "urls" if "urls" in df.columns else None
    text_col = "text" if "text" in df.columns else None

    if text_col:
        df["text"] = df[text_col].fillna("").astype(str)
    else:
        if subj is None and body is None:
            raise ValueError("Dataset needs either 'text' column OR 'subject'/'body' columns.")
        df["text"] = ""
        if subj:
            df["text"] += df[subj].fillna("").astype(str) + " "
        if body:
            df["text"] += df[body].fillna("").astype(str) + " "
        if urls:
            df["text"] += df[urls].fillna("").astype(str) + " "

    # Map labels to 0/1 robustly
    y = df[label_col].astype(str).str.strip().str.lower()
    y = y.map({
        "1": 1, "phishing": 1, "spam": 1, "malicious": 1, "bad": 1,
        "0": 0, "legitimate": 0, "benign": 0, "good": 0, "safe": 0
    })
    df["label"] = y
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["text"] = df["text"].apply(clean_text)
    df = df[["text", "label"]].dropna()

    print("✅ Loaded:", df.shape)
    print(df["label"].value_counts())
    return df


def train_logistic(df: pd.DataFrame):
    X = df["text"].values
    y = df["label"].values

    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_res, y_res = ros.fit_resample(X.reshape(-1, 1), y)
    X_res = X_res.ravel()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )
    X_vec = vectorizer.fit_transform(X_res)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y_res, test_size=0.2, random_state=RANDOM_STATE, stratify=y_res
    )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("\n=== Logistic (TF-IDF) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    try:
        print("ROC AUC:", roc_auc_score(y_test, y_prob))
    except Exception:
        pass

    joblib.dump(vectorizer, TFIDF_PATH)
    joblib.dump(clf, LOG_MODEL_PATH)
    print("💾 Saved:", TFIDF_PATH)
    print("💾 Saved:", LOG_MODEL_PATH)
    return clf, vectorizer


def train_lstm(df: pd.DataFrame):
    texts = df["text"].values
    labels = df["label"].values

    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_res, y_res = ros.fit_resample(texts.reshape(-1, 1), labels)
    texts_res = X_res.ravel()
    labels_res = y_res

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts_res)

    seqs = tokenizer.texts_to_sequences(texts_res)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    X_train, X_test, y_train, y_test = train_test_split(
        padded, labels_res, test_size=0.2, random_state=RANDOM_STATE, stratify=labels_res
    )

    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=False)),
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

    print("\n=== LSTM ===")
    model.fit(
        X_train, y_train,
        epochs=8,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    preds = model.predict(X_test, verbose=0).ravel()
    preds_label = (preds >= 0.5).astype(int)

    print("Accuracy:", accuracy_score(y_test, preds_label))
    print(classification_report(y_test, preds_label))
    try:
        print("ROC AUC:", roc_auc_score(y_test, preds))
    except Exception:
        pass

    # Save tokenizer JSON (portable)
    with open(TOKENIZER_JSON_PATH, "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())

    # Optional legacy pickle (can break on other PCs, but keeps your current setup)
    try:
        joblib.dump(tokenizer, TOKENIZER_PKL_PATH)
    except Exception:
        pass

    model.save(LSTM_MODEL_PATH)

    with open(EMAIL_META_PATH, "w", encoding="utf-8") as f:
        json.dump({"vocab_size": VOCAB_SIZE, "max_len": MAX_LEN, "embed_dim": EMBED_DIM}, f, indent=2)

    print("💾 Saved:", TOKENIZER_JSON_PATH)
    print("💾 Saved:", LSTM_MODEL_PATH)
    print("💾 Saved:", EMAIL_META_PATH)
    return model, tokenizer


def train_hybrid_meta(df: pd.DataFrame, vectorizer, logistic_model, lstm_model, tokenizer):
    texts = df["text"].values
    labels = df["label"].values

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )

    X_train_tfidf = vectorizer.transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    log_train = logistic_model.predict_proba(X_train_tfidf)[:, 1]
    log_test = logistic_model.predict_proba(X_test_tfidf)[:, 1]

    seq_train = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_LEN, padding="post", truncating="post")
    seq_test = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_LEN, padding="post", truncating="post")

    lstm_train = lstm_model.predict(seq_train, verbose=0).ravel()
    lstm_test = lstm_model.predict(seq_test, verbose=0).ravel()

    X_meta_train = np.vstack([log_train, lstm_train]).T
    X_meta_test = np.vstack([log_test, lstm_test]).T

    meta = LogisticRegression(max_iter=1000)
    meta.fit(X_meta_train, y_train)

    pred = meta.predict(X_meta_test)
    print("\n=== Hybrid Meta (Logistic on [log_prob, lstm_prob]) ===")
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))

    joblib.dump(meta, HYBRID_META_PATH)
    print("💾 Saved:", HYBRID_META_PATH)
    return meta


def main(data_path: str):
    df = load_dataset(data_path)

    log_model, vectorizer = train_logistic(df)
    lstm_model, tokenizer = train_lstm(df)
    _ = train_hybrid_meta(df, vectorizer, log_model, lstm_model, tokenizer)

    print("\n✅ Done. All email models saved under:", MODEL_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to email dataset (csv/xlsx). Needs label + subject/body or text.")
    args = parser.parse_args()
    main(args.data)
