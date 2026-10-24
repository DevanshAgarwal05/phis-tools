# train_all_visual.py (UPDATED)
import os
import re
import json
import urllib.parse
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import RandomOverSampler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ========================
# HELPERS
# ========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_graph(fig, out_path):
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved graph → {out_path}")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_email_label(x):
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

def detect_email_columns(df):
    cols = {c.strip().lower(): c for c in df.columns}
    label = None
    for c in ["label", "class", "target", "phishing"]:
        if c in cols:
            label = cols[c]; break

    subject = cols.get("subject")
    body = cols.get("body")
    urls = cols.get("urls")
    text = cols.get("text")

    if label is None:
        raise ValueError(f"Email dataset missing label column. Found: {df.columns.tolist()}")
    if not (text or subject or body):
        raise ValueError("Email dataset needs either 'text' OR 'subject'/'body' columns.")
    return label, subject, body, urls, text

def load_email_dataset(path):
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    label_col, subj_col, body_col, urls_col, text_col = detect_email_columns(df)

    if text_col:
        df["text"] = df[text_col].fillna("").astype(str)
    else:
        df["text"] = ""
        if subj_col: df["text"] += df[subj_col].fillna("").astype(str) + " "
        if body_col: df["text"] += df[body_col].fillna("").astype(str) + " "
        if urls_col: df["text"] += df[urls_col].fillna("").astype(str) + " "

    df["text"] = df["text"].apply(clean_text)
    df["label"] = df[label_col].apply(normalize_email_label)
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(int)

    print("✅ Email dataset:", df.shape)
    print(df["label"].value_counts())
    return df[["text", "label"]]

def detect_url_columns(df):
    cols = {c.strip().lower(): c for c in df.columns}
    url_col = None
    for c in ["url", "urls", "link", "uri"]:
        if c in cols:
            url_col = cols[c]; break
    if url_col is None and "URL" in df.columns:
        url_col = "URL"

    label_col = None
    for c in ["label", "type", "category", "class"]:
        if c in cols:
            label_col = cols[c]; break
    if label_col is None and "Label" in df.columns:
        label_col = "Label"

    if url_col is None or label_col is None:
        raise ValueError(f"Could not detect URL/label columns. Found: {df.columns.tolist()}")
    return url_col, label_col

def normalize_url_label(x):
    s = str(x).strip().lower()
    if s in ["1", "phishing", "malware", "defacement", "malicious", "bad"]:
        return 1
    if s in ["0", "benign", "legitimate", "good", "safe", "normal"]:
        return 0
    # unknown -> None so we can drop it instead of silently forcing benign
    return None

def extract_url_features(url: str):
    url = "" if url is None else str(url)
    try:
        parsed = urllib.parse.urlparse(url)
        domain = (parsed.netloc or "").lower()
    except Exception:
        domain = ""

    # FIXED regex: no double escaping
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

# ========================
# MAIN
# ========================
def main(email_path, url_path, out_dir):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, out_dir)
    graph_dir = os.path.join(base_dir, "graphs")
    ensure_dir(model_dir)
    ensure_dir(graph_dir)

    # ------------------------
    # 1) EMAIL LOGISTIC (TF-IDF)
    # ------------------------
    print("\n--- Training Logistic Regression (Email) ---")
    df_email = load_email_dataset(email_path)
    X_text = df_email["text"].values
    y = df_email["label"].values

    # Balance
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_text.reshape(-1, 1), y)
    X_res = X_res.ravel()

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english", min_df=2)
    X_vec = vectorizer.fit_transform(X_res)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y_res, test_size=0.25, random_state=42, stratify=y_res
    )

    log_model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    log_model.fit(X_train, y_train)

    y_pred = log_model.predict(X_test)
    y_prob = log_model.predict_proba(X_test)[:, 1]
    acc_log = accuracy_score(y_test, y_pred)

    print(f"✅ Logistic Accuracy: {acc_log:.4f}")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_title("Confusion Matrix - Email Logistic")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    save_graph(fig, os.path.join(graph_dir, "email_logistic_confusion.png"))

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], lw=2, linestyle="--")
    ax.legend()
    ax.set_title("ROC Curve - Email Logistic")
    save_graph(fig, os.path.join(graph_dir, "email_logistic_roc.png"))

    joblib.dump(log_model, os.path.join(model_dir, "phish_logistic_model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "phish_tfidf_vectorizer.pkl"))

    # ------------------------
    # 2) EMAIL LSTM
    # ------------------------
    print("\n--- Training LSTM (Email) ---")
    max_words = 5000
    max_len = 200

    # Use the same resampled text for training the tokenizer and sequences
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_res)

    seqs = tokenizer.texts_to_sequences(X_res)
    X_pad = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")

    X_train, X_test, y_train, y_test = train_test_split(
        X_pad, y_res, test_size=0.25, random_state=42, stratify=y_res
    )

    model = Sequential([
        Embedding(max_words, 64, input_length=max_len),
        Bidirectional(LSTM(64)),
        Dropout(0.35),
        Dense(32, activation="relu"),
        Dropout(0.25),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=8,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    acc_lstm = accuracy_score(y_test, y_pred)
    print(f"✅ LSTM Accuracy: {acc_lstm:.4f}")
    print(classification_report(y_test, y_pred))

    # Graphs
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["accuracy"], label="Train")
    ax.plot(history.history["val_accuracy"], label="Validation")
    ax.set_title("LSTM Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    save_graph(fig, os.path.join(graph_dir, "email_lstm_accuracy.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["loss"], label="Train")
    ax.plot(history.history["val_loss"], label="Validation")
    ax.set_title("LSTM Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    save_graph(fig, os.path.join(graph_dir, "email_lstm_loss.png"))

    model.save(os.path.join(model_dir, "phish_lstm_model.h5"))
    joblib.dump(tokenizer, os.path.join(model_dir, "tokenizer.pkl"))

    # Also save tokenizer JSON for portability
    with open(os.path.join(model_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())

    # ------------------------
    # 3) URL MODEL (RF + Calibration)
    # ------------------------
    print("\n--- Training URL Model (RandomForest + Calibration) ---")
    df_url = pd.read_csv(url_path) if url_path.lower().endswith(".csv") else pd.read_excel(url_path)
    url_col, label_col = detect_url_columns(df_url)

    df_url = df_url[[url_col, label_col]].dropna().copy()
    df_url["label"] = df_url[label_col].apply(normalize_url_label)
    df_url = df_url.dropna(subset=["label"])
    df_url["label"] = df_url["label"].astype(int)
    df_url["url"] = df_url[url_col].astype(str)

    feats = df_url["url"].apply(extract_url_features)
    X = pd.DataFrame(feats.tolist())
    y = df_url["label"].values

    # Save feature order for inference-time consistency
    feat_path = os.path.join(model_dir, "url_features.json")
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    # Class distribution graph
    fig, ax = plt.subplots(figsize=(5, 3))
    pd.Series(y).value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("URL Class Distribution (0=Benign,1=Malicious)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    save_graph(fig, os.path.join(graph_dir, "url_class_distribution.png"))

    if len(np.unique(y)) < 2:
        raise ValueError("URL dataset has only one class after normalization. Cannot train a classifier.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2
    )
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc_url = accuracy_score(y_test, y_pred)
    print(f"✅ URL Accuracy: {acc_url:.4f}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"], ax=ax)
    ax.set_title("Confusion Matrix - URL Model")
    save_graph(fig, os.path.join(graph_dir, "url_confusion_matrix.png"))

    joblib.dump(model, os.path.join(model_dir, "url_ml_model.pkl"))

    # ------------------------
    # 4) SUMMARY BAR
    # ------------------------
    models = ["Email Logistic", "Email LSTM", "URL Model"]
    accuracies = [acc_log, acc_lstm, acc_url]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(models, accuracies)
    ax.set_title("Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    for bar in bars:
        yv = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yv + 0.01, f"{yv:.2f}", ha="center")
    save_graph(fig, os.path.join(graph_dir, "combined_accuracy_comparison.png"))

    print("\n✅ All models trained + graphs saved.")
    print("📁 Models:", model_dir)
    print("📁 Graphs:", graph_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--email_data", required=True, help="Path to email dataset (csv/xlsx).")
    parser.add_argument("--url_data", required=True, help="Path to merged URL dataset (csv/xlsx).")
    parser.add_argument("--out_dir", default="models", help="Folder to save models.")
    args = parser.parse_args()
    main(args.email_data, args.url_data, args.out_dir)
