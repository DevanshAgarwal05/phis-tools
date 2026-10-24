# hybrid_predictor.py
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.url_features import extract_features

# Load all models
logistic_model = joblib.load("models/phish_logistic_model.pkl")
vectorizer = joblib.load("models/phish_tfidf_vectorizer.pkl")
lstm_model = tf.keras.models.load_model("models/phish_lstm_model.h5")
tokenizer = joblib.load("models/tokenizer.pkl")
url_model = joblib.load("models/url_model.joblib")

def predict_hybrid(email_text, urls):
    # 1️⃣ Logistic model (TF-IDF)
    X_tfidf = vectorizer.transform([email_text])
    log_prob = logistic_model.predict_proba(X_tfidf)[0][1]

    # 2️⃣ LSTM model
    seq = tokenizer.texts_to_sequences([email_text])
    padded = pad_sequences(seq, maxlen=200)
    lstm_prob = lstm_model.predict(padded)[0][0]

    # 3️⃣ URL model (average over all URLs)
    if urls:
        url_features = [extract_features(u) for u in urls]
        url_probs = [url_model.predict_proba([list(f.values())])[0][1] for f in url_features]
        avg_url_prob = np.mean(url_probs)
    else:
        avg_url_prob = 0.5  # Neutral score if no URL present

    # 4️⃣ Weighted fusion
    final_prob = (0.6 * lstm_prob) + (0.3 * log_prob) + (0.1 * avg_url_prob)
    label = "PHISHING" if final_prob > 0.5 else "LEGITIMATE"

    return {
        "label": label,
        "final_risk": round(final_prob * 100, 2),
        "lstm_prob": round(lstm_prob * 100, 2),
        "log_prob": round(log_prob * 100, 2),
        "url_prob": round(avg_url_prob * 100, 2),
    }
