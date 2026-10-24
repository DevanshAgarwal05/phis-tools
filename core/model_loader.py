# core/model_loader.py
import os

# Must be set before TensorFlow/Keras are imported anywhere in the process.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from config import settings as cfg
from utils.text_utils import clean_text
from utils.url_utils import canonicalize_url, extract_host
from utils.model_utils import _is_tfidf_fitted, _load_json
from analyzers.url.selectors import sel_url, sel_host, sel_struct

try:
    from keras.saving import register_keras_serializable
except Exception:
    from tensorflow.keras.utils import register_keras_serializable

try:
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
except Exception:
    tokenizer_from_json = None


# ----------------------------
# Keras compatibility patches
# ----------------------------
@register_keras_serializable(package="Custom")
class PatchedInputLayer(tf.keras.layers.InputLayer):
    @classmethod
    def from_config(cls, config):
        if "batch_shape" in config and "batch_input_shape" not in config:
            config["batch_input_shape"] = config.pop("batch_shape")
        return super().from_config(config)


@register_keras_serializable(package="Custom")
def DTypePolicy(name="float32"):
    return tf.keras.mixed_precision.Policy(name)


# ----------------------------
# Model loading
# ----------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    models = {
        "logistic": None,
        "vectorizer": None,
        "hybrid_meta": None,
        "tokenizer": None,
        "lstm": None,

        # URL pipeline model (host+url FeatureUnion)
        "url_text_model": None,

        "errors": {
            "logistic": None,
            "vectorizer": None,
            "hybrid_meta": None,
            "tokenizer": None,
            "lstm": None,
            "url_text_model": None,
        },
        "config": {
            "max_len": cfg.LSTM_DEFAULT_MAX_LEN
        }
    }

    # ----------------------------
    # Load URL Host+URL Pipeline model
    # ----------------------------
    try:
        if os.path.exists(cfg.URL_PIPELINE_MODEL):
            # Compatibility: old pickles may reference sel_url/sel_host in __main__.
            try:
                import __main__ as _main
                setattr(_main, "sel_url", sel_url)
                setattr(_main, "sel_host", sel_host)
                setattr(_main, "sel_struct", sel_struct)
            except Exception:
                pass
            models["url_text_model"] = joblib.load(cfg.URL_PIPELINE_MODEL)
        else:
            models["errors"]["url_text_model"] = (
                "models/url_tfidf_logistic_model.pkl not found. Retrain URL model."
            )
    except Exception as e:
        models["errors"]["url_text_model"] = f"URL pipeline load failed ({type(e).__name__}): {e}"

    # ----------------------------
    # TF-IDF vectorizer for EMAIL logistic (try multiple names)
    # ----------------------------
    vec_candidates = cfg.VECTORIZER_CANDIDATES
    vec_loaded = None
    vec_errs = []

    for p in vec_candidates:
        if os.path.exists(p):
            try:
                v = joblib.load(p)
                if _is_tfidf_fitted(v):
                    vec_loaded = v
                    break
                else:
                    vec_errs.append(f"{p}: not fitted")
            except Exception as e:
                vec_errs.append(f"{p}: {e}")

    if vec_loaded is None:
        models["errors"]["vectorizer"] = "No fitted TF-IDF vectorizer found. " + " | ".join(vec_errs)
    else:
        models["vectorizer"] = vec_loaded

    # EMAIL Logistic model
    try:
        models["logistic"] = joblib.load(cfg.EMAIL_LOGISTIC_MODEL)
    except Exception as e:
        models["errors"]["logistic"] = f"Logistic model load failed: {e}"

    # Hybrid meta model
    try:
        if os.path.exists(cfg.HYBRID_META_MODEL):
            models["hybrid_meta"] = joblib.load(cfg.HYBRID_META_MODEL)
        else:
            models["errors"]["hybrid_meta"] = "models/hybrid_model.pkl not found (run train_hybrid_model.py)."
    except Exception as e:
        models["errors"]["hybrid_meta"] = f"Hybrid meta load failed: {e}"

    # Tokenizer: prefer tokenizer.json (portable), fallback to tokenizer.pkl
    tok = None
    if os.path.exists(cfg.TOKENIZER_JSON) and tokenizer_from_json is not None:
        try:
            with open(cfg.TOKENIZER_JSON, "r", encoding="utf-8") as f:
                tok_json = f.read()
            tok = tokenizer_from_json(tok_json)
        except Exception as e:
            models["errors"]["tokenizer"] = f"Tokenizer JSON load failed: {e}"

    if tok is None:
        try:
            tok = joblib.load(cfg.TOKENIZER_PKL)
        except Exception as e:
            models["errors"]["tokenizer"] = models["errors"]["tokenizer"] or f"Tokenizer PKL load failed: {e}"

    models["tokenizer"] = tok

    # LSTM model
    try:
        if os.path.exists(cfg.LSTM_MODEL):
            models["lstm"] = tf.keras.models.load_model(
                str(cfg.LSTM_MODEL),
                compile=False,
                custom_objects={
                    "PatchedInputLayer": PatchedInputLayer,
                    "InputLayer": PatchedInputLayer,
                    "DTypePolicy": DTypePolicy,
                },
            )
            meta = _load_json(cfg.EMAIL_META)
            if isinstance(meta, dict) and "max_len" in meta:
                models["config"]["max_len"] = int(meta["max_len"])
        else:
            models["errors"]["lstm"] = "models/phish_lstm_model.h5 not found (run train_deep_model.py)."
    except Exception as e:
        models["errors"]["lstm"] = f"LSTM load failed: {e}"

    return models


models = load_models()


# ----------------------------
# Email prediction (Logistic, LSTM, Hybrid meta)
# ----------------------------
def predict_email_models(subject, body):
    raw_text = f"{subject} {body}".strip()
    text = clean_text(raw_text)

    log_prob = None
    lstm_prob = None
    hybrid_prob = None
    email_note = None

    tokenizer = models.get("tokenizer")
    lstm_model = models.get("lstm")
    logistic_model = models.get("logistic")
    vectorizer = models.get("vectorizer")
    hybrid_meta = models.get("hybrid_meta")

    # LSTM prob
    if lstm_model is not None and tokenizer is not None:
        vocab_size = 5000
        try:
            vocab_size = int(lstm_model.layers[0].input_dim)
        except Exception:
            pass

        seq = tokenizer.texts_to_sequences([text])
        seq = [[tok if tok < vocab_size else 1 for tok in s] for s in seq]
        max_len = int(models["config"].get("max_len", 200))
        seq_padded = tf.keras.preprocessing.sequence.pad_sequences(
            seq, maxlen=max_len, padding="post", truncating="post"
        )
        lstm_prob = float(lstm_model.predict(seq_padded, verbose=0)[0][0])

    # Logistic prob
    if logistic_model is not None and vectorizer is not None:
        try:
            X_vec = vectorizer.transform([text])
            log_prob = float(logistic_model.predict_proba(X_vec)[0][1])
        except Exception as e:
            email_note = f"Logistic disabled (vectorizer issue): {e}"
            log_prob = None

    # Hybrid meta (preferred)
    if hybrid_meta is not None and (log_prob is not None) and (lstm_prob is not None):
        try:
            X_meta = np.array([[log_prob, lstm_prob]], dtype=float)
            hybrid_prob = float(hybrid_meta.predict_proba(X_meta)[0][1])
            email_note = "Using trained Hybrid Meta model (logistic + LSTM)."
        except Exception as e:
            hybrid_prob = (log_prob + lstm_prob) / 2.0
            email_note = f"Hybrid meta failed ({e}); using average of Logistic+LSTM."

    # Fallbacks
    if hybrid_prob is None:
        if (log_prob is not None) and (lstm_prob is not None):
            hybrid_prob = (log_prob + lstm_prob) / 2.0
            email_note = email_note or "Using average of Logistic+LSTM (meta model not available)."
        elif lstm_prob is not None:
            hybrid_prob = lstm_prob
            email_note = email_note or "Using LSTM only (Logistic/TF-IDF unavailable)."
        elif log_prob is not None:
            hybrid_prob = log_prob
            email_note = email_note or "Using Logistic only (LSTM unavailable)."
        else:
            hybrid_prob = 0.5
            email_note = email_note or "Email ML unavailable; using neutral score."

    return log_prob, lstm_prob, hybrid_prob, email_note


# ----------------------------
# URL prediction (Pipeline model expects DataFrame with url+host)
# ----------------------------
def predict_url_ml(url: str):
    model = models.get("url_text_model")
    if model is None:
        return 0.5

    url_c = canonicalize_url(url)
    if not url_c:
        return 0.5

    host = extract_host(url_c)
    if not host:
        return 0.5

    X = pd.DataFrame([{"url": url_c, "host": host}])

    try:
        probs = model.predict_proba(X)[0]
        classes = [str(c) for c in getattr(model.named_steps["clf"], "classes_", [])]
        if "1" in classes:
            return float(probs[classes.index("1")])
        return float(probs[1]) if len(probs) > 1 else float(probs[0])
    except Exception:
        return 0.5
