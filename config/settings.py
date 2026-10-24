# config/settings.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent   # repo root, independent of cwd

MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

# Model artifacts (same files webapp.py loads today)
URL_PIPELINE_MODEL = MODELS_DIR / "url_tfidf_logistic_model.pkl"
EMAIL_LOGISTIC_MODEL = MODELS_DIR / "phish_logistic_model.pkl"
HYBRID_META_MODEL = MODELS_DIR / "hybrid_model.pkl"
TOKENIZER_JSON = MODELS_DIR / "tokenizer.json"
TOKENIZER_PKL = MODELS_DIR / "tokenizer.pkl"
LSTM_MODEL = MODELS_DIR / "phish_lstm_model.h5"
EMAIL_META = MODELS_DIR / "email_meta.json"

# Order matters: first fitted vectorizer wins (mirrors the previous vec_candidates list)
VECTORIZER_CANDIDATES = [
    MODELS_DIR / "phish_tfidf_vectorizer.pkl",
    MODELS_DIR / "vectorizer.pkl",
    MODELS_DIR / "tfidf_vectorizer.pkl",
    BASE_DIR / "phish_tfidf_vectorizer.pkl",
    BASE_DIR / "vectorizer.pkl",
    BASE_DIR / "tfidf_vectorizer.pkl",
]

SCAN_LOG_PATH = LOGS_DIR / "scan_logs.csv"
FEEDBACK_LOG_PATH = BASE_DIR / "feedback_log.csv"

LSTM_DEFAULT_MAX_LEN = 200
