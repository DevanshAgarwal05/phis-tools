# utils/model_utils.py
import json


def _is_tfidf_fitted(vec):
    if vec is None:
        return False
    if hasattr(vec, "idf_"):
        return True
    tfidf = getattr(vec, "_tfidf", None)
    if tfidf is not None and hasattr(tfidf, "idf_"):
        return True
    return False


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
