import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer

# ---- settings ----
DATA_PATHS = [
    "data/CEAS_08.csv",
    "data/malicious_phish.csv",
]
OUT_PATH = "models/tokenizer.pkl"

def pick_text_column(df: pd.DataFrame) -> str:
    # try common names first
    candidates = ["text", "body", "content", "email", "message", "mail", "Subject", "subject", "Body", "Email"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: largest string-like column
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        raise ValueError(f"No text column found. Columns: {list(df.columns)}")
    # choose column with longest avg length
    best = max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())
    return best

def load_texts() -> list[str]:
    texts = []
    for p in DATA_PATHS:
        if not Path(p).exists():
            continue
        df = pd.read_csv(p)
        col = pick_text_column(df)
        series = df[col].astype(str).fillna("").str.strip()
        texts.extend([t for t in series.tolist() if t])
        print(f"Loaded {len(series)} rows from {p} using column '{col}'")
    if not texts:
        raise ValueError("No texts loaded. Check DATA_PATHS and CSV contents.")
    return texts

def main():
    Path("models").mkdir(exist_ok=True)

    texts = load_texts()

    tok = Tokenizer(num_words=50000, oov_token="<OOV>")
    tok.fit_on_texts(texts)

    joblib.dump(tok, OUT_PATH)
    print(f"✅ Saved tokenizer to {OUT_PATH}")
    print(f"Vocab size (raw): {len(tok.word_index)}")

if __name__ == "__main__":
    main()
