# merge_datasets.py (UPDATED)
import os
import argparse
import pandas as pd

def detect_url_col(df):
    cols = {c.strip().lower(): c for c in df.columns}
    for k in ["url", "urls", "link", "uri"]:
        if k in cols:
            return cols[k]
    if "URL" in df.columns:
        return "URL"
    return None

def detect_label_col(df):
    cols = {c.strip().lower(): c for c in df.columns}
    for k in ["label", "type", "category", "class"]:
        if k in cols:
            return cols[k]
    if "Label" in df.columns:
        return "Label"
    return None

def normalize_url_label(x):
    s = str(x).strip().lower()
    if s in ["1", "phishing", "malware", "defacement", "malicious", "bad"]:
        return 1
    if s in ["0", "benign", "legitimate", "good", "safe", "normal"]:
        return 0
    return None  # drop unknown

def read_any(path):
    if path.lower().endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)

def main(path1, path2, output_path):
    df1 = read_any(path1)
    df2 = read_any(path2)

    print("✅ Loaded Dataset 1:", df1.shape)
    print("✅ Loaded Dataset 2:", df2.shape)

    u1, l1 = detect_url_col(df1), detect_label_col(df1)
    u2, l2 = detect_url_col(df2), detect_label_col(df2)

    if u1 is None or l1 is None:
        raise ValueError(f"Dataset 1 missing URL/label columns. Found: {df1.columns.tolist()}")
    if u2 is None or l2 is None:
        raise ValueError(f"Dataset 2 missing URL/label columns. Found: {df2.columns.tolist()}")

    df1 = df1[[u1, l1]].copy()
    df2 = df2[[u2, l2]].copy()

    df1.columns = ["URL", "Label"]
    df2.columns = ["URL", "Label"]

    df1["URL"] = df1["URL"].astype(str).str.strip()
    df2["URL"] = df2["URL"].astype(str).str.strip()

    df1["Label"] = df1["Label"].apply(normalize_url_label)
    df2["Label"] = df2["Label"].apply(normalize_url_label)

    df1 = df1.dropna(subset=["URL", "Label"])
    df2 = df2.dropna(subset=["URL", "Label"])

    df1["Label"] = df1["Label"].astype(int)
    df2["Label"] = df2["Label"].astype(int)

    # remove empty URLs
    df1 = df1[df1["URL"].str.len() > 0]
    df2 = df2[df2["URL"].str.len() > 0]

    merged = pd.concat([df1, df2], ignore_index=True)
    merged = merged.drop_duplicates(subset=["URL"])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    merged.to_csv(output_path, index=False)

    print("\n✅ Merged dataset saved to:", output_path)
    print("📊 Total URLs:", len(merged))
    print("Label distribution:\n", merged["Label"].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1", required=True, help="Path to dataset 1 (csv/xlsx).")
    parser.add_argument("--d2", required=True, help="Path to dataset 2 (csv/xlsx).")
    parser.add_argument("--out", default="data/merged_urls.csv", help="Output merged CSV path.")
    args = parser.parse_args()
    main(args.d1, args.d2, args.out)
