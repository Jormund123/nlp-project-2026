#!/usr/bin/env python3
"""
Simple EDA script for two parquet files (task_a_subset and task_b_subset).
Prints: avg length (chars), median, min, max, label counts, language counts.
"""
import io
import sys
from statistics import mean, median

try:
    import requests
    import pandas as pd
except Exception as e:
    print("Missing required packages. Please run: python -m pip install pandas pyarrow requests", file=sys.stderr)
    raise

RAW_URLS = {
    "task_a": "https://raw.githubusercontent.com/Jormund123/nlp-project-2026/master/data/task_a_subset.parquet",
    "task_b": "https://raw.githubusercontent.com/Jormund123/nlp-project-2026/master/data/task_b_subset.parquet",
}

PREFERRED_TEXT_COLS = ["code", "text", "content", "snippet", "body", "sentence", "post"]
PREFERRED_LABEL_COLS = ["label", "labels", "target", "class"]
PREFERRED_LANG_COLS = ["language", "lang", "lang_code"]


def infer_column(columns, preferred):
    for p in preferred:
        if p in columns:
            return p
    return None


def choose_text_column(df):
    cols = list(df.columns)
    col = infer_column(cols, PREFERRED_TEXT_COLS)
    if col:
        return col
    # fallback: choose first object/string column with many non-null strings
    string_cols = [c for c in cols if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]
    if not string_cols:
        return None
    # choose the column with the highest average length
    best = None
    best_avg = -1
    for c in string_cols:
        s = df[c].dropna().astype(str)
        if len(s)==0:
            continue
        avg = s.map(len).mean()
        if avg > best_avg:
            best_avg = avg
            best = c
    return best


def choose_col_by_preference_or_first(df, preferred_list):
    col = infer_column(list(df.columns), preferred_list)
    if col:
        return col
    # fallback: return first column with few unique values (categorical-like)
    for c in df.columns:
        if df[c].nunique(dropna=True) <= 50:
            return c
    return None


def analyze_df(df, name):
    print(f"\n==== Analysis for {name} ====")
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")

    text_col = choose_text_column(df)
    if not text_col:
        print("No suitable text column found. Columns present:", list(df.columns))
    else:
        s = df[text_col].fillna("").astype(str).map(len)
        print(f"Text column used: {text_col}")
        print(f"Avg length (chars): {s.mean():.2f}")
        print(f"Median length: {s.median():.2f}")
        print(f"Min length: {int(s.min())}")
        print(f"Max length: {int(s.max())}")

    label_col = choose_col_by_preference_or_first(df, PREFERRED_LABEL_COLS)
    if not label_col:
        print("No label-like column found.")
    else:
        print(f"\nLabel column used: {label_col}")
        try:
            counts = df[label_col].value_counts(dropna=False)
            print(counts.to_string())
        except Exception as e:
            print("Could not compute label counts:", e)

    lang_col = choose_col_by_preference_or_first(df, PREFERRED_LANG_COLS)
    if not lang_col:
        # try to find a short-string column with limited unique values
        candidates = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype==object]
        for c in candidates:
            if df[c].nunique(dropna=True) <= 50 and df[c].dropna().astype(str).map(len).mean() < 10:
                lang_col = c
                break

    if not lang_col:
        print("No language-like column found.")
    else:
        print(f"\nLanguage column used: {lang_col}")
        try:
            counts = df[lang_col].value_counts(dropna=False)
            print(counts.to_string())
        except Exception as e:
            print("Could not compute language counts:", e)


def process_url(url, name):
    print(f"\nDownloading {name} from {url} ...")
    r = requests.get(url)
    r.raise_for_status()
    bio = io.BytesIO(r.content)
    try:
        df = pd.read_parquet(bio, engine="pyarrow")
    except Exception as ex:
        print("Failed to read parquet:", ex, file=sys.stderr)
        raise
    analyze_df(df, name)


if __name__ == "__main__":
    for key, url in RAW_URLS.items():
        try:
            process_url(url, key)
        except Exception as e:
            print(f"Error processing {key}: {e}", file=sys.stderr)
            # continue to next

    print("\nDone.")
