#This file outputs parquet structures for debugging and ease of use
import pandas as pd
from pathlib import Path
import numpy as np

BASE = Path("inputs/processed")

PARQUET_FILES = [
    "commit_messages_before.parquet",
    "commit_messages_after.parquet",
    "commits_before.parquet",
    "commits_after.parquet",
    "discussion_comments_before.parquet",
    "discussion_comments_after.parquet",
    "discussion_topics_before.parquet",
    "discussion_topics_after.parquet",
    "issue_bodies_before.parquet",
    "issue_bodies_after.parquet",
    "issues_before.parquet",
    "issues_after.parquet",
    "pr_bodies_before.parquet",
    "pr_bodies_after.parquet",
    "pull_requests_before.parquet",
    "pull_requests_after.parquet",
    "review_comments_before.parquet",
    "review_comments_after.parquet",
    "reviews_before.parquet",
    "reviews_after.parquet",
]


def header(title):
    print("\n" + "=" * 100)
    print(f"{title}")
    print("=" * 100)


def inspect_df(name, df):
    header(f"FILE: {name}")

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("Column names:", list(df.columns))

    print("\ndtypes:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isna().sum())

    date_cols = [c for c in df.columns if df[c].dtype == "datetime64[ns]"]

    if date_cols:
        print("\nDate columns (min/max):")
        for c in date_cols:
            print(f"  {c}: {df[c].min()} → {df[c].max()}")

    text_cols = [
        c for c in df.columns
        if "text" in c.lower()
        or "body" in c.lower()
        or "message" in c.lower()
        or "comment" in c.lower()
    ]

    if text_cols:
        print("\nText length stats:")
        for c in text_cols:
            try:
                cleaned = df[c].fillna("").astype(str)
                lengths = cleaned.str.split().str.len()
                print(f"\n  • Column '{c}'")
                print(f"      mean words: {lengths.mean():.2f}")
                print(f"      median:     {lengths.median():.2f}")
                print(f"      p90:        {lengths.quantile(0.9):.2f}")
                print(f"      max:        {lengths.max()}")
            except Exception:
                pass

    #Sample row from each 
    print("\nSample row:")
    if len(df) > 0:
        print(df.sample(1))

    #Category counts
    print("\nCategorical counts (if any):")
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique() <= 20:
            print(f"\n  • {col} (unique={df[col].nunique()})")
            print(df[col].value_counts())

def main():
    for fname in PARQUET_FILES:
        fpath = BASE / fname

        if not fpath.exists():
            print(f"Missing file: {fname}")
            continue

        try:
            df = pd.read_parquet(fpath)
            inspect_df(fname, df)

        except Exception as e:
            print(f"Error loading {fname}: {e}")


if __name__ == "__main__":
    main()
