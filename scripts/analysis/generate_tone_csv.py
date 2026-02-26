"""
Generate rq5_tone_*.csv files from parquets with VADER sentiment analysis.

This script:
1. Loads parquet files (commit_messages_before/after, pr_bodies_before/after, etc.)
2. Applies VADER sentiment analysis to the text
3. Combines before and after data
4. Saves as CSV files in outputs/rq5/tables/

Output files:
- rq5_tone_commit_messages.csv
- rq5_tone_pr_bodies.csv
- rq5_tone_issue_bodies.csv
- rq5_tone_review_comments.csv
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "inputs" / "50prs"
OUTPUT_TABLES = ROOT / "outputs" / "rq5" / "tables"
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()

def score_tone(text: str) -> dict:
    """Score sentiment of text using VADER"""
    if not isinstance(text, str) or not text.strip():
        return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
    return analyzer.polarity_scores(text)

def categorize_sentiment(compound: float) -> str:
    """Categorize sentiment based on compound score"""
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

def load_and_process(before_file: str, after_file: str, text_col: str = "text") -> pd.DataFrame:
    """
    Load before/after parquets and apply sentiment analysis.
    
    Args:
        before_file: filename of before parquet
        after_file: filename of after parquet
        text_col: column name containing text to analyze
    
    Returns:
        Combined dataframe with sentiment scores
    """
    dfs = []
    
    for filename, group in [(before_file, "before"), (after_file, "after")]:
        filepath = INPUTS / filename
        if not filepath.exists():
            print(f"[WARN] Missing: {filepath}")
            continue
        
        print(f"Loading {filename}...")
        df = pd.read_parquet(filepath)
        
        if df.empty:
            continue
        
        # Keep only necessary columns
        if text_col not in df.columns:
            print(f"[ERROR] Column '{text_col}' not found in {filename}")
            continue
        
        # Normalize date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        elif "created_at" in df.columns:
            df["date"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        
        # Ensure we have required columns
        required_cols = [text_col, "date"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"[ERROR] Missing columns {missing} in {filename}")
            continue
        
        # Select and rename columns
        df_subset = df[[text_col, "date"]].copy()
        df_subset["text"] = df_subset[text_col]
        df_subset["date"] = df_subset["date"]
        df_subset["time"] = df_subset["date"]  # Duplicate for compatibility
        df_subset["group"] = group
        
        dfs.append(df_subset[["text", "date", "time", "group"]])
    
    if not dfs:
        print(f"[ERROR] No data loaded for {before_file}/{after_file}")
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} documents")
    
    # Apply VADER sentiment analysis
    print("Applying VADER sentiment analysis...")
    tones = combined["text"].astype(str).apply(score_tone)
    combined["neg"] = tones.apply(lambda d: d["neg"])
    combined["neu"] = tones.apply(lambda d: d["neu"])
    combined["pos"] = tones.apply(lambda d: d["pos"])
    combined["compound"] = tones.apply(lambda d: d["compound"])
    combined["sentiment_cat"] = combined["compound"].apply(categorize_sentiment)
    
    return combined

def run():
    """Generate all tone CSV files"""
    
    # Mapping of output file names to input parquet pairs
    datasets = {
        "commit_messages": ("commit_messages_before.parquet", "commit_messages_after.parquet"),
        "pr_bodies": ("pr_bodies_before.parquet", "pr_bodies_after.parquet"),
        "issue_bodies": ("issue_bodies_before.parquet", "issue_bodies_after.parquet"),
        "review_comments": ("review_comments_before.parquet", "review_comments_after.parquet"),
    }
    
    for name, (before_file, after_file) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Processing {name}...")
        print(f"{'='*60}")
        
        df = load_and_process(before_file, after_file, text_col="text")
        
        if df.empty:
            print(f"[SKIP] No data for {name}")
            continue
        
        # Save to CSV
        output_file = OUTPUT_TABLES / f"rq5_tone_{name}.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Saved {len(df)} records to {output_file}")
        
        # Print statistics
        print(f"\nStatistics for {name}:")
        print(f"  Total documents: {len(df)}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"\n  Sentiment distribution:")
        for sent in ["negative", "neutral", "positive"]:
            count = (df["sentiment_cat"] == sent).sum()
            pct = 100 * count / len(df)
            print(f"    {sent}: {count:6d} ({pct:5.2f}%)")
        
        print(f"\n  By group:")
        for group in ["before", "after"]:
            subset = df[df["group"] == group]
            print(f"    {group}: {len(subset)} documents")
            for sent in ["negative", "neutral", "positive"]:
                count = (subset["sentiment_cat"] == sent).sum()
                pct = 100 * count / len(subset) if len(subset) > 0 else 0
                print(f"      {sent}: {count:6d} ({pct:5.2f}%)")

if __name__ == "__main__":
    run()
