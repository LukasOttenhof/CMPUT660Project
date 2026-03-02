"""
Generate rq5_tone_*.csv files from parquets with VADER sentiment analysis.
Updated to use the data_loader with 3-way split (Before, After Human, After Agent).
"""

from __future__ import annotations
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure local imports work for data_loader
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from data_loader import load_all

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
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

def process_dataset(data: dict, base_key: str) -> pd.DataFrame:
    """
    Extracts before, after_human, and after_agent data from the loader dictionary
    and applies VADER analysis.
    """
    dfs = []
    
    groups = [
        (f"{base_key}_before", "before"),
        (f"{base_key}_after_human", "after_human"),
        (f"{base_key}_after_agent", "after_agent")
    ]
    
    for key, group_label in groups:
        if key not in data:
            print(f"[WARN] Key {key} not found in loaded data.")
            continue
            
        df = data[key].copy()
        if df.empty:
            continue
            
        if "text" not in df.columns:
            print(f"[ERROR] 'text' column missing in {key}")
            continue

        # Normalize date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        elif "created_at" in df.columns:
            df["date"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        
        # Prepare subset
        df_subset = df[["text", "date"]].copy()
        df_subset["group"] = group_label
        dfs.append(df_subset)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    
    print(f"Applying VADER sentiment analysis to {len(combined)} {base_key} documents...")
    tones = combined["text"].astype(str).apply(score_tone)
    
    combined["neg"] = tones.apply(lambda d: d["neg"])
    combined["neu"] = tones.apply(lambda d: d["neu"])
    combined["pos"] = tones.apply(lambda d: d["pos"])
    combined["compound"] = tones.apply(lambda d: d["compound"])
    combined["sentiment_cat"] = combined["compound"].apply(categorize_sentiment)
    
    return combined

def run():
    """Generate all tone CSV files using the data_loader"""
    
    print("Loading all data via data_loader...")
    all_data = load_all()
    
    datasets = [
        "commit_messages",
        "pr_bodies",
        "issue_bodies",
        "review_comments"
    ]
    
    for name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {name}...")
        print(f"{'='*60}")
        
        df = process_dataset(all_data, name)
        
        if df.empty:
            print(f"[SKIP] No data for {name}")
            continue
        
        # Save to CSV
        output_file = OUTPUT_TABLES / f"rq5_tone_{name}.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Saved {len(df)} records to {output_file}")
        
        # Print group-wise statistics
        print(f"\nSentiment Distribution for {name}:")
        for group in ["before", "after_human", "after_agent"]:
            subset = df[df["group"] == group]
            if subset.empty:
                continue
                
            print(f"  {group.upper()} ({len(subset)} docs):")
            for sent in ["negative", "neutral", "positive"]:
                count = (subset["sentiment_cat"] == sent).sum()
                pct = 100 * count / len(subset)
                print(f"    {sent:<10}: {count:6d} ({pct:5.2f}%)")

if __name__ == "__main__":
    run()