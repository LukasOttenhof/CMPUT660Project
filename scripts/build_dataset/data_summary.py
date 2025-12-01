import pandas as pd
from pathlib import Path
import numpy as np

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
BASE = Path("inputs/processed")

# Map friendly names to file prefixes
DATASETS = {
    "Commits": ("commits_before.parquet", "commits_after.parquet"),
    "Commit Messages": ("commit_messages_before.parquet", "commit_messages_after.parquet"),
    "Pull Requests": ("pull_requests_before.parquet", "pull_requests_after.parquet"),
    "PR Bodies": ("pr_bodies_before.parquet", "pr_bodies_after.parquet"),
    "Issues": ("issues_before.parquet", "issues_after.parquet"),
    "Issue Bodies": ("issue_bodies_before.parquet", "issue_bodies_after.parquet"),
    "Reviews": ("reviews_before.parquet", "reviews_after.parquet"),
    "Review Comments": ("review_comments_before.parquet", "review_comments_after.parquet"),
    "Discussions": ("discussion_topics_before.parquet", "discussion_topics_after.parquet"),
    "Discussion Comments": ("discussion_comments_before.parquet", "discussion_comments_after.parquet"),
}

def get_author_col(df):
    """Smartly find the author/user column."""
    possibilities = ["author", "user", "user_login", "committer", "actor"]
    for col in possibilities:
        if col in df.columns:
            return col
    return None

def get_date_col(df):
    """Smartly find the date column."""
    possibilities = ["date", "created_at", "submitted_at", "updated_at"]
    for col in possibilities:
        if col in df.columns:
            return col
    return None

def analyze_pair(name, file_before, file_after, apply_filter=True):
    path_b = BASE / file_before
    path_a = BASE / file_after
    
    # Initialize stats without dates
    stats = {
        "Dataset": name,
        "Before (Count)": 0, "After (Count)": 0,
        "Before (Authors)": 0, "After (Authors)": 0,
    }

    # --- 1. Load After Data ---
    if path_a.exists():
        df_a = pd.read_parquet(path_a)
        stats["After (Count)"] = len(df_a)
        
        auth_col_a = get_author_col(df_a)
        if auth_col_a:
            stats["After (Authors)"] = df_a[auth_col_a].nunique()
    else:
        df_a = pd.DataFrame()

    # --- 2. Load and Filter Before Data ---
    if path_b.exists():
        df_b = pd.read_parquet(path_b)
        
        # Filter Logic
        if apply_filter:
            date_col_b = get_date_col(df_b)
            if date_col_b:
                df_b[date_col_b] = pd.to_datetime(df_b[date_col_b])
                # Calculate cutoff based on the END of the before dataset
                end_date = df_b[date_col_b].max()
                if pd.notnull(end_date):
                    cutoff = end_date - pd.DateOffset(years=3)
                    df_b = df_b[df_b[date_col_b] >= cutoff]

        stats["Before (Count)"] = len(df_b)
        
        auth_col_b = get_author_col(df_b)
        if auth_col_b:
            stats["Before (Authors)"] = df_b[auth_col_b].nunique()

    return stats

def generate_report(apply_filter):
    mode_name = "3-Year Filtered" if apply_filter else "Raw (All Data)"
    filename = "dataset_stats_filtered.csv" if apply_filter else "dataset_stats_raw.csv"
    
    print(f"\nProcessing: {mode_name}...")
    
    results = []
    for name, (fb, fa) in DATASETS.items():
        try:
            row = analyze_pair(name, fb, fa, apply_filter=apply_filter)
            results.append(row)
        except Exception as e:
            print(f"⚠️ Error processing {name}: {e}")

    df_results = pd.DataFrame(results)

    # Reorder columns
    cols = [
        "Dataset", 
        "Before (Count)", "After (Count)", 
        "Before (Authors)", "After (Authors)"
    ]
    df_results = df_results[cols]

    # Print Table
    print("=" * 80)
    print(f"DATASET SUMMARY: {mode_name}")
    print("=" * 80)
    formatters = {
        'Before (Count)': '{:,.0f}'.format,
        'After (Count)': '{:,.0f}'.format,
        'Before (Authors)': '{:,.0f}'.format,
        'After (Authors)': '{:,.0f}'.format
    }
    print(df_results.to_string(index=False, formatters=formatters))
    print("=" * 80)
    
    #df_results.to_csv(filename, index=False)
    print(f"✅ Saved to '{filename}'")

def main():
    # 1. Generate Filtered Version (Consistent with Plots)
    generate_report(apply_filter=True)
    
    # 2. Generate Raw Version (All Data Mined)
    generate_report(apply_filter=False)

if __name__ == "__main__":
    main()