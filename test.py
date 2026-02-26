import pandas as pd
import numpy as np

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
BASE_DIR = "/Volumes/T7-Shield/CMPUT660Project/inputs/50prs/"

FILES = {
    "Commits": ("commits_before.parquet", "commits_after.parquet"),
    "PRs": ("pull_requests_before.parquet", "pull_requests_after.parquet"),
    "Issues": ("issues_before.parquet", "issues_after.parquet"),
    "Issue Text": ("issue_bodies_before.parquet", "issue_bodies_after.parquet")
}

def get_stats(series):
    """Calculates the specific stats required for the table."""
    return {
        "Mean": series.mean(),
        "Median": series.median(),
        "P25": series.quantile(0.25),
        "P75": series.quantile(0.75),
        "Variance": series.var(),
        "Std": series.std()
    }

def process_metric(name, file_before, file_after, is_text=False):
    df_b = pd.read_parquet(BASE_DIR + file_before)
    df_a = pd.read_parquet(BASE_DIR + file_after)

    if is_text:
        # Calculate word count for text length metrics
        # We assume the column is named 'text'
        val_b = df_b['text'].dropna().apply(lambda x: len(str(x).split()))
        val_a = df_a['text'].dropna().apply(lambda x: len(str(x).split()))
    else:
        # Calculate counts per repository
        val_b = df_b.groupby('repo').size()
        val_a = df_a.groupby('repo').size()

    stats_b = get_stats(val_b)
    stats_a = get_stats(val_a)
    
    # Calculate Diff (After - Before)
    stats_diff = {k: stats_a[k] - stats_b[k] for k in stats_b.keys()}

    return stats_b, stats_a, stats_diff

def main():
    metrics = [
        ("Commits per repository", "Commits", False),
        ("Pull requests per repository", "PRs", False),
        ("Issue text length", "Issue Text", True),
        ("Issues per repository", "Issues", False)
    ]

    print(f"{'Metric':<30} {'Period':<10} {'Mean':>10} {'Median':>10} {'P25':>10} {'P75':>10} {'Variance':>10} {'Std':>10}")
    print("-" * 105)

    for label, key, is_text in metrics:
        fb, fa = FILES[key]
        b, a, d = process_metric(label, fb, fa, is_text)
        
        print(f"\n{label}")
        for period, data in zip(["before", "after", "diff"], [b, a, d]):
            print(f"{'':<30} {period:<10} "
                  f"{data['Mean']:>10.2f} {data['Median']:>10.2f} "
                  f"{data['P25']:>10.2f} {data['P75']:>10.2f} "
                  f"{data['Variance']:>10.0f} {data['Std']:>10.2f}")

if __name__ == "__main__":
    main()