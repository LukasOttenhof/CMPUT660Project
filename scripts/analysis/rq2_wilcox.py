from pathlib import Path
import numpy as np
import pandas as pd
from data_loader import load_all
from scipy.stats import wilcoxon, chi2_contingency

pd.options.mode.chained_assignment = None

ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT / "outputs" / "RQ2" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "commits",
    "prs_created",
    "prs_merged",
    "reviews_submitted",
    "review_comments",
    "issues_opened",
    "issues_closed",
]

METRIC_MAP = {
    "commits": ("commits", None),
    "prs_created": ("pull_requests", "pr_created"),
    "prs_merged": ("pull_requests", "pr_merged"),
    "reviews_submitted": ("reviews", None),
    "review_comments": ("review_comments", None),
    "issues_opened": ("issues", "issue_opened"),
    "issues_closed": ("issues", "issue_closed"),
}
from pathlib import Path
import numpy as np
import pandas as pd
from data_loader import load_all
from scipy.stats import wilcoxon, chi2_contingency

# ... (METRICS and METRIC_MAP definitions remain the same) ...

def calculate_avg_per_dev_per_month(df):
    """Calculates the normalized intensity metric (BA, AH, AA)."""
    if df.empty or 'author' not in df.columns:
        return 0
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None) 
    df['month'] = df['date'].dt.to_period('M')
    
    # Group by month to find activity volume vs unique dev count
    monthly = df.groupby('month').agg(
        activity_count=('author', 'size'),
        dev_count=('author', 'nunique')
    ).reset_index()

    monthly['per_dev'] = np.where(
        monthly['dev_count'] > 0,
        monthly['activity_count'] / monthly['dev_count'],
        0
    )
    return monthly['per_dev'].mean()

def build_normalized_stats(data):
    """Builds the BA, AH, and AA values for statistical testing."""
    stats_data = {"before": {}, "after_h": {}, "after_a": {}}

    for metric in METRICS:
        key, subtype = METRIC_MAP[metric]
        
        # Filter subtypes if necessary (PRs created vs merged, etc)
        df_b = data[f"{key}_before"]
        df_h = data[f"{key}_after_human"]
        df_a = data[f"{key}_after_agent"]

        if subtype:
            df_b = df_b[df_b["activity_type"] == subtype]
            df_h = df_h[df_h["activity_type"] == subtype]
            df_a = df_a[df_a["activity_type"] == subtype]

        stats_data["before"][metric] = calculate_avg_per_dev_per_month(df_b)
        stats_data["after_h"][metric] = calculate_avg_per_dev_per_month(df_h)
        stats_data["after_a"][metric] = calculate_avg_per_dev_per_month(df_a)

    return stats_data

def run_stat_comparison(group_name, before_vals, target_vals):
    """Runs Wilcoxon on Intensity and Chi-Square on Ratios (Work Mix)."""
    metrics = list(before_vals.keys())
    b_vec = np.array([before_vals[m] for m in metrics])
    t_vec = np.array([target_vals[m] for m in metrics])

    # 1. Wilcoxon: Compares Intensity (BA vs AH / BA vs AA)
    stat, p_val = wilcoxon(b_vec, t_vec)
    
    # 2. Chi-Square: Compares the "DNA" / Ratios
    # We use the normalized averages as the distribution profile
    contingency = np.array([b_vec, t_vec])
    chi2, chi_p, _, _ = chi2_contingency(contingency)

    print(f"\n--- Comparison: Before vs. {group_name} ---")
    print(f"Intensity Change (Wilcoxon p): {p_val:.4f}")
    print(f"Work Mix Shift (Chi-Square p): {chi_p:.4f}")

def main():
    data = load_all()
    # ... (3-year cutoff logic) ...
    
    stats_data = build_normalized_stats(data)

    # This will now compare the 13.6 (BA) vs 6.6 (AH) type values
    run_stat_comparison("After Human (AH)", stats_data["before"], stats_data["after_h"])
    run_stat_comparison("After Agent (AA)", stats_data["before"], stats_data["after_a"])

if __name__ == "__main__":
    main()