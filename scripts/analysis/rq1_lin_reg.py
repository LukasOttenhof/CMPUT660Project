from pathlib import Path
import numpy as np
import pandas as pd
from data_loader import load_all
from sklearn.linear_model import LinearRegression
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]

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

def build_daily_cumulative(df):
    df = df.copy()
    # 1. Count events per day
    daily_counts = df.groupby("day").size()

    # 2. Fill in missing days with 0
    full_idx = pd.date_range(start=daily_counts.index.min(), 
                             end=daily_counts.index.max(), 
                             freq='D')

    daily = daily_counts.reindex(full_idx, fill_value=0).reset_index(name="count")
    
    # 3. Numeric time and Cumulative Sum
    daily["t"] = np.arange(len(daily))
    daily["cumulative"] = daily["count"].cumsum()

    return daily 

def calculate_stats(df_before, df_after, label):
    # --- 1. Get Velocity (Slope) ---
    def get_slope(df):
        X = df[["t"]].values
        y = df["cumulative"].values
        model = LinearRegression().fit(X, y)
        return model.coef_[0]

    slope_b = get_slope(df_before)
    slope_a = get_slope(df_after)
    
    # --- 2. Get Statistical Significance (Daily Counts) ---
    group1 = df_before["count"].values
    group2 = df_after["count"].values

    if len(group1) < 3 or len(group2) < 3:
        return {"Metric": label, "Error": "insufficient_data"}

    # Mann-Whitney U Test (P-Value)
    u_stat, p_value = stats.mannwhitneyu(group2, group1, alternative='two-sided')

    # Cohen's d (Effect Size)
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (np.mean(group2) - np.mean(group1)) / pooled_se if pooled_se > 0 else 0

    return {
        "Metric": label,
        "Slope Before": slope_b,
        "Slope After": slope_a,
        "P-Value": p_value,
        "Effect Size (d)": cohens_d
    }

def main():
    data = load_all()
    results = []

    for metric in METRICS:
        key, subtype = METRIC_MAP[metric]

        df_b = data[f"{key}_before"].copy()
        df_a = data[f"{key}_after"].copy()

        if subtype:
            df_b = df_b[df_b["activity_type"] == subtype]
            df_a = df_a[df_a["activity_type"] == subtype]

        # limit to last 3 years of 'Before' data
        df_b["date"] = pd.to_datetime(df_b["date"])
        df_b["day"] = df_b["date"].dt.floor("D")
        
        df_a["date"] = pd.to_datetime(df_a["date"])
        df_a["day"] = df_a["date"].dt.floor("D")

        # Find the end of the 'Before' period
        before_end = df_b["date"].max()
        
        # Calculate the cutoff date (3 years prior to the end)
        cutoff_date = before_end - pd.DateOffset(years=3)
    
        # Apply the filter
        df_b = df_b[df_b["date"] >= cutoff_date]
   
        # Safety: Ensure we didn't delete everything
        if df_b.empty:
            continue

        # ---------------------------------------------------------

        cum_before = build_daily_cumulative(df_b)
        cum_after  = build_daily_cumulative(df_a)

        results.append(calculate_stats(cum_before, cum_after, metric))

    results_df = pd.DataFrame(results)

    print("\n================ METRIC VELOCITY (Last 3 Years) =================\n")
    
    pd.options.display.float_format = '{:.4f}'.format
    pd.options.display.max_columns = 10
    pd.options.display.width = 1000
    
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()