from pathlib import Path
import numpy as np
import pandas as pd
from data_loader import load_all
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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
    # Count events per day
    daily_counts = df.groupby("day").size()
    
    # Fill in missing days
    full_idx = pd.date_range(start=daily_counts.index.min(),
                             end=daily_counts.index.max(),
                             freq='D')
    daily = daily_counts.reindex(full_idx, fill_value=0).reset_index(name="count")
    daily = daily.rename(columns={"index": "day"})
    
    # Numeric time and cumulative sum
    daily["t"] = np.arange(len(daily))
    daily["cumulative"] = daily["count"].cumsum()
    
    return daily


def calculate_stats(df_before, df_after, label, plot=True):
    # --- 1. Linear Regression Models ---
    def get_model(df):
        X = df[["t"]].values
        y = df["cumulative"].values
        model = LinearRegression().fit(X, y)
        return model, X, y

    model_b, X_b, y_b = get_model(df_before)
    model_a, X_a, y_a = get_model(df_after)

    slope_b = model_b.coef_[0]
    slope_a = model_a.coef_[0]

    # --- 2. Statistical Significance ---
    group1 = df_before["count"].values
    group2 = df_after["count"].values

    if len(group1) < 3 or len(group2) < 3:
        return {"Metric": label, "Error": "insufficient_data"}

    # Mann-Whitney U Test
    u_stat, p_value = stats.mannwhitneyu(group2, group1, alternative="two-sided")

    # Cohen's d
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (np.mean(group2) - np.mean(group1)) / pooled_se if pooled_se > 0 else 0

    # --- 3. Plotting with Seaborn ---
    if plot:
        plt.figure(figsize=(10, 5))

        # Extract dates
        dates_b = df_before["day"].values
        dates_a = df_after["day"].values

        # --- Compute mean start date for after period ---
        # Convert to numeric days since epoch
        dates_a_days = (dates_a - np.datetime64('1970-01-01')) / np.timedelta64(1, 'D')
        mean_after_start_days = dates_a_days.mean()
        mean_after_start = np.datetime64('1970-01-01') + np.timedelta64(int(mean_after_start_days), 'D')

        # Shift AFTER dates so it starts at mean start
        shift_days = (mean_after_start - dates_a.min()).astype('timedelta64[D]').astype(int)
        dates_a_shifted = dates_a + np.timedelta64(shift_days, 'D')

        # Plot cumulative lines
        sns.lineplot(x=dates_b, y=y_b.flatten(), label="Before", linewidth=1.5)
        # --- Shift AFTER cumulative data upward so it starts where BEFORE ended ---
        after_shift = y_b[-1]  # final cumulative value of BEFORE
        y_a_shifted = y_a + after_shift

        sns.lineplot(x=dates_a_shifted, y=y_a_shifted.flatten(), label="After", linewidth=1.5)


        # Regression line for BEFORE
        X_full_b = np.linspace(0, len(dates_b) - 1, 100).reshape(-1, 1)
        y_pred_b = model_b.predict(X_full_b)
        sns.lineplot(
            x=pd.to_datetime(dates_b.min()) + pd.to_timedelta(X_full_b.flatten(), unit='D'),
            y=y_pred_b.flatten(),
            label="LR Before",
            linewidth=2
        )

        # Regression line for AFTER aligned to end of BEFORE regression
        X_full_a = np.linspace(0, len(dates_a) - 1, 100).reshape(-1, 1)
        y_pred_a = model_a.coef_[0] * X_full_a + (y_pred_b[-1] - model_a.coef_[0]*0)
        sns.lineplot(
            x=pd.to_datetime(dates_a_shifted.min()) + pd.to_timedelta(X_full_a.flatten(), unit='D'),
            y=y_pred_a.flatten(),
            label="LR After",
            linewidth=2
        )

        plt.title(label)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Count")
        plt.grid(True)
        plt.legend()
        plt.show()

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

        # Convert to datetime and floor to day
        df_b["date"] = pd.to_datetime(df_b["date"])
        df_b["day"] = df_b["date"].dt.floor("D")

        df_a["date"] = pd.to_datetime(df_a["date"])
        df_a["day"] = df_a["date"].dt.floor("D")

        # Limit to last 3 years of before data
        before_end = df_b["date"].max()
        cutoff_date = before_end - pd.DateOffset(years=3)
        df_b = df_b[df_b["date"] >= cutoff_date]
        if df_b.empty:
            continue

        # Build cumulative data
        cum_before = build_daily_cumulative(df_b)
        cum_after = build_daily_cumulative(df_a)

        results.append(calculate_stats(cum_before, cum_after, metric))

    results_df = pd.DataFrame(results)

    print("\n================ METRIC VELOCITY (Last 3 Years) =================\n")
    pd.options.display.max_columns = 10
    pd.options.display.width = 1000
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
