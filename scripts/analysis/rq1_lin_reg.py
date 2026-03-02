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


# ===============================
# NORMALIZED DAILY CUMULATIVE
# ===============================
def build_daily_cumulative_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a cumulative time series of MEAN events per repo per day.
    This removes bias due to different repo counts across periods.
    """
    df = df.copy()

    # Count events per repo per day
    daily_repo = (
        df.groupby(["repo", "day"])
          .size()
          .reset_index(name="count")
    )

    # Average across repos for each day
    daily = (
        daily_repo
        .groupby("day")["count"]
        .mean()
        .reset_index()
        .sort_values("day")
    )

    daily["t"] = np.arange(len(daily))
    daily["cumulative"] = daily["count"].cumsum()

    return daily


# ===============================
# STATISTICS + PLOTTING
# ===============================
def calculate_stats(df_before, df_after, metric, group, plot=True):
    def fit_model(df):
        X = df[["t"]].values
        y = df["cumulative"].values
        model = LinearRegression().fit(X, y)
        return model, X, y

    model_b, _, y_b = fit_model(df_before)
    model_a, _, y_a = fit_model(df_after)

    slope_b = model_b.coef_[0]
    slope_a = model_a.coef_[0]

    group1 = df_before["count"].values
    group2 = df_after["count"].values

    if len(group1) < 3 or len(group2) < 3:
        return {
            "Metric": metric,
            "Group": group,
            "Error": "insufficient_data"
        }

    # Mann–Whitney U
    _, p_value = stats.mannwhitneyu(
        group2, group1, alternative="two-sided"
    )

    # Cohen's d
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled = np.sqrt(
        ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    )
    cohens_d = (
        (np.mean(group2) - np.mean(group1)) / pooled
        if pooled > 0 else 0
    )

    if plot:
        plt.figure(figsize=(10, 5))

        after_shift = y_b[-1]
        y_a_shifted = y_a + after_shift

        sns.lineplot(x=df_before["day"], y=y_b, label="Before")
        sns.lineplot(x=df_after["day"], y=y_a_shifted, label=f"After ({group})")

        plt.title(f"{metric} — normalized per repo")
        plt.xlabel("Date")
        plt.ylabel("Cumulative mean events per repo")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "Metric": metric,
        "Group": group,
        "Slope Before": slope_b,
        "Slope After": slope_a,
        "P-Value": p_value,
        "Effect Size (d)": cohens_d,
    }


# ===============================
# MAIN
# ===============================
def main():
    data = load_all()
    results = []

    for metric in METRICS:
        key, subtype = METRIC_MAP[metric]

        df_b = data[f"{key}_before"].copy()
        df_h = data[f"{key}_after_human"].copy()
        df_a = data[f"{key}_after_agent"].copy()

        if subtype:
            df_b = df_b[df_b["activity_type"] == subtype]
            df_h = df_h[df_h["activity_type"] == subtype]
            df_a = df_a[df_a["activity_type"] == subtype]

        for df in (df_b, df_h, df_a):
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["day"] = df["date"].dt.floor("D")

        # Limit BEFORE to last 3 years
        cutoff = df_b["date"].max() - pd.DateOffset(years=3)
        df_b = df_b[df_b["date"] >= cutoff]

        if df_b.empty:
            continue

        # Build normalized cumulative series
        cum_before = build_daily_cumulative_normalized(df_b)

        if not df_h.empty:
            cum_h = build_daily_cumulative_normalized(df_h)
            results.append(
                calculate_stats(cum_before, cum_h, metric, "human")
            )

        if not df_a.empty:
            cum_a = build_daily_cumulative_normalized(df_a)
            results.append(
                calculate_stats(cum_before, cum_a, metric, "agent")
            )

    results_df = pd.DataFrame(results)

    print("\n================ NORMALIZED METRIC VELOCITY =================\n")
    pd.options.display.width = 120
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()