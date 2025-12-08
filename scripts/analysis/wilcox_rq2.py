from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_loader import load_all
import seaborn as sns
from scipy.stats import chi2_contingency

pd.options.mode.chained_assignment = None

ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT / "outputs" / "pokemon" / "plots"
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


def build_monthly_dev_counts(data):
    """
    Calculate total counts of activities per metric for before and after periods.
    Returns two dicts: before_counts, after_counts
    """
    before_counts = {}
    after_counts = {}

    for metric in METRICS:
        key, subtype = METRIC_MAP[metric]
        df_b = data[f"{key}_before"].copy()
        df_a = data[f"{key}_after"].copy()

        if subtype:
            df_b = df_b[df_b["activity_type"] == subtype]
            df_a = df_a[df_a["activity_type"] == subtype]

        if 'author' not in df_b.columns or 'author' not in df_a.columns:
            print(f"Warning: Skipping {metric} - 'author' column not found.")
            continue

        before_counts[metric] = len(df_b)
        after_counts[metric] = len(df_a)

    return before_counts, after_counts


def run_chi_square_test(before_counts, after_counts):
    """
    Runs Chi-square test of independence on a 2x7 table of counts.
    """
    metrics = list(before_counts.keys())

    before_list = [before_counts[m] for m in metrics]
    after_list = [after_counts[m] for m in metrics]

    contingency = np.array([before_list, after_list])

    chi2, p, dof, expected = chi2_contingency(contingency)

    print("\n================ CHI-SQUARED TEST ON COUNTS ================\n")
    print(f"Chi2 Statistic: {chi2}")
    print(f"P-value: {p}")
    print(f"Degrees of Freedom: {dof}")
    print("Expected counts:")
    print(pd.DataFrame(expected, columns=metrics, index=["Before", "After"]))

    return chi2, p, dof, expected


def main():
    data = load_all()

    # Apply 3-year filter for 'before' datasets
    for key in data:
        if key.endswith("_before"):
            df = data[key]
            if df.empty or "date" not in df.columns:
                continue
            df["date"] = pd.to_datetime(df["date"])
            before_end = df["date"].max()
            cutoff_date = before_end - pd.DateOffset(years=3)
            data[key] = df[df["date"] >= cutoff_date]

    # Get raw counts
    before_counts, after_counts = build_monthly_dev_counts(data)

    # Print counts
    print("\n================ RAW ACTIVITY COUNTS ================\n")
    count_df = pd.DataFrame({
        "Metric": list(before_counts.keys()),
        "Before Count": list(before_counts.values()),
        "After Count": list(after_counts.values())
    })
    print(count_df.to_string(index=False))

    # Run Chi-square test
    run_chi_square_test(before_counts, after_counts)


if __name__ == "__main__":
    main()
