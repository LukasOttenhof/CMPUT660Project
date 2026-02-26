from pathlib import Path
import numpy as np
import pandas as pd
from data_loader import load_all
from scipy.stats import wilcoxon, chi2_contingency

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


def build_total_counts(data):
    before_counts = {}
    after_counts = {}

    for metric in METRICS:
        key, subtype = METRIC_MAP[metric]
        df_b = data[f"{key}_before"].copy()
        df_a = data[f"{key}_after"].copy()

        if subtype:
            df_b = df_b[df_b["activity_type"] == subtype]
            df_a = df_a[df_a["activity_type"] == subtype]

        before_counts[metric] = len(df_b)
        after_counts[metric] = len(df_a)

    return before_counts, after_counts


def run_wilcoxon_tests(before_counts, after_counts):
    metrics = list(before_counts.keys())

    before = np.array([before_counts[m] for m in metrics])
    after = np.array([after_counts[m] for m in metrics])

    # -------- Full test --------
    stat_all, p_all = wilcoxon(before, after)
    d_all = cohens_d_paired(before, after)

    # -------- Reduced test --------
    exclude = {"commits", "prs_created", "prs_merged"}
    reduced_metrics = [m for m in metrics if m not in exclude]

    before_r = np.array([before_counts[m] for m in reduced_metrics])
    after_r = np.array([after_counts[m] for m in reduced_metrics])

    stat_r, p_r = wilcoxon(before_r, after_r)
    d_r = cohens_d_paired(before_r, after_r)

    print("\n================ WILCOXON SIGNED-RANK TEST ================\n")

    print("All metrics:")
    print(f"  Statistic = {stat_all}")
    print(f"  P-value   = {p_all:.4f}")
    print(f"  Cohen's d = {d_all:.3f}")

    print("\nExcluding commits + PRs:")
    print(f"  Statistic = {stat_r}")
    print(f"  P-value   = {p_r:.4f}")
    print(f"  Cohen's d = {d_r:.3f}")

    return (p_all, d_all), (p_r, d_r)


def demonstrate_chi_square_underflow(before_counts, after_counts):
    metrics = list(before_counts.keys())
    contingency = np.array([
        [before_counts[m] for m in metrics],
        [after_counts[m] for m in metrics],
    ])

    chi2, p, _, _ = chi2_contingency(contingency)

    print("\n================ CHI-SQUARE (UNDERFLOW DEMO) ================\n")
    print(f"Chi2 Statistic: {chi2}")
    print(f"P-value (rounded): {p}")


def main():
    data = load_all()

    # Apply 3-year cutoff to before period
    for key in data:
        if key.endswith("_before") and "date" in data[key].columns:
            df = data[key]
            df["date"] = pd.to_datetime(df["date"])
            cutoff = df["date"].max() - pd.DateOffset(years=3)
            data[key] = df[df["date"] >= cutoff]

    before_counts, after_counts = build_total_counts(data)

    print("\n================ RAW COUNTS ================\n")
    print(pd.DataFrame({
        "Metric": METRICS,
        "Before": [before_counts[m] for m in METRICS],
        "After": [after_counts[m] for m in METRICS],
    }).to_string(index=False))

    run_wilcoxon_tests(before_counts, after_counts)
    demonstrate_chi_square_underflow(before_counts, after_counts)


def cohens_d_paired(before, after):
    """
    Cohen's d for paired samples.
    """
    diff = after - before
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    return mean_diff / std_diff if std_diff > 0 else 0

if __name__ == "__main__":
    main()