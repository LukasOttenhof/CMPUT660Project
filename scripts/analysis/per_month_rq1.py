# scripts/analysis/analyze_rq1.py

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

# ---------------------------------------------------------------------
# Path setup so we can import sibling modules when run as a script
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT / "outputs" / "rq1_per_month" / "tables"
PLOTS_DIR = ROOT / "outputs" / "rq1_per_month" / "plots"

from data_loader import load_all
from table_utils import save_table
from plot_utils import monthly_or_quarterly_boxplot, stacked_activity_share_bar


# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------
def group_by_month(df, date_col="date", group_col="repo"):
    if df.empty:
        return pd.DataFrame()
    df[date_col] = pd.to_datetime(df[date_col])
    df["month"] = df[date_col].dt.to_period("M").astype(str)
    return df



def per_month_stats(counts_df, value_col="count"):
    """Given a df with month and repo columns, compute summary stats per month."""
    if counts_df.empty:
        return pd.DataFrame()
    summary = counts_df.groupby("month")[value_col].agg(
        n_repos="count",
        mean="mean",
        median="median",
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75),
        variance="var",
        std="std",
    ).reset_index()
    return summary


def compute_counts(df, group_col="repo", count_type="events"):
    """Return per-repo counts depending on type."""
    if df.empty:
        return pd.DataFrame(columns=[group_col, "count", "month"])
    
    df = group_by_month(df)
    if count_type == "commits":
        counts = df.groupby(["month", group_col]).size().reset_index(name="count")
    elif count_type == "prs":
        counts = df.groupby(["month", group_col])["pr_number"].nunique().reset_index(name="count")
    elif count_type == "reviews":
        counts = df.groupby(["month", group_col])["review_id"].size().reset_index(name="count")
    elif count_type == "issues":
        opened = df[df["activity_type"] == "issue_opened"]
        counts = opened.groupby(["month", group_col])["issue_number"].nunique().reset_index(name="count")
    return counts


def text_length_per_month(df, text_col="text"):
    if df.empty or text_col not in df.columns:
        return pd.DataFrame()
    df = group_by_month(df)
    df["count"] = df[text_col].fillna("").astype(str).str.split().str.len()
    counts = df.groupby(["month", "repo"])["count"].sum().reset_index()
    return counts


# ----------------------------------------------------------
# Main analysis
# ----------------------------------------------------------
def main():
    data = load_all()

    # Split before/after
    commits_b, commits_a = data["commits_before"], data["commits_after"]
    prs_bodies_b, prs_bodies_a = data["pr_bodies_before"], data["pr_bodies_after"]
    reviews_b, reviews_a = data["reviews_before"], data["reviews_after"]
    issues_b, issues_a = data["issues_before"], data["issues_after"]
    issue_bodies_b, issue_bodies_a = data["issue_bodies_before"], data["issue_bodies_after"]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("commits", commits_b, commits_a),
        ("prs", prs_bodies_b, prs_bodies_a),
        ("reviews", reviews_b, reviews_a),
        ("issues", issues_b, issues_a),
        ("pr_text_length", prs_bodies_b, prs_bodies_a),
        ("issue_text_length", issue_bodies_b, issue_bodies_a),
    ]

    for metric, df_b, df_a in metrics:
        # Compute counts / lengths per repo per month
        if metric in ["pr_text_length"]:
            counts_b = text_length_per_month(df_b, text_col="text")
            counts_a = text_length_per_month(df_a, text_col="text")
        elif metric in ["issue_text_length"]:
            counts_b = text_length_per_month(df_b, text_col="text")
            counts_a = text_length_per_month(df_a, text_col="text")
        else:
            counts_b = compute_counts(df_b, count_type=metric)
            counts_a = compute_counts(df_a, count_type=metric)

        # Compute monthly statistics
        stats_b = per_month_stats(counts_b)
        stats_a = per_month_stats(counts_a)

        # Combine before/after into one table
        combined = stats_b.merge(stats_a, on="month", suffixes=("_before", "_after"), how="outer")
        combined = combined.sort_values("month").reset_index(drop=True)

        # Save table
        save_table(combined, f"rq1_{metric}_per_month", TABLES_DIR)

    print("[rq1] Monthly tables saved.")

    # ----------------------------------------------------------
    # Optional: generate plots (monthly/quarterly boxplots)
    # ----------------------------------------------------------
    for freq, tag in [("M", "monthly"), ("Q", "quarterly")]:
        for df_b, df_a, name, title_prefix in [
            (commits_b, commits_a, "commits", "Commits"),
            (prs_bodies_b, prs_bodies_a, "prs", "PRs"),
            (reviews_b, reviews_a, "reviews", "Reviews"),
            (issues_b, issues_a, "issues", "Issue events"),
        ]:
            monthly_or_quarterly_boxplot(
                df_b,
                df_a,
                date_col="date",
                group_col="repo",
                title=f"{title_prefix} per repository per {tag} (before vs. after agents)",
                outdir=PLOTS_DIR,
                filename=f"rq1_{name}_{tag}_boxplot.png",
                freq=freq,
            )

    # ----------------------------------------------------------
    # Stacked activity share (total counts)
    # ----------------------------------------------------------
    def count_events(df, event_type=None):
        if df.empty:
            return 0
        if event_type == "pr_created":
            return len(df[df["activity_type"] == "pr_created"])
        elif event_type == "issue_opened":
            return len(df[df["activity_type"] == "issue_opened"])
        else:
            return len(df)

    counts_before = {
        "commits": count_events(commits_b),
        "pull_requests": count_events(prs_bodies_b, "pr_created"),
        "reviews": count_events(reviews_b),
        "issues": count_events(issues_b, "issue_opened"),
    }
    counts_after = {
        "commits": count_events(commits_a),
        "pull_requests": count_events(prs_bodies_a, "pr_created"),
        "reviews": count_events(reviews_a),
        "issues": count_events(issues_a, "issue_opened"),
    }

    stacked_activity_share_bar(
        counts_before,
        counts_after,
        outdir=PLOTS_DIR,
        filename="rq1_activity_share.png",
        title="Activity Share Before/After",
    )

    print("[rq1] Analysis complete.")


if __name__ == "__main__":
    main()
