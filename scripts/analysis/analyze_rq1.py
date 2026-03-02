from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT / "outputs" / "rq1" / "tables"
PLOTS_DIR = ROOT / "outputs" / "rq1" / "plots"

from data_loader import load_all
from table_utils import summarize_before_after, save_table
from plot_utils import monthly_or_quarterly_boxplot, stacked_activity_share_bar

# ===============================
# DATA PROCESSING HELPERS
# ===============================

def process_and_combine(data, key_base):
    """
    Loads 'before', 'after_human', and 'after_agent'.
    Combines 'after' sets into one and keeps 'before' in its entirety.
    """
    df_b = data[f"{key_base}_before"].copy()
    df_h = data[f"{key_base}_after_human"].copy()
    df_a = data[f"{key_base}_after_agent"].copy()

    # Combine Human and Agent periods into a single "After" group
    df_after_total = pd.concat([df_h, df_a], ignore_index=True)

    # Standardize dates for both dataframes
    for df in [df_b, df_after_total]:
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)

    return df_b, df_after_total

# ===============================
# METRIC AGGREGATIONS
# ===============================

def commits_per_repo(df):
    return df.groupby("repo").size() if not df.empty else pd.Series(dtype=int)

def prs_per_repo(df_bodies):
    return df_bodies.groupby("repo")["pr_number"].nunique() if not df_bodies.empty else pd.Series(dtype=int)

def reviews_per_pr(df_reviews):
    return df_reviews.groupby(["repo", "pr_number"]).size() if not df_reviews.empty else pd.Series(dtype=int)

def issues_per_repo(df_issues):
    if df_issues.empty:
        return pd.Series(dtype=int)
    opened = df_issues[df_issues["activity_type"] == "issue_opened"]
    return opened.groupby("repo")["issue_number"].nunique()

def main():
    data = load_all()
    
    # 1. Load and process all data pairs (Full totals for 'Before')
    commits_b, commits_a = process_and_combine(data, "commits")
    prs_bodies_b, prs_bodies_a = process_and_combine(data, "pr_bodies")
    prs_events_b, prs_events_a = process_and_combine(data, "pull_requests")
    reviews_b, reviews_a = process_and_combine(data, "reviews")
    issues_b, issues_a = process_and_combine(data, "issues")
    issue_bodies_b, issue_bodies_a = process_and_combine(data, "issue_bodies")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Table: PR body length ---
    pr_text_len_b = prs_bodies_b["text"].fillna("").astype(str).str.split().str.len()
    pr_text_len_a = prs_bodies_a["text"].fillna("").astype(str).str.split().str.len()
    save_table(summarize_before_after(pr_text_len_b, pr_text_len_a), "rq1_pr_text_length", TABLES_DIR)

    # --- Table: Issue body length ---
    def get_text_len(df):
        return df["text"].fillna("").astype(str).str.split().str.len() if not df.empty else []
    
    save_table(summarize_before_after(get_text_len(issue_bodies_b), get_text_len(issue_bodies_a)), 
               "rq1_issue_text_length", TABLES_DIR)

    # --- Tables: Metrics per repo ---
    save_table(summarize_before_after(commits_per_repo(commits_b), commits_per_repo(commits_a)), 
               "rq1_commits_per_repo", TABLES_DIR)
    
    save_table(summarize_before_after(prs_per_repo(prs_bodies_b), prs_per_repo(prs_bodies_a)), 
               "rq1_prs_per_repo", TABLES_DIR)
    
    save_table(summarize_before_after(reviews_per_pr(reviews_b), reviews_per_pr(reviews_a)), 
               "rq1_reviews_per_pr", TABLES_DIR)
    
    save_table(summarize_before_after(issues_per_repo(issues_b), issues_per_repo(issues_a)), 
               "rq1_issues_per_repo", TABLES_DIR)

    # --- Raw Repo-level CSV ---
    commits_pb, commits_pa = commits_per_repo(commits_b), commits_per_repo(commits_a)
    prs_pb, prs_pa = prs_per_repo(prs_bodies_b), prs_per_repo(prs_bodies_a)
    issues_pb, issues_pa = issues_per_repo(issues_b), issues_per_repo(issues_a)

    all_repos = sorted(set(commits_pb.index) | set(commits_pa.index) | 
                       set(prs_pb.index) | set(prs_pa.index) | 
                       set(issues_pb.index) | set(issues_pa.index))
    
    per_repo_df = pd.DataFrame({"repo": all_repos})
    mapping = {
        "commits_before": commits_pb, "commits_after": commits_pa,
        "prs_before": prs_pb, "prs_after": prs_pa,
        "issues_before": issues_pb, "issues_after": issues_pa
    }
    for col, source in mapping.items():
        per_repo_df[col] = per_repo_df["repo"].map(source).fillna(0).astype(int)
    
    per_repo_df.to_csv(TABLES_DIR / "rq1_repo_level_counts_raw.csv", index=False)

    # --- Boxplots ---
    plot_configs = [
        (commits_b, commits_a, "Commits", "rq1_commits"),
        (prs_bodies_b, prs_bodies_a, "PRs", "rq1_prs"),
        (reviews_b, reviews_a, "Reviews", "rq1_reviews"),
        (issues_b, issues_a, "Issue events", "rq1_issues"),
    ]

    for freq, tag in [("M", "monthly"), ("Q", "quarterly")]:
        for df_b, df_a, label, fname in plot_configs:
            monthly_or_quarterly_boxplot(
                df_b, df_a, date_col="date", group_col="repo",
                title=f"{label} per repository per {tag} (Full Before vs. Total After)",
                outdir=PLOTS_DIR, filename=f"{fname}_{tag}_boxplot.png", freq=freq,
            )

    # --- Activity Share ---
    def count_type(df, subtype=None):
        if df.empty: return 0
        if subtype:
            return len(df[df["activity_type"] == subtype])
        return len(df)

    counts_before = {
        "commits": count_type(commits_b),
        "pull_requests": count_type(prs_events_b, "pr_created"),
        "reviews": count_type(reviews_b),
        "issues": count_type(issues_b, "issue_opened"),
    }
    counts_after = {
        "commits": count_type(commits_a),
        "pull_requests": count_type(prs_events_a, "pr_created"),
        "reviews": count_type(reviews_a),
        "issues": count_type(issues_a, "issue_opened"),
    }

    stacked_activity_share_bar(
        counts_before, counts_after, outdir=PLOTS_DIR,
        filename="rq1_activity_share.png", title="Activity Share (Full Totals)"
    )

    print("[rq1] Summary statistics generated using total historical data.")

if __name__ == "__main__":
    main()