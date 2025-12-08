from __future__ import annotations

from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT / "outputs" / "rq1" / "tables"
PLOTS_DIR = ROOT / "outputs" / "rq1" / "plots"

from data_loader import load_all
from table_utils import summarize_before_after, save_table
from plot_utils import monthly_or_quarterly_boxplot, stacked_activity_share_bar


def commits_per_repo(df):
    if df.empty:
        return df.groupby("repo").size()
    return df.groupby("repo").size()


def prs_per_repo(df_bodies):
    if df_bodies.empty:
        return df_bodies.groupby("repo")["pr_number"].nunique()
    return df_bodies.groupby("repo")["pr_number"].nunique()


def reviews_per_pr(df_reviews):
    if df_reviews.empty:
        return df_reviews.groupby(["repo", "pr_number"]).size()
    return df_reviews.groupby(["repo", "pr_number"]).size()


def issues_per_repo(df_issues):
    if df_issues.empty:
        return df_issues.groupby("repo")["issue_number"].nunique()
    opened = df_issues[df_issues["activity_type"] == "issue_opened"]
    if opened.empty:
        return opened.groupby("repo")["issue_number"].nunique()
    return opened.groupby("repo")["issue_number"].nunique()


def main():
    data = load_all()

    commits_b = data["commits_before"]
    commits_a = data["commits_after"]
    prs_bodies_b = data["pr_bodies_before"]
    prs_bodies_a = data["pr_bodies_after"]
    prs_events_b = data["pull_requests_before"]
    prs_events_a = data["pull_requests_after"]
    reviews_b = data["reviews_before"]
    reviews_a = data["reviews_after"]
    issues_b = data["issues_before"]
    issues_a = data["issues_after"]
    issue_bodies_b = data["issue_bodies_before"]
    issue_bodies_a = data["issue_bodies_after"]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    #PR body by length
    pr_text_len_b = prs_bodies_b["text"].fillna("").astype(str).str.split().str.len()
    pr_text_len_a = prs_bodies_a["text"].fillna("").astype(str).str.split().str.len()
    pr_text_table = summarize_before_after(pr_text_len_b, pr_text_len_a)
    save_table(pr_text_table, "rq1_pr_text_length", TABLES_DIR)

    #Issue body text length
    def issue_text_len(df):
        if df.empty or "text" not in df.columns:
            return []
        return df["text"].fillna("").astype(str).str.split().str.len()

    issue_text_len_b = issue_text_len(issue_bodies_b)
    issue_text_len_a = issue_text_len(issue_bodies_a)
    issue_text_table = summarize_before_after(issue_text_len_b, issue_text_len_a)
    save_table(issue_text_table, "rq1_issue_text_length", TABLES_DIR)

    #Metrics per repo
    commits_per_repo_b = commits_per_repo(commits_b)
    commits_per_repo_a = commits_per_repo(commits_a)
    save_table(
        summarize_before_after(commits_per_repo_b, commits_per_repo_a),
        "rq1_commits_per_repo",
        TABLES_DIR,
    )

    prs_per_repo_b = prs_per_repo(prs_bodies_b)
    prs_per_repo_a = prs_per_repo(prs_bodies_a)
    save_table(
        summarize_before_after(prs_per_repo_b, prs_per_repo_a),
        "rq1_prs_per_repo",
        TABLES_DIR,
    )

    reviews_per_pr_b = reviews_per_pr(reviews_b)
    reviews_per_pr_a = reviews_per_pr(reviews_a)
    save_table(
        summarize_before_after(reviews_per_pr_b, reviews_per_pr_a),
        "rq1_reviews_per_pr",
        TABLES_DIR,
    )

    issues_per_repo_b = issues_per_repo(issues_b)
    issues_per_repo_a = issues_per_repo(issues_a)
    save_table(
        summarize_before_after(issues_per_repo_b, issues_per_repo_a),
        "rq1_issues_per_repo",
        TABLES_DIR,
    )

    all_repos = sorted(
        set(commits_per_repo_b.index)
        | set(commits_per_repo_a.index)
        | set(prs_per_repo_b.index)
        | set(prs_per_repo_a.index)
        | set(issues_per_repo_b.index)
        | set(issues_per_repo_a.index)
    )
    per_repo_df = pd.DataFrame({"repo": all_repos})
    per_repo_df["commits_before"] = per_repo_df["repo"].map(commits_per_repo_b).fillna(0).astype(int)
    per_repo_df["commits_after"] = per_repo_df["repo"].map(commits_per_repo_a).fillna(0).astype(int)
    per_repo_df["prs_before"] = per_repo_df["repo"].map(prs_per_repo_b).fillna(0).astype(int)
    per_repo_df["prs_after"] = per_repo_df["repo"].map(prs_per_repo_a).fillna(0).astype(int)
    per_repo_df["issues_before"] = per_repo_df["repo"].map(issues_per_repo_b).fillna(0).astype(int)
    per_repo_df["issues_after"] = per_repo_df["repo"].map(issues_per_repo_a).fillna(0).astype(int)
    per_repo_df.to_csv(TABLES_DIR / "rq1_repo_level_counts_raw.csv", index=False)
    print("[rq1] Saved repo-level raw counts.")

    #Boxplots over time
    for freq, tag in [("M", "monthly"), ("Q", "quarterly")]:
        monthly_or_quarterly_boxplot(
            commits_b,
            commits_a,
            date_col="date",
            group_col="repo",
            title=f"Commits per repository per {tag} (before vs. after agents)",
            outdir=PLOTS_DIR,
            filename=f"rq1_commits_{tag}_boxplot.png",
            freq=freq,
        )

        monthly_or_quarterly_boxplot(
            prs_bodies_b,
            prs_bodies_a,
            date_col="date",
            group_col="repo",
            title=f"PRs per repository per {tag} (before vs. after agents)",
            outdir=PLOTS_DIR,
            filename=f"rq1_prs_{tag}_boxplot.png",
            freq=freq,
        )

        monthly_or_quarterly_boxplot(
            reviews_b,
            reviews_a,
            date_col="date",
            group_col="repo",
            title=f"Reviews per repository per {tag} (before vs. after agents)",
            outdir=PLOTS_DIR,
            filename=f"rq1_reviews_{tag}_boxplot.png",
            freq=freq,
        )

        monthly_or_quarterly_boxplot(
            issues_b,
            issues_a,
            date_col="date",
            group_col="repo",
            title=f"Issue events per repository per {tag} (before vs. after agents)",
            outdir=PLOTS_DIR,
            filename=f"rq1_issues_{tag}_boxplot.png",
            freq=freq,
        )

    #Activit share stacked bars
    def count_commits(df):
        return len(df)

    def count_pr_creations(df):
        if df.empty:
            return 0
        created = df[df["activity_type"] == "pr_created"]
        return len(created)

    def count_reviews(df):
        return len(df)

    def count_issue_openings(df):
        if df.empty:
            return 0
        opened = df[df["activity_type"] == "issue_opened"]
        return len(opened)

    counts_before = {
        "commits": count_commits(commits_b),
        "pull_requests": count_pr_creations(prs_events_b),
        "reviews": count_reviews(reviews_b),
        "issues": count_issue_openings(issues_b),
    }
    counts_after = {
        "commits": count_commits(commits_a),
        "pull_requests": count_pr_creations(prs_events_a),
        "reviews": count_reviews(reviews_a),
        "issues": count_issue_openings(issues_a),
    }

    stacked_activity_share_bar(
        counts_before,
        counts_after,
        outdir=PLOTS_DIR,
        filename="rq1_activity_share.png",
        title="Activity Share Before/After"
    )



    print("[rq1] Analysis complete.")


if __name__ == "__main__":
    import pandas as pd

    main()
