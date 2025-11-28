# scripts/analysis/analyze_rq2.py

from __future__ import annotations

from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT / "outputs" / "rq2" / "tables"
PLOTS_DIR = ROOT / "outputs" / "rq2" / "plots"

from data_loader import load_all
from table_utils import summarize_before_after, save_table
from plot_utils import monthly_or_quarterly_boxplot

import pandas as pd


def build_dev_activity_df(
    commits: pd.DataFrame,
    prs_events: pd.DataFrame,
    reviews: pd.DataFrame,
    review_comments: pd.DataFrame,
    issues: pd.DataFrame,
    period_label: str,
) -> pd.DataFrame:
    """
    Build per-developer activity table WITH derived rate metrics.
    Unlike raw totals, these rates normalize behavior beyond time frame length.
    """
    authors = set()

    for df in (commits, prs_events, reviews, review_comments, issues):
        if not df.empty and "author" in df.columns:
            authors.update(df["author"].dropna().unique().tolist())

    authors = sorted(authors)
    out = pd.DataFrame({"author": authors})

    # Helper for zero-safe division
    def safe_div(num, den):
        return num / den if den != 0 else 0

    # -------------------------------------------------------------
    # RAW COUNTS
    # -------------------------------------------------------------
    # Commits + LOC
    if not commits.empty:
        c_group = commits.groupby("author")
        out["commits"] = out["author"].map(c_group.size()).fillna(0).astype(int)
        out["loc_added"] = out["author"].map(c_group["loc_added"].sum()).fillna(0).astype(int)
        out["loc_deleted"] = out["author"].map(c_group["loc_deleted"].sum()).fillna(0).astype(int)
    else:
        out["commits"] = 0
        out["loc_added"] = 0
        out["loc_deleted"] = 0

    # PR events
    if not prs_events.empty:
        pr_created = prs_events[prs_events["activity_type"] == "pr_created"]
        pr_merged = prs_events[prs_events["activity_type"] == "pr_merged"]
        out["prs_created"] = out["author"].map(pr_created.groupby("author").size()).fillna(0).astype(int)
        out["prs_merged"] = out["author"].map(pr_merged.groupby("author").size()).fillna(0).astype(int)
    else:
        out["prs_created"] = 0
        out["prs_merged"] = 0

    # Reviews submitted
    if not reviews.empty:
        out["reviews_submitted"] = out["author"].map(
            reviews.groupby("author").size()
        ).fillna(0).astype(int)
    else:
        out["reviews_submitted"] = 0

    # Review comments
    if not review_comments.empty:
        out["review_comments"] = out["author"].map(
            review_comments.groupby("author").size()
        ).fillna(0).astype(int)
    else:
        out["review_comments"] = 0

    # Issues opened / closed
    if not issues.empty:
        opened = issues[issues["activity_type"] == "issue_opened"]
        closed = issues[issues["activity_type"] == "issue_closed"]
        out["issues_opened"] = out["author"].map(
            opened.groupby("author").size()
        ).fillna(0).astype(int)
        out["issues_closed"] = out["author"].map(
            closed.groupby("author").size()
        ).fillna(0).astype(int)
    else:
        out["issues_opened"] = 0
        out["issues_closed"] = 0

    # -------------------------------------------------------------
    # DERIVED RATES (per developer)
    # -------------------------------------------------------------
    out["reviews_per_pr"] = [
        safe_div(row.reviews_submitted, row.prs_created) for row in out.itertuples()
    ]

    out["review_comment_rate"] = [
        safe_div(row.review_comments, row.reviews_submitted) for row in out.itertuples()
    ]

    out["issues_per_commit"] = [
        safe_div(row.issues_opened, row.commits) for row in out.itertuples()
    ]

    out["issue_close_rate"] = [
        safe_div(row.issues_closed, row.issues_opened) for row in out.itertuples()
    ]

    out["pr_merge_rate"] = [
        safe_div(row.prs_merged, row.prs_created) for row in out.itertuples()
    ]

    out["period"] = period_label
    return out


def main():
    data = load_all()

    commits_b = data["commits_before"]
    commits_a = data["commits_after"]
    prs_events_b = data["pull_requests_before"]
    prs_events_a = data["pull_requests_after"]
    reviews_b = data["reviews_before"]
    reviews_a = data["reviews_after"]
    review_comments_b = data["review_comments_before"]
    review_comments_a = data["review_comments_after"]
    issues_b = data["issues_before"]
    issues_a = data["issues_after"]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Developer-level aggregated activity
    # ----------------------------------------------------------
    dev_b = build_dev_activity_df(
        commits_b, prs_events_b, reviews_b, review_comments_b, issues_b, "before"
    )
    dev_a = build_dev_activity_df(
        commits_a, prs_events_a, reviews_a, review_comments_a, issues_a, "after"
    )

        # ----------------------------------------------------------
# PER-REPOSITORY RATES
# ----------------------------------------------------------
    def build_repo_rates(commits, prs_events, reviews, review_comments, issues, period):
        repos = set()
        for df in (commits, prs_events, reviews, review_comments, issues):
            if not df.empty:
                repos.update(df["repo"].dropna().unique())

        out = pd.DataFrame({"repo": sorted(repos)})

        # Helper
        def safe_div(n, d):
            return n / d if d != 0 else 0

        # Commits
        c = commits.groupby("repo").size() if not commits.empty else {}
        out["commits"] = out["repo"].map(c).fillna(0).astype(int)

        # PRs
        if not prs_events.empty:
            created = prs_events[prs_events["activity_type"] == "pr_created"]
            merged = prs_events[prs_events["activity_type"] == "pr_merged"]
            pr_c = created.groupby("repo").size()
            pr_m = merged.groupby("repo").size()
        else:
            pr_c = {}; pr_m = {}

        out["prs_created"] = out["repo"].map(pr_c).fillna(0).astype(int)
        out["prs_merged"] = out["repo"].map(pr_m).fillna(0).astype(int)

        # Reviews + review comments
        rev = reviews.groupby("repo").size() if not reviews.empty else {}
        rcom = review_comments.groupby("repo").size() if not review_comments.empty else {}

        out["reviews_submitted"] = out["repo"].map(rev).fillna(0).astype(int)
        out["review_comments"] = out["repo"].map(rcom).fillna(0).astype(int)

        # Issues opened/closed
        if not issues.empty:
            opened = issues[issues["activity_type"] == "issue_opened"].groupby("repo").size()
            closed = issues[issues["activity_type"] == "issue_closed"].groupby("repo").size()
        else:
            opened = {}; closed = {}

        out["issues_opened"] = out["repo"].map(opened).fillna(0).astype(int)
        out["issues_closed"] = out["repo"].map(closed).fillna(0).astype(int)

        # Rates per repo
        out["reviews_per_pr"] = [
            safe_div(r, p) for r, p in zip(out["reviews_submitted"], out["prs_created"])
        ]
        out["issues_per_commit"] = [
            safe_div(i, c) for i, c in zip(out["issues_opened"], out["commits"])
        ]
        out["issue_close_rate"] = [
            safe_div(c, o) for c, o in zip(out["issues_closed"], out["issues_opened"])
        ]
        out["pr_merge_rate"] = [
            safe_div(m, p) for m, p in zip(out["prs_merged"], out["prs_created"])
        ]
        out["review_comment_rate"] = [
            safe_div(c, r) for c, r in zip(out["review_comments"], out["reviews_submitted"])
        ]

        out["period"] = period
        return out


    repo_b = build_repo_rates(commits_b, prs_events_b, reviews_b, review_comments_b, issues_b, "before")
    repo_a = build_repo_rates(commits_a, prs_events_a, reviews_a, review_comments_a, issues_a, "after")

    repo_all = pd.concat([repo_b, repo_a], ignore_index=True)
    repo_all.to_csv(TABLES_DIR / "rq2_repo_rates_raw.csv", index=False)
    print("[rq2] Saved repo-level rates data.")


    dev_all = pd.concat([dev_b, dev_a], ignore_index=True)
    dev_all.to_csv(TABLES_DIR / "rq2_dev_activity_raw.csv", index=False)
    print("[rq2] Saved developer-level raw activity table.")

    # Summaries on key metrics
    metrics = [
        "commits",
        "prs_created",
        "prs_merged",
        "reviews_submitted",
        "review_comments",
        "issues_opened",
        "issues_closed",
    ]
    for col in metrics:
        summary = summarize_before_after(dev_b[col], dev_a[col])
        save_table(summary, f"rq2_dev_{col}", TABLES_DIR)
    
    # ----------------------------------------------------------
    # Review latency (time_to_merge_hours)
    # ----------------------------------------------------------
    def extract_merge_times(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        merged = df[df["activity_type"] == "pr_merged"]
        merged = merged.dropna(subset=["time_to_merge_hours"])
        return merged["time_to_merge_hours"].astype(float)

    merge_before = extract_merge_times(prs_events_b)
    merge_after = extract_merge_times(prs_events_a)

    merge_summary = summarize_before_after(merge_before, merge_after)
    save_table(merge_summary, "rq2_merge_time_hours", TABLES_DIR)

    # per-PR merge times
    def per_pr_merge_df(df: pd.DataFrame, period: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=["repo", "pr_number", "date", "time_to_merge_hours", "period"]
            )
        merged = df[df["activity_type"] == "pr_merged"].dropna(
            subset=["time_to_merge_hours"]
        )
        out = merged[["repo", "pr_number", "date", "time_to_merge_hours"]].copy()
        out["time_to_merge_hours"] = out["time_to_merge_hours"].astype(float)
        out["period"] = period
        return out

    per_pr_b = per_pr_merge_df(prs_events_b, "before")
    per_pr_a = per_pr_merge_df(prs_events_a, "after")
    per_pr_all = pd.concat([per_pr_b, per_pr_a], ignore_index=True)
    per_pr_all.to_csv(TABLES_DIR / "rq2_per_pr_merge_times.csv", index=False)
    print("[rq2] Saved per-PR merge-time data.")

    # ----------------------------------------------------------
    # Review workload: reviews per PR and per reviewer
    # ----------------------------------------------------------
    def reviews_per_pr(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        return df.groupby(["repo", "pr_number"]).size()

    rpp_b = reviews_per_pr(reviews_b)
    rpp_a = reviews_per_pr(reviews_a)
    rpp_all = pd.concat(
        [
            rpp_b.rename("before").reset_index(),
            rpp_a.rename("after").reset_index(),
        ],
        axis=0,
        ignore_index=True,
    )
    rpp_all.to_csv(TABLES_DIR / "rq2_reviews_per_pr_raw.csv", index=False)

    rpp_summary = summarize_before_after(rpp_b, rpp_a)
    save_table(rpp_summary, "rq2_reviews_per_pr", TABLES_DIR)

    def reviews_per_reviewer(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        return df.groupby("author").size()

    rpr_b = reviews_per_reviewer(reviews_b)
    rpr_a = reviews_per_reviewer(reviews_a)
    save_table(
        summarize_before_after(rpr_b, rpr_a),
        "rq2_reviews_per_reviewer",
        TABLES_DIR,
    )

    # Summaries of derived metrics
    rate_metrics = [
        "reviews_per_pr",
        "issues_per_commit",
        "issue_close_rate",
        "pr_merge_rate",
        "review_comment_rate",
    ]

    for col in rate_metrics:
        save_table(
            summarize_before_after(dev_b[col], dev_a[col]),
            f"rq2_dev_rate_{col}",
            TABLES_DIR,
        )
        save_table(
            summarize_before_after(repo_b[col], repo_a[col]),
            f"rq2_repo_rate_{col}",
            TABLES_DIR,
        )


    # ----------------------------------------------------------
    # Time-series boxplots of review workload per repo
    # ----------------------------------------------------------
    for freq, tag in [("M", "monthly"), ("Q", "quarterly")]:
        monthly_or_quarterly_boxplot(
            reviews_b,
            reviews_a,
            date_col="date",
            group_col="repo",
            title=f"Reviews per repository per {tag} (before vs. after agents)",
            outdir=PLOTS_DIR,
            filename=f"rq2_reviews_{tag}_boxplot.png",
            freq=freq,
        )

    print("[rq2] Analysis complete.")


if __name__ == "__main__":
    main()
