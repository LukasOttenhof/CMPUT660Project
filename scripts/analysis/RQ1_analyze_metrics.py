#!/usr/bin/env python3
"""
RQ1_behavior_metrics.py

Behavioral (normalized) metric analysis for RQ1/RQ2.

Focuses on per-unit metrics instead of raw volume:
- Commit message length
- Lines changed per commit
- Files changed per commit
- Commits per PR
- Lines changed per PR
- Reviews per PR
- Time-to-first-review
- Time-to-merge / close
- Review comment length / sentiment-ready text

Inputs:
    inputs/processed/
        commits_before.parquet
        commits_after.parquet
        pull_requests_before.parquet
        pull_requests_after.parquet
        reviews_before.parquet
        reviews_after.parquet

Outputs:
    outputs/tables/*.csv (console + CSV)
    outputs/plots/*.png  (seaborn)

Assumptions:
- Commits parquet has:
    repo, author, date, sha, message, additions, deletions, changed_files
  (fallbacks supported)
- PR parquet has at least:
    repo, author, pr_number, created_at, merged_at/closed_at,
    additions, deletions, changed_files
- Reviews parquet has:
    repo, author, pr_number (or pull_request_id), submitted_at, body, state
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set(style="whitegrid")

INPUT  = Path("inputs/processed")
TABLES = Path("outputs/tables")
PLOTS  = Path("outputs/plots")
TABLES.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name.lower())

def print_and_save(df: pd.DataFrame, name: str):
    print("\n" + "="*70)
    print(name)
    print("="*70)
    print(df)
    df.to_csv(TABLES / f"{safe_filename(name)}.csv", index=False)

def needs_log(series: pd.Series):
    s = series.dropna()
    s = s[s > 0]
    if len(s) < 2:
        return False
    return (s.max() / s.min()) > 1000

def save_boxplot(long_df, x, y, title, out_base):
    plt.figure(figsize=(7,6))
    sns.boxplot(data=long_df, x=x, y=y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(PLOTS / f"{out_base}.png", dpi=300)
    plt.close()

    if needs_log(long_df[y]):
        plt.figure(figsize=(7,6))
        sns.boxplot(data=long_df, x=x, y=y)
        plt.yscale("log")
        plt.title(title + " (log)")
        plt.tight_layout()
        plt.savefig(PLOTS / f"{out_base}_log.png", dpi=300)
        plt.close()

def save_hist(series, title, out_base):
    plt.figure(figsize=(7,5))
    sns.histplot(series.dropna(), bins=40, kde=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(PLOTS / f"{out_base}.png", dpi=300)
    plt.close()

    if needs_log(series):
        plt.figure(figsize=(7,5))
        sns.histplot(series.dropna(), bins=40, kde=True)
        plt.xscale("log")
        plt.title(title + " (log)")
        plt.tight_layout()
        plt.savefig(PLOTS / f"{out_base}_log.png", dpi=300)
        plt.close()

def load_parquet(name: str):
    df = pd.read_parquet(INPUT / name)
    # normalize datetimes
    for c in ["date", "created_at", "merged_at", "closed_at", "submitted_at"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df

def coalesce(df, target, candidates):
    """Rename first existing candidate to target."""
    for c in candidates:
        if c in df.columns:
            if c != target:
                df[target] = df[c]
            return
    df[target] = np.nan

# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------
comm_b = load_parquet("commits_before.parquet")
comm_a = load_parquet("commits_after.parquet")
prs_b  = load_parquet("pull_requests_before.parquet")
prs_a  = load_parquet("pull_requests_after.parquet")
rev_b  = load_parquet("reviews_before.parquet")
rev_a  = load_parquet("reviews_after.parquet")

# ---------------------------------------------------------------------
# Column normalization (robust to naming drift)
# ---------------------------------------------------------------------
def normalize_commits(df):
    df = df.copy()
    coalesce(df, "message", ["message", "text", "commit_message", "msg"])
    coalesce(df, "additions", ["additions", "loc_added", "insertions"])
    coalesce(df, "deletions", ["deletions", "loc_deleted"])
    coalesce(df, "changed_files", ["changed_files", "files_changed", "files"])
    return df

def normalize_prs(df):
    df = df.copy()
    coalesce(df, "pr_number", ["pr_number", "number", "pull_request_id", "pr_id"])
    coalesce(df, "created_at", ["created_at", "date_opened", "date"])
    coalesce(df, "merged_at", ["merged_at"])
    coalesce(df, "closed_at", ["closed_at"])
    coalesce(df, "additions", ["additions", "loc_added", "insertions"])
    coalesce(df, "deletions", ["deletions", "loc_deleted"])
    coalesce(df, "changed_files", ["changed_files", "files_changed", "files"])
    return df

def normalize_reviews(df):
    df = df.copy()
    coalesce(df, "pr_number", ["pr_number", "pull_request_id", "pr_id", "number"])
    coalesce(df, "submitted_at", ["submitted_at", "date", "created_at"])
    coalesce(df, "body", ["body", "text", "comment"])
    coalesce(df, "state", ["state", "review_state"])
    return df

comm_b = normalize_commits(comm_b)
comm_a = normalize_commits(comm_a)
prs_b  = normalize_prs(prs_b)
prs_a  = normalize_prs(prs_a)
rev_b  = normalize_reviews(rev_b)
rev_a  = normalize_reviews(rev_a)

# ---------------------------------------------------------------------
# 1) Commit-level behavioural metrics
# ---------------------------------------------------------------------
def commit_metrics(df):
    d = df.copy()
    d["msg_len_chars"] = d["message"].fillna("").astype(str).str.len()
    d["msg_len_words"] = d["message"].fillna("").astype(str).str.split().str.len()
    d["lines_changed"] = d["additions"].fillna(0) + d["deletions"].fillna(0)
    d["churn_ratio"]   = (d["additions"].fillna(0)+1) / (d["deletions"].fillna(0)+1)
    return d

comm_bm = commit_metrics(comm_b)
comm_am = commit_metrics(comm_a)

# ---------------------------------------------------------------------
# 2) PR-level behavioural metrics
# ---------------------------------------------------------------------
def pr_metrics(prs_df, comm_df, rev_df):
    prs = prs_df.copy()

    # PR size
    prs["pr_lines_changed"] = prs["additions"].fillna(0) + prs["deletions"].fillna(0)

    # Commits-per-PR using commit parquet
    comm_sub = comm_df.dropna(subset=["pr_number"]).copy() if "pr_number" in comm_df.columns else None
    if comm_sub is not None and len(comm_sub):
        commits_per_pr = comm_sub.groupby(["repo", "pr_number"])["sha"].nunique().reset_index(name="commits_per_pr")
        prs = prs.merge(commits_per_pr, on=["repo", "pr_number"], how="left")
    else:
        prs["commits_per_pr"] = np.nan

    # Reviews-per-PR
    rsub = rev_df.dropna(subset=["pr_number"]).copy()
    if len(rsub):
        reviews_per_pr = rsub.groupby(["repo", "pr_number"]).size().reset_index(name="reviews_per_pr")
        prs = prs.merge(reviews_per_pr, on=["repo", "pr_number"], how="left")
    else:
        prs["reviews_per_pr"] = 0

    # Time-to-first-review
    if len(rsub) and "submitted_at" in rsub.columns:
        first_review = rsub.groupby(["repo", "pr_number"])["submitted_at"].min().reset_index(name="first_review_at")
        prs = prs.merge(first_review, on=["repo", "pr_number"], how="left")
        prs["time_to_first_review_h"] = (
            (prs["first_review_at"] - prs["created_at"]).dt.total_seconds() / 3600
        )
    else:
        prs["time_to_first_review_h"] = np.nan

    # Time-to-merge/close
    end_time = prs["merged_at"].combine_first(prs["closed_at"])
    prs["time_to_resolution_h"] = (end_time - prs["created_at"]).dt.total_seconds() / 3600

    return prs

prs_bm = pr_metrics(prs_b, comm_bm, rev_b)
prs_am = pr_metrics(prs_a, comm_am, rev_a)

# ---------------------------------------------------------------------
# 3) Review-level behavioural metrics
# ---------------------------------------------------------------------
def review_metrics(df):
    r = df.copy()
    r["review_len_chars"] = r["body"].fillna("").astype(str).str.len()
    r["review_len_words"] = r["body"].fillna("").astype(str).str.split().str.len()
    return r

rev_bm = review_metrics(rev_b)
rev_am = review_metrics(rev_a)

# ---------------------------------------------------------------------
# 4) Aggregations (AUTHOR / REPO / OVERALL)
# ---------------------------------------------------------------------
def agg_commit_level(df, group_col):
    return df.groupby(group_col).agg(
        commits=("sha","nunique"),
        mean_msg_chars=("msg_len_chars","mean"),
        median_msg_chars=("msg_len_chars","median"),
        mean_msg_words=("msg_len_words","mean"),
        mean_lines_changed=("lines_changed","mean"),
        median_lines_changed=("lines_changed","median"),
        mean_files_changed=("changed_files","mean")
    ).reset_index()

def agg_pr_level(df, group_col):
    return df.groupby(group_col).agg(
        prs=("pr_number","nunique"),
        mean_commits_per_pr=("commits_per_pr","mean"),
        median_commits_per_pr=("commits_per_pr","median"),
        mean_pr_lines=("pr_lines_changed","mean"),
        median_pr_lines=("pr_lines_changed","median"),
        mean_reviews_per_pr=("reviews_per_pr","mean"),
        median_reviews_per_pr=("reviews_per_pr","median"),
        mean_ttf_review_h=("time_to_first_review_h","mean"),
        median_ttf_review_h=("time_to_first_review_h","median"),
        mean_ttr_h=("time_to_resolution_h","mean"),
        median_ttr_h=("time_to_resolution_h","median")
    ).reset_index()

def agg_review_level(df, group_col):
    return df.groupby(group_col).agg(
        reviews=("body","count"),
        mean_review_chars=("review_len_chars","mean"),
        median_review_chars=("review_len_chars","median"),
        mean_review_words=("review_len_words","mean"),
        pct_changes_requested=("state", lambda s: (s=="CHANGES_REQUESTED").mean() if s.notna().any() else np.nan)
    ).reset_index()

# AUTHORS active on both sides (any activity)
authors_before = set(comm_bm.author.unique()) | set(prs_bm.author.unique()) | set(rev_bm.author.unique())
authors_after  = set(comm_am.author.unique()) | set(prs_am.author.unique()) | set(rev_am.author.unique())
authors_both   = sorted(authors_before & authors_after)

# AUTHOR tables
author_commit_before = agg_commit_level(comm_bm[comm_bm.author.isin(authors_both)], "author")
author_commit_after  = agg_commit_level(comm_am[comm_am.author.isin(authors_both)], "author")
author_pr_before     = agg_pr_level(prs_bm[prs_bm.author.isin(authors_both)], "author")
author_pr_after      = agg_pr_level(prs_am[prs_am.author.isin(authors_both)], "author")
author_rev_before    = agg_review_level(rev_bm[rev_bm.author.isin(authors_both)], "author")
author_rev_after     = agg_review_level(rev_am[rev_am.author.isin(authors_both)], "author")

# REPO tables
repo_commit_before = agg_commit_level(comm_bm, "repo")
repo_commit_after  = agg_commit_level(comm_am, "repo")
repo_pr_before     = agg_pr_level(prs_bm, "repo")
repo_pr_after      = agg_pr_level(prs_am, "repo")
repo_rev_before    = agg_review_level(rev_bm, "repo")
repo_rev_after     = agg_review_level(rev_am, "repo")

# OVERALL tables (single-row)
overall_commit_before = agg_commit_level(comm_bm, lambda _: "overall")
overall_commit_after  = agg_commit_level(comm_am, lambda _: "overall")
overall_pr_before     = agg_pr_level(prs_bm, lambda _: "overall")
overall_pr_after      = agg_pr_level(prs_am, lambda _: "overall")
overall_rev_before    = agg_review_level(rev_bm, lambda _: "overall")
overall_rev_after     = agg_review_level(rev_am, lambda _: "overall")

# ---------------------------------------------------------------------
# Print + save TABLES
# ---------------------------------------------------------------------
print_and_save(author_commit_before, "AUTHOR COMMIT METRICS BEFORE")
print_and_save(author_commit_after,  "AUTHOR COMMIT METRICS AFTER")
print_and_save(author_pr_before,     "AUTHOR PR METRICS BEFORE")
print_and_save(author_pr_after,      "AUTHOR PR METRICS AFTER")
print_and_save(author_rev_before,    "AUTHOR REVIEW METRICS BEFORE")
print_and_save(author_rev_after,     "AUTHOR REVIEW METRICS AFTER")

print_and_save(repo_commit_before, "REPO COMMIT METRICS BEFORE")
print_and_save(repo_commit_after,  "REPO COMMIT METRICS AFTER")
print_and_save(repo_pr_before,     "REPO PR METRICS BEFORE")
print_and_save(repo_pr_after,      "REPO PR METRICS AFTER")
print_and_save(repo_rev_before,    "REPO REVIEW METRICS BEFORE")
print_and_save(repo_rev_after,     "REPO REVIEW METRICS AFTER")

print_and_save(overall_commit_before, "OVERALL COMMIT METRICS BEFORE")
print_and_save(overall_commit_after,  "OVERALL COMMIT METRICS AFTER")
print_and_save(overall_pr_before,     "OVERALL PR METRICS BEFORE")
print_and_save(overall_pr_after,      "OVERALL PR METRICS AFTER")
print_and_save(overall_rev_before,    "OVERALL REVIEW METRICS BEFORE")
print_and_save(overall_rev_after,     "OVERALL REVIEW METRICS AFTER")


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
# 1) Commit message length distributions
save_hist(comm_bm["msg_len_words"], "Commit message words per commit (BEFORE)", "hist_commit_msg_words_before")
save_hist(comm_am["msg_len_words"], "Commit message words per commit (AFTER)",  "hist_commit_msg_words_after")

# 2) Lines changed per commit
save_hist(comm_bm["lines_changed"], "Lines changed per commit (BEFORE)", "hist_lines_per_commit_before")
save_hist(comm_am["lines_changed"], "Lines changed per commit (AFTER)",  "hist_lines_per_commit_after")

# 3) Files per commit
save_hist(comm_bm["changed_files"], "Files changed per commit (BEFORE)", "hist_files_per_commit_before")
save_hist(comm_am["changed_files"], "Files changed per commit (AFTER)",  "hist_files_per_commit_after")

# 4) Commits per PR
save_hist(prs_bm["commits_per_pr"], "Commits per PR (BEFORE)", "hist_commits_per_pr_before")
save_hist(prs_am["commits_per_pr"], "Commits per PR (AFTER)",  "hist_commits_per_pr_after")

# 5) PR size (lines)
save_hist(prs_bm["pr_lines_changed"], "Lines changed per PR (BEFORE)", "hist_lines_per_pr_before")
save_hist(prs_am["pr_lines_changed"], "Lines changed per PR (AFTER)",  "hist_lines_per_pr_after")

# 6) Reviews per PR
save_hist(prs_bm["reviews_per_pr"], "Reviews per PR (BEFORE)", "hist_reviews_per_pr_before")
save_hist(prs_am["reviews_per_pr"], "Reviews per PR (AFTER)",  "hist_reviews_per_pr_after")

# 7) Time-to-first-review
save_hist(prs_bm["time_to_first_review_h"], "Time to first review (hours) (BEFORE)", "hist_ttf_review_before")
save_hist(prs_am["time_to_first_review_h"], "Time to first review (hours) (AFTER)",  "hist_ttf_review_after")

# 8) Time-to-resolution
save_hist(prs_bm["time_to_resolution_h"], "Time to PR resolution (hours) (BEFORE)", "hist_ttr_before")
save_hist(prs_am["time_to_resolution_h"], "Time to PR resolution (hours) (AFTER)",  "hist_ttr_after")

# 9) Boxplots by author (before vs after)
def author_box(before_df, after_df, metric, title, out_base):
    b = before_df[["author", metric]].assign(phase="before")
    a = after_df[["author", metric]].assign(phase="after")
    long = pd.concat([b,a], ignore_index=True)
    save_boxplot(long, "phase", metric, title, out_base)

author_box(author_commit_before, author_commit_after, "mean_msg_words",
           "Author mean commit message words (before vs after)",
           "box_author_mean_msg_words")

author_box(author_commit_before, author_commit_after, "mean_lines_changed",
           "Author mean lines changed per commit (before vs after)",
           "box_author_mean_lines_per_commit")

author_box(author_pr_before, author_pr_after, "mean_commits_per_pr",
           "Author mean commits per PR (before vs after)",
           "box_author_commits_per_pr")

author_box(author_pr_before, author_pr_after, "mean_reviews_per_pr",
           "Author mean reviews per PR (before vs after)",
           "box_author_reviews_per_pr")

author_box(author_pr_before, author_pr_after, "median_ttf_review_h",
           "Author median time-to-first-review (hours) (before vs after)",
           "box_author_ttf_review")

# 10) Repo boxplots (before vs after)
def repo_box(before_df, after_df, metric, title, out_base):
    b = before_df[["repo", metric]].assign(phase="before")
    a = after_df[["repo", metric]].assign(phase="after")
    long = pd.concat([b,a], ignore_index=True)
    save_boxplot(long, "phase", metric, title, out_base)

repo_box(repo_commit_before, repo_commit_after, "mean_lines_changed",
         "Repo mean lines changed per commit (before vs after)",
         "box_repo_lines_per_commit")

repo_box(repo_pr_before, repo_pr_after, "mean_pr_lines",
         "Repo mean PR size (lines changed) (before vs after)",
         "box_repo_pr_lines")

repo_box(repo_pr_before, repo_pr_after, "mean_reviews_per_pr",
         "Repo mean reviews per PR (before vs after)",
         "box_repo_reviews_per_pr")

repo_box(repo_pr_before, repo_pr_after, "median_ttr_h",
         "Repo median time-to-resolution (hours) (before vs after)",
         "box_repo_ttr")

print("\nBehavioral RQ1 metrics complete.")
print("Tables -> outputs/tables/")
print("Plots  -> outputs/plots/")
