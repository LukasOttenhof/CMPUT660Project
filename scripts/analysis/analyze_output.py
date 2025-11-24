#!/usr/bin/env python3
"""
basic_analysis.py

Quick exploratory analysis for your repo_analysis.py output.

Loads the 20 Parquets produced under ./outputs/, computes summary
tables, and generates basic boxplots for commits, PRs, and issues.

Plots are saved under ./analysis_outputs/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = Path("outputs")
ANALYSIS_DIR = Path("analysis_outputs")
ANALYSIS_DIR.mkdir(exist_ok=True)

# Use default style
sns.set(style="whitegrid")


# =============================================================================
# LOADER
# =============================================================================

def load_parquet(name):
    path = OUTPUT_DIR / name
    if not path.exists():
        print(f"⚠️ Missing parquet: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


# Load everything
commits_b  = load_parquet("commits_before.parquet")
commits_a  = load_parquet("commits_after.parquet")
prs_b      = load_parquet("pull_requests_before.parquet")
prs_a      = load_parquet("pull_requests_after.parquet")
issues_b   = load_parquet("issues_before.parquet")
issues_a   = load_parquet("issues_after.parquet")
reviews_b  = load_parquet("reviews_before.parquet")
reviews_a  = load_parquet("reviews_after.parquet")

commit_msgs_b = load_parquet("commit_messages_before.parquet")
commit_msgs_a = load_parquet("commit_messages_after.parquet")
pr_bodies_b   = load_parquet("pr_bodies_before.parquet")
pr_bodies_a   = load_parquet("pr_bodies_after.parquet")


# =============================================================================
# SUMMARY TABLES
# =============================================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

print_section("🔍 DATASET SUMMARY")

summary = {
    "commits_before": len(commits_b),
    "commits_after": len(commits_a),
    "prs_before": len(prs_b),
    "prs_after": len(prs_a),
    "issues_before": len(issues_b),
    "issues_after": len(issues_a),
    "reviews_before": len(reviews_b),
    "reviews_after": len(reviews_a),
    "commit_messages_before": len(commit_msgs_b),
    "commit_messages_after": len(commit_msgs_a),
    "pr_bodies_before": len(pr_bodies_b),
    "pr_bodies_after": len(pr_bodies_a),
}

df_summary = pd.DataFrame(summary.items(), columns=["dataset", "count"])
print(df_summary.to_string(index=False))


# =============================================================================
# PER-REPO & PER-AUTHOR ACTIVITY TABLES
# =============================================================================

print_section("📊 COMMITS PER REPO")

commits_all = pd.concat([commits_b.assign(phase="before"),
                         commits_a.assign(phase="after")],
                        ignore_index=True)

table_repo = (
    commits_all.groupby(["repo", "phase"])
    .agg(commits=("sha", "count"),
         loc_added=("loc_added", "sum"),
         loc_deleted=("loc_deleted", "sum"))
    .reset_index()
)

print(table_repo.head(20).to_string(index=False))


print_section("📊 COMMITS PER AUTHOR (Top 20)")
table_author = (
    commits_all.groupby("author")
               .agg(count=("sha", "count"),
                    repos=("repo", pd.Series.nunique))
               .sort_values("count", ascending=False)
               .head(20)
)

print(table_author.to_string())


# =============================================================================
# BOXPLOTS
# =============================================================================

def save_plot(fig, name):
    fig.savefig(ANALYSIS_DIR / name, bbox_inches="tight")
    plt.close(fig)


# ---- Commit LOC delta boxplot ----
if len(commits_all):
    commits_all["loc_delta"] = commits_all["loc_added"] - commits_all["loc_deleted"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=commits_all, x="phase", y="loc_delta", ax=ax)
    ax.set_title("Commit LOC Delta (Before vs After)")
    save_plot(fig, "boxplot_loc_delta.png")


# ---- PR merge time boxplot ----
prs_all = pd.concat([prs_b.assign(phase="before"),
                     prs_a.assign(phase="after")],
                    ignore_index=True)

# PR merge events contain time_to_merge_hours column
prs_merge = prs_all.dropna(subset=["time_to_merge_hours"])

if len(prs_merge):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=prs_merge, x="phase", y="time_to_merge_hours", ax=ax)
    ax.set_title("PR Merge Time (Hours) — Before vs After")
    ax.set_yscale("log")  # usually merge times have long tails
    save_plot(fig, "boxplot_pr_merge_times.png")


# ---- Issue close time boxplot ----
issues_all = pd.concat([issues_b.assign(phase="before"),
                        issues_a.assign(phase="after")],
                       ignore_index=True)

issues_close = issues_all.dropna(subset=["time_to_close_hours"])

if len(issues_close):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=issues_close, x="phase", y="time_to_close_hours", ax=ax)
    ax.set_title("Issue Close Time (Hours) — Before vs After")
    ax.set_yscale("log")
    save_plot(fig, "boxplot_issue_close_times.png")


# =============================================================================
# TEXT ANALYSIS
# =============================================================================

print_section("📝 TEXT LENGTH ANALYSIS")

def add_length(df):
    if len(df) == 0:
        return df
    df = df.copy()
    df["length"] = df["text"].fillna("").str.len()
    return df

commit_msgs = pd.concat([
    add_length(commit_msgs_b).assign(phase="before"),
    add_length(commit_msgs_a).assign(phase="after")
])

pr_bodies = pd.concat([
    add_length(pr_bodies_b).assign(phase="before"),
    add_length(pr_bodies_a).assign(phase="after")
])


# --- Commit message lengths ---
print("Commit message length (summary):")
print(commit_msgs.groupby("phase")["length"].describe().to_string())

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=commit_msgs, x="phase", y="length", ax=ax)
ax.set_title("Commit Message Length — Before vs After")
save_plot(fig, "boxplot_commit_msg_length.png")


# --- PR body lengths ---
print("\nPR body length (summary):")
print(pr_bodies.groupby("phase")["length"].describe().to_string())

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=pr_bodies, x="phase", y="length", ax=ax)
ax.set_title("PR Body Length — Before vs After")
save_plot(fig, "boxplot_pr_body_length.png")


print("\n🎉 Analysis complete.")
print(f"Plots saved in: {ANALYSIS_DIR.resolve()}")
