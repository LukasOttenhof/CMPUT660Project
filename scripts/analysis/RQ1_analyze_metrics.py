#!/usr/bin/env python3
"""
RQ1: Author-level activity change after introduction of code agents.

Inputs:  inputs/processed/*.parquet
Outputs: outputs/tables/*.csv
         outputs/plots/*.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# PATHS
# =============================================================================

INPUT_DIR = Path("inputs/processed")
TABLE_DIR = Path("outputs/tables")
PLOT_DIR  = Path("outputs/plots")

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

MIN_MONTHS_WINDOW = 1.0


# =============================================================================
# LOADING
# =============================================================================

def load_parquet(name: str) -> pd.DataFrame:
    path = INPUT_DIR / name
    if not path.exists():
        print(f"⚠ Missing {path}. Returning empty df.")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df

commits_b = load_parquet("commits_before.parquet")
commits_a = load_parquet("commits_after.parquet")

prs_b     = load_parquet("pull_requests_before.parquet")
prs_a     = load_parquet("pull_requests_after.parquet")

reviews_b = load_parquet("reviews_before.parquet")
reviews_a = load_parquet("reviews_after.parquet")


# =============================================================================
# TAG PHASE
# =============================================================================

def tag(df, label):
    if len(df) == 0:
        return df
    df = df.copy()
    df["phase"] = label
    return df

commits_b = tag(commits_b, "before")
commits_a = tag(commits_a, "after")

prs_b     = tag(prs_b, "before")
prs_a     = tag(prs_a, "after")

reviews_b = tag(reviews_b, "before")
reviews_a = tag(reviews_a, "after")


# =============================================================================
# CONCAT
# =============================================================================

commits_all = pd.concat([commits_b, commits_a], ignore_index=True)
prs_all     = pd.concat([prs_b, prs_a], ignore_index=True)
reviews_all = pd.concat([reviews_b, reviews_a], ignore_index=True)


# =============================================================================
# AUTHOR-LEVEL WINDOWS
# =============================================================================

def months_between(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.nan
    days = (b - a).total_seconds() / (3600 * 24)
    return max(days / 30.4375, MIN_MONTHS_WINDOW)

def author_windows(df, label):
    """Compute active window length for EACH author in each phase."""
    if len(df) == 0:
        return pd.DataFrame(columns=["author", f"months_{label}"])

    tmp = (
        df.groupby(["author"])["date"]
          .agg(["min", "max"])
          .reset_index()
    )
    tmp[f"months_{label}"] = tmp.apply(lambda r: months_between(r["min"], r["max"]), axis=1)
    return tmp[["author", f"months_{label}"]]


win_b = author_windows(pd.concat([commits_b, prs_b, reviews_b]), "before")
win_a = author_windows(pd.concat([commits_a, prs_a, reviews_a]), "after")


# =============================================================================
# AUTHOR-LEVEL COUNTS
# =============================================================================

def author_counts(df, metric_name):
    """Count events per author per phase."""
    if len(df) == 0:
        return pd.DataFrame(columns=["author", "phase", metric_name])

    return (
        df.groupby(["author", "phase"])
          .size()
          .reset_index(name=metric_name)
    )


commit_counts = author_counts(commits_all, "commits")
pr_counts     = author_counts(prs_all, "prs")
review_counts = author_counts(reviews_all, "reviews")


# =============================================================================
# PIVOT BEFORE/AFTER FOR EACH METRIC
# =============================================================================

def pivot_author(df, metric):
    """Pivot to one row per author with before/after/delta fields."""
    wide = df.pivot(index="author", columns="phase", values=metric).fillna(0).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={
        "before": f"{metric}_before",
        "after":  f"{metric}_after",
    })
    wide[f"{metric}_delta"] = wide[f"{metric}_after"] - wide[f"{metric}_before"]
    wide[f"{metric}_pct_change"] = np.where(
        wide[f"{metric}_before"] > 0,
        (wide[f"{metric}_delta"] / wide[f"{metric}_before"]) * 100,
        np.nan
    )
    return wide

commits_w = pivot_author(commit_counts, "commits")
prs_w     = pivot_author(pr_counts, "prs")
reviews_w = pivot_author(review_counts, "reviews")

# merge metrics into one author table
metrics = commits_w.merge(prs_w, on="author", how="outer").merge(reviews_w, on="author", how="outer")


# =============================================================================
# MERGE WINDOWS + CREATE RATE METRICS
# =============================================================================

metrics = metrics.merge(win_b, on="author", how="left")
metrics = metrics.merge(win_a, on="author", how="left")

for m in ["commits", "prs", "reviews"]:
    metrics[f"{m}_per_month_before"] = metrics[f"{m}_before"] / metrics["months_before"]
    metrics[f"{m}_per_month_after"]  = metrics[f"{m}_after"]  / metrics["months_after"]
    metrics[f"{m}_per_month_delta"]  = (
        metrics[f"{m}_per_month_after"] - metrics[f"{m}_per_month_before"]
    )


# =============================================================================
# SAVE TABLES
# =============================================================================

metrics.to_csv(TABLE_DIR / "rq1_author_metrics.csv", index=False)

print("Saved author-level metrics to", TABLE_DIR / "rq1_author_metrics.csv")


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def summary(series):
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    return pd.Series({
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(),
        "min": s.min(),
        "max": s.max(),
        "n_authors": s.count(),
    })

rows = []
for m in ["commits", "prs", "reviews"]:
    for col in [
        f"{m}_before", f"{m}_after",
        f"{m}_delta", f"{m}_pct_change",
        f"{m}_per_month_before", f"{m}_per_month_after",
        f"{m}_per_month_delta",
    ]:
        s = summary(metrics[col])
        s["metric"] = col
        rows.append(s)

summary_df = pd.DataFrame(rows)
summary_df = summary_df[["metric", "mean", "median", "std", "min", "max", "n_authors"]]
summary_df.to_csv(TABLE_DIR / "rq1_author_summary.csv", index=False)

print("Saved summary to", TABLE_DIR / "rq1_author_summary.csv")


# =============================================================================
# PLOTS
# =============================================================================

def savefig(name):
    plt.tight_layout()
    plt.savefig(PLOT_DIR / name, dpi=200)
    plt.close()

# Boxplots (raw)
for metric,title in [("commits","Commits"),
                     ("prs","Pull Requests"),
                     ("reviews","Reviews")]:

    df_long = metrics.melt(
        id_vars=["author"],
        value_vars=[f"{metric}_before", f"{metric}_after"],
        var_name="phase",
        value_name="count",
    )
    df_long["phase"] = df_long["phase"].str.replace(f"{metric}_","")

    plt.figure(figsize=(7,5))
    sns.boxplot(data=df_long, x="phase", y="count")
    plt.yscale("log")
    plt.title(f"{title} per Author — Before vs After")
    savefig(f"author_box_{metric}.png")


# Paired scatter (before vs after)
for metric,title in [("commits","Commits"),
                     ("prs","Pull Requests"),
                     ("reviews","Reviews")]:

    plt.figure(figsize=(6,6))
    sns.scatterplot(
        data=metrics,
        x=f"{metric}_before",
        y=f"{metric}_after",
    )
    maxv = max(
        metrics[f"{metric}_before"].max(),
        metrics[f"{metric}_after"].max()
    )
    plt.plot([0, maxv], [0, maxv], linestyle="--")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"{title} per Author — Before vs After")
    savefig(f"author_paired_{metric}.png")


# Rate-normalized
for metric,title in [("commits","Commits"),
                     ("prs","Pull Requests"),
                     ("reviews","Reviews")]:

    df_long = metrics.melt(
        id_vars=["author"],
        value_vars=[f"{metric}_per_month_before", f"{metric}_per_month_after"],
        var_name="phase",
        value_name="rate",
    )
    df_long["phase"] = df_long["phase"].str.replace(f"{metric}_per_month_", "")

    plt.figure(figsize=(7,5))
    sns.boxplot(data=df_long, x="phase", y="rate")
    plt.yscale("log")
    plt.title(f"{title} per Month per Author — Before vs After")
    savefig(f"author_rate_box_{metric}.png")

print("\n🎉 RQ1 per-author analysis complete.")
