#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

NUMERIC_PATH = "inputs/processed/numeric_raw.parquet"
OUT_DIR = Path("outputs/plots_before_after/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading numeric dataset...")
df = pd.read_parquet(NUMERIC_PATH)

# Ensure date column is datetime
df["date"] = pd.to_datetime(df["date"], utc=True)

repos = df["repo"].unique()

# Metrics we want to compute before/after
metrics = {
    "commit_count": lambda x: (x["activity_type"] == "commit").sum(),
    "pr_created":   lambda x: (x["activity_type"] == "pr_created").sum(),
    "pr_merged":    lambda x: (x["activity_type"] == "pr_merged").sum(),
    "loc_added":    lambda x: x["loc_added"].sum(),
    "loc_deleted":  lambda x: x["loc_deleted"].sum(),
}

rows = []

print("\n=== Computing BEFORE/AFTER using same logic as diagnostic script ===\n")

for repo in repos:
    repo_df = df[df["repo"] == repo]

    # Identify boundary: earliest PR CREATED
    pr_events = repo_df[repo_df["activity_type"] == "pr_created"]

    if pr_events.empty:
        print(f"{repo:45} -> NO pr_created events, skipping")
        continue

    first_pr_date = pr_events["date"].min()

    before_df = repo_df[repo_df["date"] < first_pr_date]
    after_df  = repo_df[repo_df["date"] > first_pr_date]

    print(f"{repo:45} → before={len(before_df)}  after={len(after_df)}")

    for metric_name, func in metrics.items():
        rows.append({
            "repo": repo,
            "metric": metric_name,
            "period": "before",
            "value": func(before_df)
        })
        rows.append({
            "repo": repo,
            "metric": metric_name,
            "period": "after",
            "value": func(after_df)
        })

# Build long-form dataframe
result = pd.DataFrame(rows)

print("\nSaving plots...\n")

# Generate boxplots
for metric in metrics.keys():
    df_m = result[result["metric"] == metric]

    before_vals = df_m[df_m["period"] == "before"]["value"]
    after_vals  = df_m[df_m["period"] == "after"]["value"]

    plt.figure(figsize=(10, 6))
    plt.boxplot([before_vals, after_vals],
                labels=["Before", "After"],
                showfliers=True)
    plt.title(f"{metric} — Before vs After First PR Created Event")
    plt.ylabel(metric)
    plt.grid(axis="y", alpha=0.3)

    out_path = OUT_DIR / f"{metric}_before_after.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✔ Saved {out_path}")

print("\n🎉 FINISHED — boxplots available in outputs/plots_before_after/")
