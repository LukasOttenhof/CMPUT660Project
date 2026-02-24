import json
from pathlib import Path
import pandas as pd
from data_loader import load_all
from scipy import stats
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "inputs" / "processed" / "repo_month_complexity_cache"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def get_stats_row(data_series):
    """Returns the Mean and the 95% CI in a bracketed string."""
    n = len(data_series)
    mean = data_series.mean()
    if n < 2:
        return mean, "[N/A]"
    
    # Standard Error of the Mean
    sem = stats.sem(data_series)
    # Calculate Margin of Error (t-score * SEM)
    # 0.95 confidence level
    margin = sem * stats.t.ppf((1 + 0.95) / 2., n - 1)
    
    low = mean - margin
    high = mean + margin
    
    return mean, f"[{low:.2f}, {high:.2f}]"

def load_repo_complexity(repo: str) -> dict | None:
    # Match the "safe" naming convention used in your generation script
    safe_repo = repo.replace("/", "_")
    path = BASE / f"{safe_repo}_cache.json"
    
    if not path.exists():
        print(f"Warning: Complexity cache not found for repo '{repo}' at {path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("sha_to_metrics", {})
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
import os
def main():
    TARGET_REPOS = [
    'noperson83/WBee-appware', 
    'dBinet/SquirrelAttack', 
    'supermarsx/smtp-burst'
]

    # Set this to your project data root (e.g., 'E:/CMPUT660Project/data')
    DATA_ROOT = 'E:/CMPUT660Project' 


    findings = {repo: [] for repo in TARGET_REPOS}
    
    print(f"--- Auditing project data for target repositories ---\n")

    for root, dirs, files in os.walk(DATA_ROOT):
        for file in files:
            if file.endswith(('.json', '.csv', '.txt')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        for repo in TARGET_REPOS:
                            if repo in content:
                                findings[repo].append(file_path)
                except Exception as e:
                    print(f"Could not read {file}: {e}")

    # Display Results
    for repo, locations in findings.items():
        print(f"Repo: **{repo}**")
        if locations:
            # Set converts to unique list, sorted for readability
            unique_locations = sorted(list(set(locations)))
            print(f"Found in {len(unique_locations)} files:")
            for loc in unique_locations:
                print(f"  - {os.path.relpath(loc, DATA_ROOT)}")
        else:
            print("  [!] NOT FOUND in any data files.")
        print("-" * 30)
    # # For now, I'll assume commits_b and commits_a are DataFrames 
    # # containing at least 'repo' and 'sha' columns.

    # before_rows = []
    # after_rows = []
    # repo_cache: dict[str, dict] = {}

    # def get_repo_data(repo):
    #     if repo not in repo_cache:
    #         repo_cache[repo] = load_repo_complexity(repo)
    #     return repo_cache[repo]

    # def process_commits(df_commits, target_list):
    #     for c in df_commits.itertuples(index=False):
    #         repo_data = get_repo_data(c.repo)
    #         if not repo_data:
    #             continue

    #         metrics = repo_data.get(c.sha)
    #         # Skip if error or missing
    #         if not metrics or "_error" in metrics:
    #             continue

    #         # Create a copy of metrics but exclude the large nested list
    #         # so the DataFrame remains flat and readable
    #         flat_metrics = {k: v for k, v in metrics.items() if k != "functions_info"}

    #         target_list.append({
    #             "repo": c.repo,
    #             "sha": c.sha,
    #             "author": c.author,
    #             "date": c.date,
    #             "loc_added": c.loc_added,
    #             "loc_deleted": c.loc_deleted,
    #             "files_changed": c.files_changed,
    #             **flat_metrics,
    #         })

    # # Process both sets
    # process_commits(commits_b, before_rows)
    # process_commits(commits_a, after_rows)

    # df_before = pd.DataFrame(before_rows)
    # df_after = pd.DataFrame(after_rows)

    # print(f"Before shape: {df_before.shape}")
    # print(f"After shape: {df_after.shape}")

    # # -----------------------------------------------------------------
    # # Aggregation
    # # -----------------------------------------------------------------

    # # Updated to match the keys in your new JSON snippet
    # numeric_cols = [
    #     "total_nloc",
    #     "avg_ccn",
    #     "function_count",
    #     "max_nloc",
    #     "max_ccn",
    #     "max_tokens",
    #     "max_params",
    #     "max_length",
    #     "loc_added",
    #     "loc_deleted",
    #     "files_changed"
    # ]

    # # Ensure columns exist before aggregating (handles empty DFs or missing keys)
    # existing_cols = [c for c in numeric_cols if c in df_before.columns]

    # # Overall Metrics
    # before_mean = df_before[existing_cols].mean().rename("mean_before")
    # after_mean = df_after[existing_cols].mean().rename("mean_after")
    # before_median = df_before[existing_cols].median().rename("median_before")
    # after_median = df_after[existing_cols].median().rename("median_after")

    # agg_df = pd.concat([before_mean, before_median, after_mean, after_median], axis=1)

    # print("\nAggregated metrics (Overall):")
    # print(agg_df)

    # # Per-Repo Metrics
    # repo_agg = pd.concat([
    #     df_before.groupby("repo")[existing_cols].mean().add_suffix("_mean_before"),
    #     df_after.groupby("repo")[existing_cols].mean().add_suffix("_mean_after")
    # ], axis=1)

    # # Save to CSV for easier viewing if needed
    # # agg_df.to_csv("complexity_summary.csv")
    # numeric_cols = ["avg_ccn", "total_nloc", "function_count"]
    # repo_means_before = df_before.groupby("repo")[numeric_cols].mean()
    # repo_means_after = df_after.groupby("repo")[numeric_cols].mean()

    # results = []

    # for col in numeric_cols:
    #     # Calculate for Before
    #     b_mean, b_ci = get_stats_row(repo_means_before[col])
    #     results.append({
    #         "Metric": col,
    #         "Period": "Before",
    #         "Mean": round(b_mean, 2),
    #         "95% CI": b_ci
    #     })
        
    #     # Calculate for After
    #     a_mean, a_ci = get_stats_row(repo_means_after[col])
    #     results.append({
    #         "Metric": col,
    #         "Period": "After",
    #         "Mean": round(a_mean, 2),
    #         "95% CI": a_ci
    #     })

    # # Create final table
    # final_table = pd.DataFrame(results)
    # print(final_table.to_string(index=False))

if __name__ == "__main__":
    main()