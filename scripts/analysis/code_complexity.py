import json
from pathlib import Path
import pandas as pd
from data_loader import load_all

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "inputs/processed/repo_month_complexity_cache"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_repo_complexity(repo: str) -> dict | None:

    safe_repo = repo.replace("/", "_") # files are saved with _ but repos in before and after have /
    path = BASE / f"{safe_repo}_cache.json"
    if not path.exists():
        print(f"Warning: Complexity cache not found for repo '{repo}' at {path}")
        return None

    with open(path, "r") as f:
        data = json.load(f)

    return data.get("sha_to_metrics", {})


def main():
    data = load_all()

    commits_b = data["commits_before"]
    commits_a = data["commits_after"]

    before_rows = []
    after_rows = []

    # cache repo jsons so we don't re-read them multiple times
    repo_cache: dict[str, dict] = {}

    def get_repo_data(repo):
        if repo not in repo_cache:
            repo_cache[repo] = load_repo_complexity(repo)
        return repo_cache[repo]

    #before
    for c in commits_b.itertuples(index=False):
        repo_data = get_repo_data(c.repo)
        if not repo_data:
            continue

        metrics = repo_data.get(c.sha)
        if not metrics:
            continue

        before_rows.append({
            "repo": c.repo,
            "sha": c.sha,
            "author": c.author,
            "date": c.date,
            "loc_added": c.loc_added,
            "loc_deleted": c.loc_deleted,
            "files_changed": c.files_changed,
            **metrics,
        })

    # afta
    for c in commits_a.itertuples(index=False):
        repo_data = get_repo_data(c.repo)
        if not repo_data:
            continue

        metrics = repo_data.get(c.sha)
        if not metrics:
            continue

        after_rows.append({
            "repo": c.repo,
            "sha": c.sha,
            "author": c.author,
            "date": c.date,
            "loc_added": c.loc_added,
            "loc_deleted": c.loc_deleted,
            "files_changed": c.files_changed,
            **metrics,
        })

    df_before = pd.DataFrame(before_rows)
    df_after = pd.DataFrame(after_rows)

    print("Before shape:", df_before.shape)
    print("After shape:", df_after.shape)

    # UNCOMMENT TO SEE SAMPELE DATA
    # print("\nBefore sample:")
    # print(df_before.head())

    # print("\nAfter sample:")
    # print(df_after.head())


    numeric_cols = [
        "total_nloc", "avg_nloc", "avg_ccn", "avg_tokens",
        "function_count", "files_analyzed",
        "loc_added", "loc_deleted", "files_changed"
    ]


    before_mean = df_before[numeric_cols].mean().rename("mean_before")
    after_mean = df_after[numeric_cols].mean().rename("mean_after")

    before_median = df_before[numeric_cols].median().rename("median_before")
    after_median = df_after[numeric_cols].median().rename("median_after")

    agg_df = pd.concat([before_mean, before_median, after_mean, after_median], axis=1)
    print("\nAggregated metrics before vs after (overall):")
    print(agg_df)


    repo_mean_before = df_before.groupby("repo")[numeric_cols].mean()
    repo_mean_after = df_after.groupby("repo")[numeric_cols].mean()

    repo_median_before = df_before.groupby("repo")[numeric_cols].median()
    repo_median_after = df_after.groupby("repo")[numeric_cols].median()


    repo_agg = pd.concat([
        repo_mean_before.add_suffix("_mean_before"),
        repo_median_before.add_suffix("_median_before"),
        repo_mean_after.add_suffix("_mean_after"),
        repo_median_after.add_suffix("_median_after")
    ], axis=1)

    # print("\nAggregated metrics per repo (first 5):")
    # print(repo_agg.head())



if __name__ == "__main__":
    main()
