import pandas as pd

AIDEV_REPO_PATH = "/Users/lukas./Desktop/CMPUT660Project/inputs/processed/commits_before.parquet"

# Load the full Parquet file
repo_df = pd.read_parquet(AIDEV_REPO_PATH)

# Inspect the first few rows
print(repo_df.head())

# Optional: get summary info
print(repo_df.info())
print(repo_df.columns)
print(f"Total rows: {len(repo_df)}")
unique_repos_count = repo_df['author'].nunique()
print(unique_repos_count)

