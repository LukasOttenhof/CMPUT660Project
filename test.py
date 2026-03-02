# import glob

# path = r'G:\CMPUT660Project\inputs\human_2\incremental\*_commits_before.parquet'
# file_count = len(glob.glob(path))

# print(f"Total files matching pattern: {file_count}")
# import glob

# path = r'G:\CMPUT660Project\inputs\human_2\incremental\*_commits_before.parquet'

# # Get the list of all matching files
# files = glob.glob(path)

# # Print the total count
# print(f"Total files matching pattern: {len(files)}\n")

# # Print each file path
# for f in files:
#     print(f)
import glob
import os

# --- 1. Get all parquet files that actually exist ---
parquet_pattern = r'G:\CMPUT660Project\inputs\human_2\incremental\*_prs_before.parquet'
existing_files = glob.glob(parquet_pattern)

# Extract repo names from parquet filenames
existing_repos = set()
for f in existing_files:
    basename = os.path.basename(f)  # e.g., codecrafters-io_build-your-own-x_prs_before.parquet
    repo_name = basename.replace("_prs_before.parquet", "")
    existing_repos.add(repo_name.lower())

print(f"Usable repos found in incremental folder ({len(existing_repos)}):")
for r in sorted(existing_repos):
    print(r)

# --- 2. Filter your original repo list ---
REPO_LIST_TXT = r"G:\CMPUT660Project\scripts\build_dataset\true_human_baseline_con_reformatted.txt"
filtered_repos = []

with open(REPO_LIST_TXT, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Get the repo part before the first " | "
        repo_full = line.split("|")[0].strip()  # e.g., codecrafters-io/build-your-own-x
        # Convert to underscore form to match parquet filenames
        repo_formatted = repo_full.replace("/", "_").lower()
        if repo_formatted in existing_repos:
            filtered_repos.append(line)

# --- 3. Save filtered list ---
OUTPUT_TXT = r"G:\CMPUT660Project\scripts\build_dataset\true_human_baseline_filtered.txt"
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    for line in filtered_repos:
        f.write(line + "\n")

print(f"\nFiltered repo list saved to {OUTPUT_TXT} ({len(filtered_repos)} repos)")