from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd


# ==========================================================
# CONFIG
# ==========================================================

REPO_LIST_TXT = "filtered_repos_3year100star.txt"
CLONE_DIR = "repos"
OUTPUT_DIR = "inputs/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# Parse repo list
# ==========================================================

def parse_repo_list(path: str):
    specs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]
            full_name = parts[0]

            boundary = None
            for p in parts[1:]:
                if "first agent pr" in p.lower():
                    boundary_str = p.split(":")[-1].strip()
                    boundary = pd.to_datetime(boundary_str, utc=True)

            if boundary is None:
                continue

            specs.append((full_name, boundary))

    return specs


# ==========================================================
# Lizard runner
# ==========================================================

def run_lizard(file_path: str) -> Dict[str, Any]:

    try:
        result = subprocess.run(
            ["python", "-m", "lizard", file_path],
            capture_output=True,
            text=True,
            timeout=10   # 🔥 10 second hard limit
        )
    except subprocess.TimeoutExpired:
        # Skip file if it hangs
        return {}
    except Exception:
        # Skip file if anything weird happens
        return {}

    output = result.stdout

    function_pattern = r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.+)"
    file_pattern = r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(.+)$"

    max_vals = {
        "max_nloc": 0,
        "max_ccn": 0,
        "max_tokens": 0,
        "max_params": 0,
        "max_length": 0
    }

    sum_ccn = 0
    sum_nloc = 0
    function_count = 0

    try:
        for line in output.splitlines():
            m = re.match(function_pattern, line)
            if m:
                nloc, ccn, tokens, params, length, _ = m.groups()
                nloc, ccn, tokens, params, length = map(int, [nloc, ccn, tokens, params, length])

                function_count += 1
                sum_ccn += ccn
                sum_nloc += nloc

                max_vals["max_nloc"] = max(max_vals["max_nloc"], nloc)
                max_vals["max_ccn"] = max(max_vals["max_ccn"], ccn)
                max_vals["max_tokens"] = max(max_vals["max_tokens"], tokens)
                max_vals["max_params"] = max(max_vals["max_params"], params)
                max_vals["max_length"] = max(max_vals["max_length"], length)
    except Exception:
        return {}

    file_summary = None
    for line in output.splitlines():
        m = re.match(file_pattern, line)
        if m:
            total_nloc, avg_nloc, avg_ccn, avg_tokens, func_count, _ = m.groups()
            file_summary = {
                "total_nloc": int(total_nloc),
                "avg_nloc": float(avg_nloc),
                "avg_ccn": float(avg_ccn),
                "avg_tokens": float(avg_tokens),
                "function_count": int(func_count),
            }
            break

    if not file_summary:
        return {}

    return {
        **file_summary,
        **max_vals,
        "sum_ccn": sum_ccn,
        "mean_ccn": sum_ccn / function_count if function_count else 0,
    }

# ==========================================================
# Extract PR merge commits locally
# ==========================================================

def get_merge_commits(repo_dir: str):

    result = subprocess.run(
        ["git", "-C", repo_dir, "log", "--merges",
         "--pretty=format:%H|%cI|%s"],
        capture_output=True,
        text=True
    )

    merges = []

    for line in result.stdout.splitlines():
        sha, date_str, message = line.split("|", 2)

        pr_match = re.search(r"#(\d+)", message)
        if not pr_match:
            continue

        pr_number = int(pr_match.group(1))
        date = pd.to_datetime(date_str, utc=True)

        merges.append((sha, pr_number, date))

    return merges


def get_files_changed(repo_dir: str, sha: str):

    result = subprocess.run(
        ["git", "-C", repo_dir, "diff",
         "--name-only", f"{sha}^", sha],
        capture_output=True,
        text=True
    )

    return result.stdout.splitlines()



# ==========================================================
# MAIN
# ==========================================================

def main():

    specs = parse_repo_list(REPO_LIST_TXT)

    before_rows: List[Dict[str, Any]] = []
    after_rows: List[Dict[str, Any]] = []

    total_repos = len(specs)

    for repo_index, (repo_name, boundary) in enumerate(specs, start=1):

        repo_dir = os.path.join(CLONE_DIR, repo_name.replace("/", "_"))
        if not os.path.isdir(repo_dir):
            print(f"[{repo_index}/{total_repos}] Skipping {repo_name} (not cloned)")
            continue

        print("\n" + "=" * 80)
        print(f"[{repo_index}/{total_repos}] Processing repo: {repo_name}")
        print("=" * 80)

        merges = get_merge_commits(repo_dir)
        total_merges = len(merges)

        print(f"Total PR merge commits found: {total_merges}")

        if total_merges == 0:
            print("No merges found. Skipping.")
            continue

        repo_before_count = 0
        repo_after_count = 0
        repo_files_analyzed = 0

        
        for i, (sha, pr_number, date) in enumerate(merges, start=1):

            LOG_EVERY = 10

            subprocess.run(
                ["git", "-C", repo_dir, "checkout", sha],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            files = get_files_changed(repo_dir, sha)

            pr_file_count = 0

            for file_path in files:

                full_path = os.path.join(repo_dir, file_path)
                if not os.path.exists(full_path):
                    continue

                metrics = run_lizard(full_path)
                if not metrics:
                    continue

                row = {
                    "repo": repo_name,
                    "pr_number": pr_number,
                    "commit_sha": sha,
                    "file_path": file_path,
                    "date": date,
                    **metrics
                }

                if date < boundary:
                    before_rows.append(row)
                    repo_before_count += 1
                else:
                    after_rows.append(row)
                    repo_after_count += 1

                pr_file_count += 1
                repo_files_analyzed += 1

            # 🔥 LOGGING MOVED HERE (after PR finishes)
            if i % LOG_EVERY == 0 or i == total_merges:
                percent_complete = (i / total_merges) * 100
                print(
                    f"[{repo_name}] PR {i}/{total_merges} "
                    f"({percent_complete:.1f}%) | PR #{pr_number} | "
                    f"Files in PR: {pr_file_count} | "
                    f"Total Files so far: {repo_files_analyzed}"
                )


        print("\nFinished repo:", repo_name)
        print(f"   Files analyzed: {repo_files_analyzed}")
        print(f"   Before boundary: {repo_before_count}")
        print(f"   After boundary: {repo_after_count}")
        print("-" * 80)

    print("\nWriting parquet files...")

    pd.DataFrame(before_rows).to_parquet(
        os.path.join(OUTPUT_DIR, "pr_file_complexity_before.parquet"),
        index=False
    )

    pd.DataFrame(after_rows).to_parquet(
        os.path.join(OUTPUT_DIR, "pr_file_complexity_after.parquet"),
        index=False
    )

    print("\nAll repos complete.")
    print(f"Total before rows: {len(before_rows)}")
    print(f"Total after rows: {len(after_rows)}")


if __name__ == "__main__":
    main()
