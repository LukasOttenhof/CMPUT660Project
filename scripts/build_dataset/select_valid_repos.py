#!/usr/bin/env python3
"""
select_valid_repos.py

Select 50 repos that:
  1) have at least one agentic PR (from AIDEV)
  2) have stars >= MIN_STARS (from AIDEV repo metadata)
  3) have repo age >= MIN_REPO_AGE_YEARS
  4) have total commits <= MAX_COMMITS (GitHub API)
  5) have enough PR + text activity before/after agentic introduction

Outputs: inputs/processed/final_repo_list.csv
"""

import os
import time
import random
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple

# ======================================================
# CONFIG
# ======================================================
AIDEV_PR_PATH   = "inputs/raw/pull_request.parquet"
AIDEV_REPO_PATH = "inputs/raw/all_repository.parquet"

TOKENS = [
    "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",
    "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",
    "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"
]
TOKENS = [t for t in TOKENS if t]
if len(TOKENS) == 0:
    raise RuntimeError("No GitHub tokens found.")

MIN_STARS = 50
MIN_REPO_AGE_YEARS = 3

MIN_BEFORE_TEXT = 10
MIN_AFTER_TEXT  = 10
MIN_BEFORE_PRS  = 20
MIN_AFTER_PRS   = 20

MAX_COMMITS     = 3000
TARGET_REPOS    = 50

OUT_PATH = "inputs/processed/final_repo_list.csv"

PER_PAGE = 100
SLEEP_ON_FULL_LIMIT = 60  # seconds

# ======================================================
# TOKEN ROTATION + SAFE REQUEST
# ======================================================
token_index = 0

def next_headers() -> Dict[str, str]:
    """Rotate GitHub tokens each request."""
    global token_index
    tok = TOKENS[token_index]
    token_index = (token_index + 1) % len(TOKENS)
    return {
        "Authorization": f"Bearer {tok}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "aidev-repo-selector"
    }

def gh_get(url: str, params: Optional[Dict[str, Any]] = None, retries: int = 6):
    """GET with rate-limit handling and token rotation."""
    last_err = None
    for attempt in range(retries):
        r = requests.get(url, headers=next_headers(), params=params)
        if r.status_code == 200:
            return r

        if r.status_code in (404, 451):  # repo gone/private
            return r

        # Rate limit
        if r.status_code == 403:
            msg = ""
            try:
                msg = (r.json().get("message") or "").lower()
            except:
                pass

            remaining = r.headers.get("X-RateLimit-Remaining")
            reset_ts  = r.headers.get("X-RateLimit-Reset")

            if remaining == "0" or "rate limit" in msg:
                if attempt >= len(TOKENS) - 1:
                    sleep_for = SLEEP_ON_FULL_LIMIT
                    if reset_ts:
                        try:
                            reset_ts = int(reset_ts)
                            sleep_for = max(5, reset_ts - int(time.time()))
                        except:
                            pass
                    print(f"    ⏳ All tokens limited. Sleeping {sleep_for}s...")
                    time.sleep(sleep_for)
                continue

            # secondary rate limit
            time.sleep(5)
            continue

        last_err = r
        time.sleep(3)

    return last_err if last_err is not None else None


# ======================================================
# FETCH REPO CREATION DATE
# ======================================================
def get_repo_creation_date(repo_fullname: str) -> Optional[datetime]:
    """Fetch repository creation timestamp from GitHub API."""
    url = f"https://api.github.com/repos/{repo_fullname}"
    r = gh_get(url)
    if r is None or r.status_code != 200:
        return None
    created = r.json().get("created_at")
    if created:
        return pd.to_datetime(created, utc=True, errors="coerce")
    return None


# ======================================================
# FAST COMMIT COUNT
# ======================================================
def count_commits(repo_fullname: str) -> int:
    """Count commits via contributors endpoint."""
    total = 0
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo_fullname}/contributors"
        r = gh_get(url, params={"anon": 1, "per_page": PER_PAGE, "page": page})
        if r is None:
            return 0
        if r.status_code == 404:
            return 0
        if r.status_code != 200:
            return 0

        data = r.json()
        if not data:
            break

        total += sum(c.get("contributions", 0) for c in data)
        page += 1

        if page > 50:  # safety cap
            break

    return total


# ======================================================
# PR VIABILITY (BEFORE/AFTER)
# ======================================================
def check_before_after_viability(repo: str, boundary: pd.Timestamp) -> Tuple[bool, Dict[str, int]]:
    """
    Check if a repo has enough PR + text activity before/after first agentic PR.
    """
    before_prs = after_prs = 0
    before_text = after_text = 0

    page = 1
    passed_boundary = False

    while True:
        url = f"https://api.github.com/repos/{repo}/pulls"
        r = gh_get(url, params={
            "state": "all",
            "sort": "created",
            "direction": "asc",
            "per_page": PER_PAGE,
            "page": page
        })

        if r is None or r.status_code != 200:
            return False, {}

        prs = r.json()
        if not prs:
            break

        for pr in prs:
            created_at = pd.to_datetime(pr.get("created_at"), utc=True, errors="coerce")
            if pd.isna(created_at):
                continue

            title = (pr.get("title") or "").strip()
            body  = (pr.get("body")  or "").strip()
            has_text = (len(title) + len(body)) > 0

            if created_at < boundary:
                before_prs += 1
                if has_text:
                    before_text += 1
            else:
                passed_boundary = True
                after_prs += 1
                if has_text:
                    after_text += 1

        if passed_boundary and \
           before_prs >= MIN_BEFORE_PRS and after_prs >= MIN_AFTER_PRS and \
           before_text >= MIN_BEFORE_TEXT and after_text >= MIN_AFTER_TEXT:
            return True, {
                "before_prs": before_prs, "after_prs": after_prs,
                "before_text": before_text, "after_text": after_text
            }

        page += 1
        if page > 200:
            break

    ok = (
        before_prs >= MIN_BEFORE_PRS and after_prs >= MIN_AFTER_PRS and
        before_text >= MIN_BEFORE_TEXT and after_text >= MIN_AFTER_TEXT
    )
    return ok, {
        "before_prs": before_prs, "after_prs": after_prs,
        "before_text": before_text, "after_text": after_text
    }


# ======================================================
# MAIN
# ======================================================
def main():
    print("\n=== Loading AIDEV dataset ===")
    pr_df   = pd.read_parquet(AIDEV_PR_PATH)
    repo_df = pd.read_parquet(AIDEV_REPO_PATH)

    AGENTIC = {"Claude_Code", "Copilot", "Cursor", "Devin", "OpenAI_Codex"}

    pr_df["repo"] = pr_df["repo_url"].astype(str).str.replace(
        "https://api.github.com/repos/", "", regex=False
    ).str.strip()

    pr_df["created_at"] = pd.to_datetime(pr_df["created_at"], utc=True, errors="coerce")

    agentic_prs = pr_df[pr_df["agent"].isin(AGENTIC)].copy()
    if agentic_prs.empty:
        raise ValueError("No agentic PRs in AIDEV dataset.")

    first_agentic = (
        agentic_prs.groupby("repo")["created_at"]
        .min()
        .reset_index()
        .rename(columns={"created_at": "first_agentic_date"})
    )

    candidates = first_agentic["repo"].dropna().unique().tolist()
    print(f"Found {len(candidates)} repos with agentic PRs")

    # ===================== STAR FILTER =====================
    repo_meta = repo_df[["full_name", "stars"]].copy()
    merged_meta = pd.DataFrame({"repo": candidates}).merge(
        repo_meta, left_on="repo", right_on="full_name", how="left"
    )

    merged_meta = merged_meta.dropna(subset=["stars"])
    merged_meta = merged_meta[merged_meta["stars"] >= MIN_STARS]

    print(f"After stars ≥ {MIN_STARS}: {len(merged_meta)} repos")

    candidates = merged_meta["repo"].tolist()

    # ===================== AGE FILTER ======================
    print("\n=== Checking repo age (via GitHub API) ===")

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=365 * MIN_REPO_AGE_YEARS)

    aged_candidates = []
    for repo in candidates:
        print(f"  Checking age for {repo}...")
        created_at = get_repo_creation_date(repo)
        if created_at is None:
            print("    - Could not fetch creation date (skip)")
            continue

        if created_at <= cutoff_date:
            print(f"    + OK (created {created_at.date()})")
            aged_candidates.append(repo)
        else:
            print(f"    - Too new (created {created_at.date()})")

    print(f"After age ≥ {MIN_REPO_AGE_YEARS} years: {len(aged_candidates)} repos")

    # ===================== SHUFFLE =====================
    random.shuffle(aged_candidates)

    # ===================== CHECK BEFORE/AFTER ACTIVITY =====================
    accepted = []
    debug_rows = []

    print("\n=== Checking PR/text viability ===")
    for repo in aged_candidates:
        if len(accepted) >= TARGET_REPOS:
            break

        boundary_row = first_agentic[first_agentic["repo"] == repo]
        if boundary_row.empty:
            continue

        boundary = boundary_row["first_agentic_date"].iloc[0]
        if pd.isna(boundary):
            continue

        print(f"[{len(accepted):02d}/{TARGET_REPOS}] {repo}")

        commits = count_commits(repo)
        if commits == 0:
            print("  - skip (no commit data)")
            continue
        if commits > MAX_COMMITS:
            print(f"  - skip (too many commits: {commits})")
            continue

        ok, stats = check_before_after_viability(repo, boundary)
        if not ok:
            print(
                f"  - skip (before_prs={stats.get('before_prs',0)}, "
                f"after_prs={stats.get('after_prs',0)}, "
                f"before_text={stats.get('before_text',0)}, "
                f"after_text={stats.get('after_text',0)})"
            )
            continue

        accepted.append(repo)
        debug_rows.append({
            "repo": repo,
            "commit_count": commits,
            **stats
        })
        print(f"  + ACCEPT (commits={commits}, {stats})")

    print(f"\n=== Final repo count: {len(accepted)} ===")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    pd.DataFrame({"repo": accepted}).to_csv(OUT_PATH, index=False)
    pd.DataFrame(debug_rows).to_csv("inputs/processed/final_repo_list_debug.csv", index=False)

    print(f"Saved final repo list to {OUT_PATH}")
    print("Saved debug stats to inputs/processed/final_repo_list_debug.csv\n")


if __name__ == "__main__":
    main()
