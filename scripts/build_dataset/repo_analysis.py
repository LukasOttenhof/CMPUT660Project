#!/usr/bin/env python3
"""
repo_analysis.py (HYBRID FIXED)

Centralized data-collection script for CMPUT660Project.

Fixes applied vs your current version:
1) Removed GitHub Search API usage entirely:
   - No more 403s from ancient date ranges or bot authors.
   - No more massive backoff loops.
   - PRs and Issues are collected by listing everything once, then filtering.

2) Commit author matching made robust:
   - First tries your heuristic matching.
   - If heuristic fails, resolves commit SHA -> GitHub login via API (cached).
   - This recovers agentic commits whose email/name doesn't match PR login.

3) Tracked developers kept consistent with your intent:
   - top N contributors
   - ANY PR author in the repo
   - (so bot/agent PR authors are always tracked)

Outputs:
- Same 20 parquets, same names, same columns.

Dependencies:
pip install PyGithub pydriller pandas pyarrow python-dotenv tqdm
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set

import pandas as pd
from tqdm import tqdm

# ---- load dotenv early ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from github import Github, Auth, GithubException
from pydriller import Repository


# =============================================================================
# CONFIG
# =============================================================================

REPO_LIST_TXT = "filtered_repos_3year100star.txt"
CLONE_DIR = "repos"
OUTPUT_DIR = "outputs"

START_DATE = datetime(1970, 1, 1, tzinfo=timezone.utc)
END_DATE   = datetime(2025, 12, 31, tzinfo=timezone.utc)

TOP_N = 5
PER_PAGE = 100

# Logging / stability
PYDRILLER_LOG_EVERY = 500
API_RETRIES = 8
SHORT_BACKOFF = 5
FULL_LIMIT_SLEEP = 60

MAX_COMMENTS_PER_THREAD = None  # None = no cap


# =============================================================================
# TOKEN POOL (3 tokens, rotation & safe calls)
# =============================================================================

class TokenPoolGithub:
    def __init__(self, tokens: List[str]):
        tokens = [t.strip() for t in tokens if t and t.strip()]
        if len(tokens) != 3:
            raise RuntimeError(
                f"Expected exactly 3 tokens from env (GITHUB_TOKEN_1..3). Got {len(tokens)}."
            )
        self.tokens = tokens
        self.i = 0
        self.gh = self._make_client(self.tokens[self.i])

    def _make_client(self, token: str) -> Github:
        return Github(auth=Auth.Token(token), per_page=PER_PAGE)

    def rotate(self):
        old = self.i
        self.i = (self.i + 1) % len(self.tokens)
        self.gh = self._make_client(self.tokens[self.i])
        print(f"🔄 Rotating token {old} → {self.i}")

    def safe(self, func, *args, label: str = "", **kwargs):
        """
        Safe wrapper around any PyGithub call.
        - rotates token on 401/403 or rate/secondary limit
        - retries a few times
        """
        for attempt in range(1, API_RETRIES + 1):
            try:
                return func(*args, **kwargs)

            except GithubException as e:
                status = getattr(e, "status", None)
                msg = ""
                try:
                    if isinstance(e.data, dict):
                        msg = (e.data.get("message") or "").lower()
                except Exception:
                    pass

                if status in (401, 403) or "rate limit" in msg or "secondary rate limit" in msg:
                    rotated = False
                    for _ in range(len(self.tokens)):
                        rem = None
                        try:
                            rem, _ = self.gh.rate_limiting
                        except Exception:
                            rem = None

                        if rem is None or rem > 0:
                            break

                        self.rotate()
                        rotated = True

                    if rotated:
                        continue

                    print(f"⏳ All tokens exhausted. Sleeping {FULL_LIMIT_SLEEP}s...")
                    time.sleep(FULL_LIMIT_SLEEP)
                    continue

                print(f"⚠️ [{label}] HTTP {status} attempt={attempt}/{API_RETRIES}. Retrying...")
                time.sleep(SHORT_BACKOFF)

        print(f"❌ Giving up after retries [{label}]")
        raise RuntimeError(f"Giving up after retries [{label}]")


def load_tokens_from_env() -> TokenPoolGithub:
    t1 = os.getenv("GITHUB_TOKEN_1", "").strip()
    t2 = os.getenv("GITHUB_TOKEN_2", "").strip()
    t3 = os.getenv("GITHUB_TOKEN_3", "").strip()
    return TokenPoolGithub([t1, t2, t3])


# =============================================================================
# REPO LIST PARSING
# =============================================================================

@dataclass
class RepoSpec:
    full_name: str
    boundary_date: datetime
    created_at: Optional[datetime] = None
    url: Optional[str] = None


def parse_repo_list(path: str) -> List[RepoSpec]:
    specs: List[RepoSpec] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]
            full_name = parts[0]

            boundary = None
            created = None
            url = None
            for p in parts[1:]:
                if p.startswith("http"):
                    url = p
                if "first agent pr" in p.lower():
                    boundary_str = p.split(":")[-1].strip()
                    boundary = pd.to_datetime(boundary_str, utc=True, errors="coerce")
                if "repo creation" in p.lower():
                    created_str = p.split(":")[-1].strip()
                    created = pd.to_datetime(created_str, utc=True, errors="coerce")

            if boundary is None or pd.isna(boundary):
                print(f"⚠️ Skipping malformed line (no boundary): {line}")
                continue

            specs.append(
                RepoSpec(
                    full_name=full_name,
                    boundary_date=boundary.to_pydatetime(),
                    created_at=None if created is None or pd.isna(created) else created.to_pydatetime(),
                    url=url,
                )
            )
    return specs


# =============================================================================
# CLONING
# =============================================================================

def safe_repo_dirname(full_name: str) -> str:
    return full_name.replace("/", "_")

def ensure_clone(full_name: str) -> str:
    os.makedirs(CLONE_DIR, exist_ok=True)
    repo_dir = os.path.join(CLONE_DIR, safe_repo_dirname(full_name))

    if os.path.isdir(repo_dir) and os.path.isdir(os.path.join(repo_dir, ".git")):
        print(f"📁 Using existing clone: {repo_dir}")
        return repo_dir

    print(f"📥 Cloning {full_name} → {repo_dir}")
    url = f"https://github.com/{full_name}.git"

    subprocess.run(["git", "clone", "--quiet", url, repo_dir], check=True)
    return repo_dir


# =============================================================================
# TRACKED DEVELOPERS (top 5 + ALL PR authors)
# =============================================================================

def get_top_contributors(tp: TokenPoolGithub, repo) -> List[str]:
    print("👥 Fetching contributors…")
    contribs = tp.safe(repo.get_contributors, label="get_contributors")
    lst = list(contribs)
    lst.sort(key=lambda c: getattr(c, "contributions", 0), reverse=True)
    top = [c.login for c in lst[:TOP_N] if getattr(c, "login", None)]
    print(f"👥 Top {TOP_N} contributors: {top}")
    return top

def get_all_pr_authors(tp: TokenPoolGithub, repo) -> Set[str]:
    """
    List ALL PRs once, collect all authors.
    (No Search API, so no 403 explosions.)
    """
    authors: Set[str] = set()
    pulls = tp.safe(
        repo.get_pulls,
        state="all",
        sort="created",
        direction="asc",
        label="get_pulls_for_authors"
    )
    for pr in pulls:
        if pr.user and pr.user.login:
            authors.add(pr.user.login)
    return authors

def build_tracked_users(tp: TokenPoolGithub, repo, top_users: List[str]) -> List[str]:
    pr_authors = get_all_pr_authors(tp, repo)
    tracked = sorted(set(top_users) | pr_authors)
    print(f"👥 Tracked users count: {len(tracked)} (top + PR authors)")
    return tracked


# =============================================================================
# AUTHOR MATCHING HEURISTICS (kept, but now fallback to SHA->login)
# =============================================================================

def matches_login(author_name: str, author_email: str, login: str) -> bool:
    l = login.lower().strip()
    n = (author_name or "").lower().strip()
    e = (author_email or "").lower().strip()

    if not l:
        return False
    if n == l:
        return True
    if l in n and len(l) >= 3:
        return True
    if e:
        local = e.split("@")[0]
        if local == l:
            return True
        if l in local and len(l) >= 3:
            return True
    return False

def heuristic_match_login(author_name: str, author_email: str, tracked_users: List[str]) -> Optional[str]:
    """
    Return matching login if heuristics find one, else None.
    """
    for u in tracked_users:
        if matches_login(author_name, author_email, u):
            return u
    return None

def canonical_author(author_name: str, author_email: str, matched_login: Optional[str]) -> str:
    if matched_login:
        return matched_login
    if author_name:
        return author_name.strip()
    if author_email:
        return author_email.strip()
    return "unknown"


def prefetch_tracked_commit_shas(tp: TokenPoolGithub, repo, tracked_users: List[str]) -> Dict[str, Set[str]]:
    """
    Pre-fetches commit SHAs for each tracked GitHub login.
    This avoids calling repo.get_commit() for every PyDriller commit.
    """
    commits_by_author: Dict[str, Set[str]] = {u: set() for u in tracked_users}

    print("🔍 Prefetching commit SHAs for tracked users…")

    for login in tracked_users:
        try:
            # GitHub supports filtering commits by author login
            gh_commits = tp.safe(repo.get_commits, author=login, label=f"prefetch_commits:{login}")
            for c in gh_commits:
                commits_by_author[login].add(c.sha)
        except Exception as e:
            print(f"⚠️ Could not prefetch commits for author {login}: {e}")

    total = sum(len(v) for v in commits_by_author.values())
    print(f"   → Prefetched {total} SHAs for {len(tracked_users)} authors.")
    return commits_by_author


# =============================================================================
# PYDRILLER COMMITS (numeric + text) with SHA->login fallback
# =============================================================================

def extract_commits_from_clone(
    tp: TokenPoolGithub,
    repo_api,
    repo_dir: str,
    repo_name: str,
    boundary: datetime,
    tracked_users: List[str],
    numeric_before: List[Dict[str, Any]],
    numeric_after: List[Dict[str, Any]],
    text_before: List[Dict[str, Any]],
    text_after: List[Dict[str, Any]],
):
    print("📊 PyDriller: commits (FAST mode)")

    # Pre-fetch all commits by tracked users to avoid expensive SHA lookups
    tracked_shas = prefetch_tracked_commit_shas(tp, repo_api, tracked_users)

    processed = 0
    kept_before = 0
    kept_after = 0

    for commit in Repository(repo_dir).traverse_commits():
        processed += 1
        if processed % 500 == 0:
            print(f"   …PyDriller processed {processed} commits so far")

        dt = commit.author_date
        if not dt:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        if dt < START_DATE or dt > END_DATE:
            continue

        sha = commit.hash
        author_name = (commit.author.name or "").strip()
        author_email = (commit.author.email or "").strip()

        # Skip obvious bots early (saves tons of checks)
        if "bot" in (author_name + author_email).lower():
            continue

        # --- 1. Fast path: SHA found in prefetch lists ---
        matched_login = None
        for login, sha_set in tracked_shas.items():
            if sha in sha_set:
                matched_login = login
                break

        # --- 2. Heuristic match (cheap local check) ---
        if matched_login is None:
            matched_login = heuristic_match_login(author_name, author_email, tracked_users)

        # --- 3. Otherwise skip: not tracked
        if matched_login is None:
            continue

        # Build canonical author label
        author = canonical_author(author_name, author_email, matched_login)

        # Numeric record
        rec_num = {
            "repo": repo_name,
            "author": author,
            "date": dt,
            "sha": sha,
            "loc_added": commit.insertions,
            "loc_deleted": commit.deletions,
            "files_changed": commit.files,
        }

        # Text record
        rec_txt = {
            "repo": repo_name,
            "author": author,
            "date": dt,
            "source": "commit",
            "text": (commit.msg or "").strip(),
            "sha": sha,
        }

        # Split into before/after
        if dt < boundary:
            numeric_before.append(rec_num)
            text_before.append(rec_txt)
            kept_before += 1
        else:
            numeric_after.append(rec_num)
            text_after.append(rec_txt)
            kept_after += 1

    print(f"   ✅ commits kept: before={kept_before} after={kept_after} "
          f"({kept_before+kept_after} total)")



# =============================================================================
# PRs + REVIEWS (LIST ONCE, FILTER)
# =============================================================================

def extract_prs_reviews_comments(
    tp: TokenPoolGithub,
    repo,
    repo_name: str,
    boundary: datetime,
    tracked_users: List[str],
    numeric_before_prs: List[Dict[str, Any]],
    numeric_after_prs: List[Dict[str, Any]],
    numeric_before_reviews: List[Dict[str, Any]],
    numeric_after_reviews: List[Dict[str, Any]],
    text_before_pr_bodies: List[Dict[str, Any]],
    text_after_pr_bodies: List[Dict[str, Any]],
    text_before_review_comments: List[Dict[str, Any]],
    text_after_review_comments: List[Dict[str, Any]],
):
    print("📥 API: PRs + reviews (list-all, filter tracked)")

    pulls = tp.safe(
        repo.get_pulls,
        state="all",
        sort="created",
        direction="asc",
        label="get_all_pulls_for_prs"
    )
    pulls = list(pulls)
    print(f"📦 Total PRs in repo: {len(pulls)}")

    tracked_pulls = [
        pr for pr in pulls
        if pr.user and pr.user.login in tracked_users
    ]
    print(f"📦 PRs by tracked authors: {len(tracked_pulls)}")

    for pr in tqdm(tracked_pulls, desc="PRs (tracked authors)"):
        try:
            created_at = pr.created_at.replace(tzinfo=timezone.utc)
            if created_at < START_DATE or created_at > END_DATE:
                continue

            author_login = pr.user.login if pr.user else "unknown"

            pr_num_rec = {
                "repo": repo_name,
                "author": author_login,
                "date": created_at,
                "activity_type": "pr_created",
                "pr_number": pr.number,
            }

            pr_text = (pr.title or "") + "\n" + (pr.body or "")
            pr_txt_rec = {
                "repo": repo_name,
                "author": author_login,
                "date": created_at,
                "source": "pull_request",
                "text": pr_text.strip(),
                "pr_number": pr.number,
            }

            pr_merge_rec = None
            if pr.merged and pr.merged_at:
                merged_at = pr.merged_at.replace(tzinfo=timezone.utc)
                if START_DATE <= merged_at <= END_DATE:
                    delta_h = (merged_at - created_at).total_seconds() / 3600
                    pr_merge_rec = {
                        "repo": repo_name,
                        "author": author_login,
                        "date": merged_at,
                        "activity_type": "pr_merged",
                        "pr_number": pr.number,
                        "time_to_merge_hours": delta_h,
                    }

            if created_at < boundary:
                numeric_before_prs.append(pr_num_rec)
                text_before_pr_bodies.append(pr_txt_rec)
            else:
                numeric_after_prs.append(pr_num_rec)
                text_after_pr_bodies.append(pr_txt_rec)

            if pr_merge_rec:
                if pr_merge_rec["date"] < boundary:
                    numeric_before_prs.append(pr_merge_rec)
                else:
                    numeric_after_prs.append(pr_merge_rec)

            # ---- reviews (numeric + text) ----
            try:
                reviews = tp.safe(pr.get_reviews, label="get_reviews")
                for r in reviews:
                    if not r.user or not r.submitted_at:
                        continue
                    reviewer = r.user.login
                    if reviewer not in tracked_users:
                        continue

                    submitted = r.submitted_at.replace(tzinfo=timezone.utc)
                    if submitted < START_DATE or submitted > END_DATE:
                        continue

                    num_rev_rec = {
                        "repo": repo_name,
                        "author": reviewer,
                        "date": submitted,
                        "activity_type": "review_submitted",
                        "pr_number": pr.number,
                        "state": r.state,
                    }
                    txt_rev_rec = {
                        "repo": repo_name,
                        "author": reviewer,
                        "date": submitted,
                        "source": "pr_review",
                        "text": (r.body or "").strip(),
                        "pr_number": pr.number,
                        "state": r.state,
                    }

                    if submitted < boundary:
                        numeric_before_reviews.append(num_rev_rec)
                        text_before_review_comments.append(txt_rev_rec)
                    else:
                        numeric_after_reviews.append(num_rev_rec)
                        text_after_review_comments.append(txt_rev_rec)
            except Exception:
                pass

        except Exception as e:
            print(f"⚠️ PR {getattr(pr, 'number', '?')} failed: {e}")
            continue


# =============================================================================
# ISSUES + COMMENTS (LIST ONCE, FILTER)
# =============================================================================

def extract_issues_and_comments(
    tp: TokenPoolGithub,
    repo,
    repo_name: str,
    boundary: datetime,
    tracked_users: List[str],
    numeric_before_issues: List[Dict[str, Any]],
    numeric_after_issues: List[Dict[str, Any]],
    text_before_issue_bodies: List[Dict[str, Any]],
    text_after_issue_bodies: List[Dict[str, Any]],
):
    print("📥 API: issues + comments (list-all, filter tracked)")

    # list all issues once; GitHub "issues" API includes PRs so we filter them out
    all_issues = tp.safe(
        repo.get_issues,
        state="all",
        since=repo.created_at or START_DATE,
        label="get_all_issues"
    )
    all_issues = list(all_issues)

    issues = [
        iss for iss in all_issues
        if iss.pull_request is None
        and iss.user
        and iss.user.login in tracked_users
    ]

    print(f"📦 Total issues in repo (non-PR): {len([i for i in all_issues if i.pull_request is None])}")
    print(f"📦 Issues by tracked authors: {len(issues)}")

    for issue in tqdm(issues, desc="Issues (tracked authors)"):
        try:
            created_at = issue.created_at.replace(tzinfo=timezone.utc)
            if created_at < START_DATE or created_at > END_DATE:
                continue

            author_login = issue.user.login if issue.user else "unknown"

            num_open = {
                "repo": repo_name,
                "author": author_login,
                "date": created_at,
                "activity_type": "issue_opened",
                "issue_number": issue.number,
            }
            txt_body = (issue.title or "") + "\n" + (issue.body or "")
            txt_open = {
                "repo": repo_name,
                "author": author_login,
                "date": created_at,
                "source": "issue",
                "text": txt_body.strip(),
                "issue_number": issue.number,
            }

            if created_at < boundary:
                numeric_before_issues.append(num_open)
                text_before_issue_bodies.append(txt_open)
            else:
                numeric_after_issues.append(num_open)
                text_after_issue_bodies.append(txt_open)

            if issue.closed_at:
                closed_at = issue.closed_at.replace(tzinfo=timezone.utc)
                if START_DATE <= closed_at <= END_DATE:
                    delta_h = (closed_at - created_at).total_seconds() / 3600
                    num_close = {
                        "repo": repo_name,
                        "author": author_login,
                        "date": closed_at,
                        "activity_type": "issue_closed",
                        "issue_number": issue.number,
                        "time_to_close_hours": delta_h,
                    }
                    if closed_at < boundary:
                        numeric_before_issues.append(num_close)
                    else:
                        numeric_after_issues.append(num_close)

            # Issue comments by tracked devs
            try:
                comments = tp.safe(issue.get_comments, label="get_issue_comments")
                count = 0
                for c in comments:
                    if MAX_COMMENTS_PER_THREAD is not None and count >= MAX_COMMENTS_PER_THREAD:
                        break
                    if not c.user or not c.created_at:
                        continue
                    commenter = c.user.login
                    if commenter not in tracked_users:
                        continue
                    dt = c.created_at.replace(tzinfo=timezone.utc)
                    if dt < START_DATE or dt > END_DATE:
                        continue
                    txt = {
                        "repo": repo_name,
                        "author": commenter,
                        "date": dt,
                        "source": "issue_comment",
                        "text": (c.body or "").strip(),
                        "issue_number": issue.number,
                    }
                    if dt < boundary:
                        text_before_issue_bodies.append(txt)
                    else:
                        text_after_issue_bodies.append(txt)
                    count += 1
            except Exception:
                pass

        except Exception as e:
            print(f"⚠️ issue {getattr(issue, 'number', '?')} failed: {e}")
            continue


# =============================================================================
# DISCUSSIONS (optional)
# =============================================================================

def extract_discussions(
    tp: TokenPoolGithub,
    repo,
    repo_name: str,
    boundary: datetime,
    tracked_users: List[str],
    text_before_topics: List[Dict[str, Any]],
    text_after_topics: List[Dict[str, Any]],
    text_before_comments: List[Dict[str, Any]],
    text_after_comments: List[Dict[str, Any]],
):
    print("📥 API: discussions (optional)")
    try:
        discussions = tp.safe(repo.get_discussions, label="get_discussions")
    except Exception:
        print("⚠️ Discussions not enabled or inaccessible. (normal)")
        return

    for d in tqdm(discussions, desc="Discussions"):
        try:
            if not d.user or not d.created_at:
                continue
            author = d.user.login
            if author not in tracked_users:
                continue

            dt = d.created_at.replace(tzinfo=timezone.utc)
            if dt < START_DATE or dt > END_DATE:
                continue

            txt = (d.title or "") + "\n" + (d.body or "")
            rec = {
                "repo": repo_name,
                "author": author,
                "date": dt,
                "source": "discussion",
                "text": txt.strip(),
                "discussion_number": d.number,
            }
            if dt < boundary:
                text_before_topics.append(rec)
            else:
                text_after_topics.append(rec)

            try:
                comments = tp.safe(d.get_comments, label="get_discussion_comments")
                count = 0
                for c in comments:
                    if MAX_COMMENTS_PER_THREAD is not None and count >= MAX_COMMENTS_PER_THREAD:
                        break
                    if not c.user or not c.created_at:
                        continue
                    commenter = c.user.login
                    if commenter not in tracked_users:
                        continue
                    cdt = c.created_at.replace(tzinfo=timezone.utc)
                    if cdt < START_DATE or cdt > END_DATE:
                        continue
                    crec = {
                        "repo": repo_name,
                        "author": commenter,
                        "date": cdt,
                        "source": "discussion_comment",
                        "text": (c.body or "").strip(),
                        "discussion_number": d.number,
                    }
                    if cdt < boundary:
                        text_before_comments.append(crec)
                    else:
                        text_after_comments.append(crec)
                    count += 1
            except Exception:
                pass

        except Exception:
            continue


# =============================================================================
# MAIN
# =============================================================================

def main():
    tp = load_tokens_from_env()
    specs = parse_repo_list(REPO_LIST_TXT)

    print(f"📦 Loaded {len(specs)} repos.\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 20 parquet buckets
    commits_before: List[Dict[str, Any]] = []
    commits_after:  List[Dict[str, Any]] = []

    prs_before: List[Dict[str, Any]] = []
    prs_after:  List[Dict[str, Any]] = []

    issues_before: List[Dict[str, Any]] = []
    issues_after:  List[Dict[str, Any]] = []

    reviews_before: List[Dict[str, Any]] = []
    reviews_after:  List[Dict[str, Any]] = []

    commit_msgs_before: List[Dict[str, Any]] = []
    commit_msgs_after:  List[Dict[str, Any]] = []

    pr_bodies_before: List[Dict[str, Any]] = []
    pr_bodies_after:  List[Dict[str, Any]] = []

    issue_bodies_before: List[Dict[str, Any]] = []
    issue_bodies_after:  List[Dict[str, Any]] = []

    review_comments_before: List[Dict[str, Any]] = []
    review_comments_after:  List[Dict[str, Any]] = []

    discussion_topics_before: List[Dict[str, Any]] = []
    discussion_topics_after:  List[Dict[str, Any]] = []

    discussion_comments_before: List[Dict[str, Any]] = []
    discussion_comments_after:  List[Dict[str, Any]] = []

    for idx, spec in enumerate(specs, start=1):
        repo_name = spec.full_name
        boundary = spec.boundary_date

        print("\n" + "="*80)
        print(f"[{idx}/{len(specs)}] 🔍 Processing {repo_name} | boundary = {boundary.date()}")
        print("="*80)

        try:
            repo_dir = ensure_clone(repo_name)
        except Exception as e:
            print(f"⚠️ Skipping due to clone failure: {e}")
            continue

        try:
            repo = tp.safe(tp.gh.get_repo, repo_name, label="get_repo")
        except Exception as e:
            print(f"⚠️ Skipping {repo_name} due to API repo failure: {e}")
            continue

        # ===================================================================
        # INSERT SIZE LIMITS HERE (right after repo = ...)
        # ===================================================================

        # 1. Commit count
        try:
            commit_count = tp.safe(repo.get_commits, label="count_commits").totalCount
        except Exception:
            commit_count = None

        if commit_count and commit_count > 5000:
            print(f"⚠️ SKIPPING {repo_name}: too many commits ({commit_count} > 5000).")
            continue

        # 2. PR count
        try:
            pr_count = tp.safe(repo.get_pulls, state="all", label="count_prs").totalCount
        except Exception:
            pr_count = None

        if pr_count and pr_count > 1200:
            print(f"⚠️ SKIPPING {repo_name}: too many PRs ({pr_count} > 1000).")
            continue

        # ===================================================================
        # END OF SIZE LIMIT BLOCK
        # ===================================================================

        try:
            top_users = get_top_contributors(tp, repo)
            tracked_users = build_tracked_users(tp, repo, top_users)
        except Exception as e:
            print(f"⚠️ Skipping {repo_name} due to contributor failure: {e}")
            continue

        try:
            extract_commits_from_clone(
                tp, repo,
                repo_dir, repo_name, boundary, tracked_users,
                commits_before, commits_after,
                commit_msgs_before, commit_msgs_after,
            )
        except Exception as e:
            print(f"⚠️ Commit extraction failed for {repo_name}: {e}")

        try:
            extract_prs_reviews_comments(
                tp, repo, repo_name, boundary, tracked_users,
                prs_before, prs_after,
                reviews_before, reviews_after,
                pr_bodies_before, pr_bodies_after,
                review_comments_before, review_comments_after,
            )
        except Exception as e:
            print(f"⚠️ PR extraction failed for {repo_name}: {e}")

        try:
            extract_issues_and_comments(
                tp, repo, repo_name, boundary, tracked_users,
                issues_before, issues_after,
                issue_bodies_before, issue_bodies_after,
            )
        except Exception as e:
            print(f"⚠️ Issue extraction failed for {repo_name}: {e}")

        try:
            extract_discussions(
                tp, repo, repo_name, boundary, tracked_users,
                discussion_topics_before, discussion_topics_after,
                discussion_comments_before, discussion_comments_after,
            )
        except Exception:
            pass

        print(f"✔ Finished {repo_name}")


    # ---- write outputs ----
    def to_parquet(rows: List[Dict[str, Any]], path: str):
        df = pd.DataFrame(rows)
        if len(df) == 0:
            df = pd.DataFrame(columns=["repo", "author", "date"])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df = df.dropna(subset=["date"])
            df = df.sort_values("date")
        df.to_parquet(path, index=False)

    print("\n==============================")
    print("💾 Writing 20 parquet outputs…")
    print("==============================")

    to_parquet(commits_before, os.path.join(OUTPUT_DIR, "commits_before.parquet"))
    to_parquet(commits_after,  os.path.join(OUTPUT_DIR, "commits_after.parquet"))

    to_parquet(prs_before, os.path.join(OUTPUT_DIR, "pull_requests_before.parquet"))
    to_parquet(prs_after,  os.path.join(OUTPUT_DIR, "pull_requests_after.parquet"))

    to_parquet(issues_before, os.path.join(OUTPUT_DIR, "issues_before.parquet"))
    to_parquet(issues_after,  os.path.join(OUTPUT_DIR, "issues_after.parquet"))

    to_parquet(reviews_before, os.path.join(OUTPUT_DIR, "reviews_before.parquet"))
    to_parquet(reviews_after,  os.path.join(OUTPUT_DIR, "reviews_after.parquet"))

    to_parquet(commit_msgs_before, os.path.join(OUTPUT_DIR, "commit_messages_before.parquet"))
    to_parquet(commit_msgs_after,  os.path.join(OUTPUT_DIR, "commit_messages_after.parquet"))

    to_parquet(pr_bodies_before, os.path.join(OUTPUT_DIR, "pr_bodies_before.parquet"))
    to_parquet(pr_bodies_after,  os.path.join(OUTPUT_DIR, "pr_bodies_after.parquet"))

    to_parquet(issue_bodies_before, os.path.join(OUTPUT_DIR, "issue_bodies_before.parquet"))
    to_parquet(issue_bodies_after,  os.path.join(OUTPUT_DIR, "issue_bodies_after.parquet"))

    to_parquet(review_comments_before, os.path.join(OUTPUT_DIR, "review_comments_before.parquet"))
    to_parquet(review_comments_after,  os.path.join(OUTPUT_DIR, "review_comments_after.parquet"))

    to_parquet(discussion_topics_before, os.path.join(OUTPUT_DIR, "discussion_topics_before.parquet"))
    to_parquet(discussion_topics_after,  os.path.join(OUTPUT_DIR, "discussion_topics_after.parquet"))

    to_parquet(discussion_comments_before, os.path.join(OUTPUT_DIR, "discussion_comments_before.parquet"))
    to_parquet(discussion_comments_after,  os.path.join(OUTPUT_DIR, "discussion_comments_after.parquet"))

    print("🎉 Done. Outputs are in ./outputs/")


if __name__ == "__main__":
    main()
