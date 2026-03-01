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

#load dotenv early
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from github import Github, Auth, GithubException
from pydriller import Repository

# REPO_LIST_TXT = r"G:\CMPUT660Project\filtered_repos_3year50pr.txt"
# REPO_LIST_TXT = r"G:\CMPUT660Project\scripts\build_dataset\true_human_baseline.txt"
REPO_LIST_TXT = r"G:\CMPUT660Project\scripts\build_dataset\true_human_baseline_con_reformatted.txt"
CLONE_DIR = r"G:\CMPUT660Project\repos_human_2"
OUTPUT_DIR = r"inputs/human_2"

START_DATE = datetime(1970, 1, 1, tzinfo=timezone.utc)
END_DATE   = datetime(2025, 12, 31, tzinfo=timezone.utc)

TOP_N = 5
PER_PAGE = 100

#Logging/stability
PYDRILLER_LOG_EVERY = 500
API_RETRIES = 8
SHORT_BACKOFF = 5
FULL_LIMIT_SLEEP = 60

MAX_COMMENTS_PER_THREAD = None

#Rotates 3 tokens for large github API querying

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
        print(f"Rotating token {old} → {self.i}")

    def safe(self, func, *args, label: str = "", **kwargs):
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

                    print(f"All tokens exhausted. Sleeping {FULL_LIMIT_SLEEP}s...")
                    time.sleep(FULL_LIMIT_SLEEP)
                    continue

                print(f"[{label}] HTTP {status} attempt={attempt}/{API_RETRIES}. Retrying...")
                time.sleep(SHORT_BACKOFF)

        print(f"Giving up after retries [{label}]")
        raise RuntimeError(f"Giving up after retries [{label}]")


def load_tokens_from_env() -> TokenPoolGithub:
    t1 = os.getenv("GITHUB_TOKEN_1", "").strip()
    t2 = os.getenv("GITHUB_TOKEN_2", "").strip()
    t3 = os.getenv("GITHUB_TOKEN_3", "").strip()
    return TokenPoolGithub([t1, t2, t3])

#Parse repo list
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
                print(f"Skipping malformed line (no boundary): {line}")
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


#Clone repos
def safe_repo_dirname(full_name: str) -> str:
    return full_name.replace("/", "_")

def ensure_clone(full_name: str) -> str:
    os.makedirs(CLONE_DIR, exist_ok=True)
    repo_dir = os.path.join(CLONE_DIR, safe_repo_dirname(full_name))

    if os.path.isdir(repo_dir) and os.path.isdir(os.path.join(repo_dir, ".git")):
        print(f"Using existing clone: {repo_dir}")
        return repo_dir

    print(f"Cloning {full_name} → {repo_dir}")
    url = f"https://github.com/{full_name}.git"

    subprocess.run(["git", "clone", "--quiet", url, repo_dir], check=True)
    return repo_dir


#Top 5 authors per repo

def get_top_contributors(tp: TokenPoolGithub, repo) -> List[str]:
    print("Fetching contributors…")
    contribs = tp.safe(repo.get_contributors, label="get_contributors")
    lst = list(contribs)
    lst.sort(key=lambda c: getattr(c, "contributions", 0), reverse=True)
    top = [c.login for c in lst[:TOP_N] if getattr(c, "login", None)]
    print(f"Top {TOP_N} contributors: {top}")
    return top

def get_all_pr_authors(tp: TokenPoolGithub, repo) -> Set[str]:
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
    print(f"Tracked users count: {len(tracked)} (top + PR authors)")
    return tracked


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
    commits_by_author: Dict[str, Set[str]] = {u: set() for u in tracked_users}

    print("Prefetching commit SHAs for tracked users…")

    for login in tracked_users:
        try:
            # GitHub supports filtering commits by author login
            gh_commits = tp.safe(repo.get_commits, author=login, label=f"prefetch_commits:{login}")
            for c in gh_commits:
                commits_by_author[login].add(c.sha)
        except Exception as e:
            print(f"Could not prefetch commits for author {login}: {e}")

    total = sum(len(v) for v in commits_by_author.values())
    print(f"   → Prefetched {total} SHAs for {len(tracked_users)} authors.")
    return commits_by_author


#Pydriller commit handling
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
    print("PyDriller: commits (FAST mode)")

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

        #1. Fast path: SHA found in prefetch lists
        matched_login = None
        for login, sha_set in tracked_shas.items():
            if sha in sha_set:
                matched_login = login
                break

        #2. Heuristic match (cheap local check)
        if matched_login is None:
            matched_login = heuristic_match_login(author_name, author_email, tracked_users)

        #3. Otherwise skip: not tracked
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



#PRs + Reviews

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

            #reviews (numeric + text)
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


#Issues + Comments

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


# Discussions (these are broken/inaccessible)

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
    skipped = []
    
    # Setup directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    INC_DIR = os.path.join(OUTPUT_DIR, "incremental")
    os.makedirs(INC_DIR, exist_ok=True)

    print(f"📦 Loaded {len(specs)} repos.\n")

    # Mapping of variable names to their final filenames
    data_map = {
        "commits_before": "commits_before.parquet",
        "commits_after": "commits_after.parquet",
        "prs_before": "pull_requests_before.parquet",
        "prs_after": "pull_requests_after.parquet",
        "issues_before": "issues_before.parquet",
        "issues_after": "issues_after.parquet",
        "reviews_before": "reviews_before.parquet",
        "reviews_after": "reviews_after.parquet",
        "commit_msgs_before": "commit_messages_before.parquet",
        "commit_msgs_after": "commit_messages_after.parquet",
        "pr_bodies_before": "pr_bodies_before.parquet",
        "pr_bodies_after": "pr_bodies_after.parquet",
        "issue_bodies_before": "issue_bodies_before.parquet",
        "issue_bodies_after": "issue_bodies_after.parquet",
        "review_comments_before": "review_comments_before.parquet",
        "review_comments_after": "review_comments_after.parquet",
        "discussion_topics_before": "discussion_topics_before.parquet",
        "discussion_topics_after": "discussion_topics_after.parquet",
        "discussion_comments_before": "discussion_comments_before.parquet",
        "discussion_comments_after": "discussion_comments_after.parquet",
    }

    # for idx, spec in enumerate(specs, start=1):
    #     repo_name = spec.full_name
    #     boundary = spec.boundary_date
        
    #     # --- CRASH RECOVERY CHECK ---
    #     # We create a small sentinel file to mark a repo as "done"
    #     sentinel = os.path.join(INC_DIR, f"{repo_name.replace('/', '_')}.done")
    #     if os.path.exists(sentinel):
    #         print(f"⏩ Skipping {repo_name} (already processed).")
    #         continue

    #     print("\n" + "="*80)
    #     print(f"[{idx}/{len(specs)}] 🔍 Processing {repo_name} | boundary = {boundary.date()}")
    #     print("="*80)

    #     # Initialize temporary local buckets for THIS repo only
    #     buckets = {key: [] for key in data_map.keys()}

    #     try:
    #         repo_dir = ensure_clone(repo_name)
    #         subprocess.run(["git", "config", "--global", "--add", "safe.directory", repo_dir.replace("\\", "/")], check=False)
    #         repo = tp.safe(tp.gh.get_repo, repo_name, label="get_repo")
            
    #         # Size limits
    #         commit_count = 0
    #         try: commit_count = tp.safe(repo.get_commits, label="count_commits").totalCount
    #         except: pass
            
    #         pr_count = 0
    #         try: pr_count = tp.safe(repo.get_pulls, state="all", label="count_prs").totalCount
    #         except: pass

    #         if (commit_count > 10000) or (pr_count > 1000):
    #             print(f"⚠️ SKIPPING {repo_name}: too large.")
    #             skipped.append(repo_name)
    #             continue

    #         top_users = get_top_contributors(tp, repo)
    #         tracked_users = build_tracked_users(tp, repo, top_users)

    #         # --- EXTRACTION ---
    #         extract_commits_from_clone(
    #             tp, repo, repo_dir, repo_name, boundary, tracked_users,
    #             buckets["commits_before"], buckets["commits_after"],
    #             buckets["commit_msgs_before"], buckets["commit_msgs_after"]
    #         )
    #         extract_prs_reviews_comments(
    #             tp, repo, repo_name, boundary, tracked_users,
    #             buckets["prs_before"], buckets["prs_after"],
    #             buckets["reviews_before"], buckets["reviews_after"],
    #             buckets["pr_bodies_before"], buckets["pr_bodies_after"],
    #             buckets["review_comments_before"], buckets["review_comments_after"]
    #         )
    #         extract_issues_and_comments(
    #             tp, repo, repo_name, boundary, tracked_users,
    #             buckets["issues_before"], buckets["issues_after"],
    #             buckets["issue_bodies_before"], buckets["issue_bodies_after"]
    #         )
    #         try:
    #             extract_discussions(
    #                 tp, repo, repo_name, boundary, tracked_users,
    #                 buckets["discussion_topics_before"], buckets["discussion_topics_after"],
    #                 buckets["discussion_comments_before"], buckets["discussion_comments_after"]
    #             )
    #         except: pass

    #         # --- INCREMENTAL SAVE ---
    #         # Save this repo's data to a unique parquet file in the incremental folder
    #         safe_name = repo_name.replace("/", "_")
    #         for key, rows in buckets.items():
    #             if rows:
    #                 inc_path = os.path.join(INC_DIR, f"{safe_name}_{key}.parquet")
    #                 pd.DataFrame(rows).to_parquet(inc_path, index=False)

    #         # Create sentinel file
    #         with open(sentinel, "w") as f: f.write("done")
    #         print(f"✔ Finished and saved {repo_name}")

    #     except Exception as e:
    #         print(f"💥 Failed processing {repo_name}: {e}")
    #         continue

    # ===================================================================
    # FINAL CONSOLIDATION
    # ===================================================================
   # ===================================================================
    # FINAL CONSOLIDATION
    # ===================================================================
    print("\n" + "="*30)
    print("Consolidating all data into final Parquets…")
    print("="*30)

    def finalize_parquet(key, final_filename):
        all_dfs = []
        files = [f for f in os.listdir(INC_DIR) if f.endswith(f"_{key}.parquet")]
        
        for file in files:
            file_path = os.path.join(INC_DIR, file)
            tmp_df = None
            
            # ATTEMPT 1: Default (PyArrow)
            try:
                tmp_df = pd.read_parquet(file_path, engine='pyarrow')
            except Exception:
                # ATTEMPT 2: Rescue with fastparquet if pyarrow fails on 'fixed' timezone
                try:
                    tmp_df = pd.read_parquet(file_path, engine='fastparquet')
                except Exception as e:
                    print(f"❌ Critical Failure: Could not read {file} with any engine. Error: {e}")
                    continue
            
            # If we successfully got a dataframe, normalize it
            if tmp_df is not None:
                for col in tmp_df.columns:
                    if "date" in col or "at" in col:
                        tmp_df[col] = pd.to_datetime(tmp_df[col], utc=True, errors="coerce")
                all_dfs.append(tmp_df)
        
        if not all_dfs:
            df = pd.DataFrame(columns=["repo", "author", "date"])
        else:
            df = pd.concat(all_dfs, ignore_index=True)

        if "date" in df.columns and not df.empty:
            df = df.dropna(subset=["date"]).sort_values("date")
        
        output_path = os.path.join(OUTPUT_DIR, final_filename)
        # We save using the default engine (pyarrow) which is now safe because we normalized everything
        df.to_parquet(output_path, index=False)
        print(f"✅ Saved {final_filename} ({len(df)} rows)")

    for key, filename in data_map.items():
        finalize_parquet(key, filename)

    print(f"\nProcessing complete. Skipped: {skipped}")
    print(len(skipped))
if __name__ == "__main__":
    main()
