from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "inputs" / "processed"
LANG_CACHE = INPUTS / "repo_languages.csv"


def load_parquet(name: str) -> pd.DataFrame:
    path = INPUTS / name
    if not path.exists():
        print(f"[data_loader] Missing parquet: {name}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_all() -> Dict[str, pd.DataFrame]:
    """Load all parquet files we care about into a dict."""
    files = {
        "commit_messages_before": "commit_messages_before.parquet",
        "commit_messages_after": "commit_messages_after.parquet",
        "commits_before": "commits_before.parquet",
        "commits_after": "commits_after.parquet",
        "discussion_comments_before": "discussion_comments_before.parquet",
        "discussion_comments_after": "discussion_comments_after.parquet",
        "discussion_topics_before": "discussion_topics_before.parquet",
        "discussion_topics_after": "discussion_topics_after.parquet",
        "issue_bodies_before": "issue_bodies_before.parquet",
        "issue_bodies_after": "issue_bodies_after.parquet",
        "issues_before": "issues_before.parquet",
        "issues_after": "issues_after.parquet",
        "pr_bodies_before": "pr_bodies_before.parquet",
        "pr_bodies_after": "pr_bodies_after.parquet",
        "pull_requests_before": "pull_requests_before.parquet",
        "pull_requests_after": "pull_requests_after.parquet",
        "review_comments_before": "review_comments_before.parquet",
        "review_comments_after": "review_comments_after.parquet",
        "reviews_before": "reviews_before.parquet",
        "reviews_after": "reviews_after.parquet",
    }

    data = {}
    for key, fname in files.items():
        df = load_parquet(fname)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        data[key] = df
    return data


#Language classification///DEPRECATED (unused in paper)

STATIC_LANGS = {
    "C", "C++", "C#", "Java", "Go", "Rust", "Scala", "Kotlin",
    "TypeScript", "Haskell", "OCaml", "Swift"
}
DYNAMIC_LANGS = {
    "Python", "JavaScript", "Ruby", "PHP", "Perl", "Lua", "R"
}


def _github_get_repo_language(repo_full_name: str) -> str | None:
    url = f"https://api.github.com/repos/{repo_full_name}"
    token = os.getenv("GITHUB_TOKEN_1") or os.getenv("GH_TOKEN")

    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.get(url, headers=headers, timeout=10)
    if resp.status_code != 200:
        print(f"[data_loader] GitHub API error for {repo_full_name}: {resp.status_code}")
        return None

    data = resp.json()
    return data.get("language")


def get_repo_language_mapping(repos: Iterable[str]) -> pd.DataFrame:
    repos = sorted(set(r for r in repos if isinstance(r, str) and "/" in r))

    if LANG_CACHE.exists():
        cache = pd.read_csv(LANG_CACHE)
    else:
        cache = pd.DataFrame(columns=["repo", "language", "lang_type"])

    known = set(cache["repo"].unique())
    missing = [r for r in repos if r not in known]

    new_rows = []
    for r in missing:
        lang = _github_get_repo_language(r)
        if lang is None:
            lang_type = "unknown"
        elif lang in STATIC_LANGS:
            lang_type = "static"
        elif lang in DYNAMIC_LANGS:
            lang_type = "dynamic"
        else:
            lang_type = "other"

        print(f"[data_loader] {r} → language={lang}, type={lang_type}")
        new_rows.append({"repo": r, "language": lang, "lang_type": lang_type})

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache.drop_duplicates(subset=["repo"], keep="last", inplace=True)
        LANG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(LANG_CACHE, index=False)

    return cache[["repo", "language", "lang_type"]]
