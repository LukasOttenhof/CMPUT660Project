from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
INPUTS_A = ROOT / "inputs" / "50prs"
LANG_CACHE = INPUTS_A / "repo_languages.csv"
INPUTS_H = ROOT / "inputs" / "human_2"
LANG_CACHE_H = INPUTS_H / "repo_languages.csv"

def load_parquet(base: Path, name: str) -> pd.DataFrame:
    path = base / name
    if not path.exists():
        print(f"[data_loader] Missing parquet: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)

def load_all() -> Dict[str, pd.DataFrame]:
    files = {
        "commit_messages": ("commit_messages_before.parquet", "commit_messages_after.parquet"),
        "commits": ("commits_before.parquet", "commits_after.parquet"),
        "discussion_comments": ("discussion_comments_before.parquet", "discussion_comments_after.parquet"),
        "discussion_topics": ("discussion_topics_before.parquet", "discussion_topics_after.parquet"),
        "issue_bodies": ("issue_bodies_before.parquet", "issue_bodies_after.parquet"),
        "issues": ("issues_before.parquet", "issues_after.parquet"),
        "pr_bodies": ("pr_bodies_before.parquet", "pr_bodies_after.parquet"),
        "pull_requests": ("pull_requests_before.parquet", "pull_requests_after.parquet"),
        "review_comments": ("review_comments_before.parquet", "review_comments_after.parquet"),
        "reviews": ("reviews_before.parquet", "reviews_after.parquet"),
    }

    data: Dict[str, pd.DataFrame] = {}
    repos = set()

    for key, (before_file, after_file) in files.items():

        # ----- LOAD BEFORE FROM BOTH -----
        before_h = load_parquet(INPUTS_H, before_file)
        before_a = load_parquet(INPUTS_A, before_file)

        before = pd.concat([before_h, before_a], ignore_index=True)

        if "date" in before.columns:
            before["date"] = pd.to_datetime(before["date"], utc=True)

        if "repo" in before.columns:
            repos.update(before["repo"].dropna().unique())

        data[f"{key}_before"] = before

        # ----- LOAD AFTER SEPARATELY -----
        after_h = load_parquet(INPUTS_H, after_file)
        after_a = load_parquet(INPUTS_A, after_file)

        if "date" in after_h.columns:
            after_h["date"] = pd.to_datetime(after_h["date"], utc=True)

        if "date" in after_a.columns:
            after_a["date"] = pd.to_datetime(after_a["date"], utc=True)

        if "repo" in after_h.columns:
            repos.update(after_h["repo"].dropna().unique())

        if "repo" in after_a.columns:
            repos.update(after_a["repo"].dropna().unique())

        data[f"{key}_after_human"] = after_h
        data[f"{key}_after_agent"] = after_a

    # Language mapping
    data["repo_languages"] = get_repo_language_mapping(repos)

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
