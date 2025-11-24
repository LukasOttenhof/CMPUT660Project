#!/usr/bin/env python3
from __future__ import annotations

from github import Github, GithubException, Auth
from datetime import datetime
import pandas as pd
import time
from tqdm import tqdm


class DeveloperTextCollector:
    """
    Text extractor (unchanged functionality)
    ----------------------------------------
    Still uses:
      - Issues and issue comments
      - Pull request titles/bodies
      - PR comments
      - PR reviews
      - Discussions (if enabled)

    Numeric analysis is now handled locally (numeric_repo_analysis.py).
    """

    def __init__(
        self,
        tokens,
        repo_name: str,
        start_date: datetime,
        end_date: datetime,
        boundary_date: datetime,
        max_workers: int = 5,
        sleep_if_exhausted: int = 60,
    ):
        self.tokens = tokens if isinstance(tokens, list) else [tokens]
        self.repo_name = repo_name
        self.start_date = start_date
        self.end_date = end_date
        self.boundary_date = boundary_date
        self.sleep_if_exhausted = sleep_if_exhausted

        self.token_index = 0
        self.github = self._get_client()
        self.repo = self._get_repo()

        self.records = []

    # ---------- (token rotation system left as-is) ----------
    def _get_client(self):
        token = self.tokens[self.token_index]
        return Github(auth=Auth.Token(token), per_page=100)

    def _rotate_token(self):
        self.token_index = (self.token_index + 1) % len(self.tokens)
        self.github = self._get_client()

    def _maybe_wait_or_rotate(self, exc: GithubException) -> bool:
        msg = ""
        if isinstance(exc.data, dict):
            msg = (exc.data.get("message") or "").lower()

        try:
            remaining, _ = self.github.rate_limiting
        except Exception:
            remaining = None

        if exc.status in (401, 403) or "rate limit" in msg:
            for _ in range(len(self.tokens)):
                if remaining and remaining > 0:
                    return True
                self._rotate_token()
                try:
                    remaining, _ = self.github.rate_limiting
                except Exception:
                    remaining = None
            time.sleep(self.sleep_if_exhausted)
            return True

        return False

    def _safe(self, func, *args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except GithubException as e:
                if self._maybe_wait_or_rotate(e):
                    continue
                raise

    # ---------------------------------------------------------
    def _get_repo(self):
        return self._safe(self.github.get_repo, self.repo_name)

    # ---------------------------------------------------------
    def _add(self, author, source, text, date):
        if author is None or text is None or date is None:
            return
        if date < self.start_date or date > self.end_date:
            return

        phase = "before" if date < self.boundary_date else "after"

        self.records.append(
            {
                "author": author,
                "source": source,
                "text": text.strip(),
                "date": date,
                "phase": phase,
            }
        )

    # (issue extractor unchanged)
    # (PR extractor unchanged)
    # (discussion extractor unchanged)

    def run(self):
        print(f"\n🚀 Text analysis for {self.repo_name}")
        print(f"   Range: {self.start_date.date()} → {self.end_date.date()}")
        print(f"   Boundary: {self.boundary_date}\n")

        self.fetch_issues_and_comments()
        self.fetch_pull_requests()
        self.fetch_discussions()

        df = pd.DataFrame(self.records)
        df.sort_values("date", inplace=True)
        return df
