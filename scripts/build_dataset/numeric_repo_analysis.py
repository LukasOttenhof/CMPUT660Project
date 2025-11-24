#!/usr/bin/env python3
from __future__ import annotations

import git
import os
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path
import subprocess
import json


class DeveloperAnalyzer:
    """
    NEW VERSION (Local Git Extraction)
    ----------------------------------
    Replaces API calls with local repo mining:
    - Extracts commits, LOC changes, files changed directly from git.
    - No rate limits, no backoff, no freezes.
    - Fully compatible with your main pipeline (same .run() signature).
    """

    def __init__(
        self,
        tokens,              # ignored now (kept for compatibility)
        repo_name: str,      # e.g., "owner/repo"
        start_date: datetime,
        end_date: datetime,
        boundary_date: datetime,
        max_workers: int = 1,
        sleep_if_exhausted: int = 60,
    ):
        self.repo_name = repo_name
        self.start_date = start_date
        self.end_date = end_date
        self.boundary_date = boundary_date

        # Path to local clone: repos/owner_repo
        owner, name = repo_name.split("/")
        local_path = f"repos/{owner}_{name}"
        self.repo_path = Path(local_path)

        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"Local repo not found: {self.repo_path}. "
                "Make sure cloning step completed successfully."
            )

        self.repo = git.Repo(self.repo_path)
        self.records = []

    # ------------------------------------------------------------
    # LOCAL COMMIT EXTRACTION
    # ------------------------------------------------------------
    def fetch_commits(self):
        print(f"📥 Scanning commits locally via rev-list for {self.repo_name} ...")

        # Use git CLI instead of GitPython (10–100× faster, never hangs)
        cmd = [
            "git", "-C", str(self.repo_path),
            "log",
            "--since", self.start_date.isoformat(),
            "--until", self.end_date.isoformat(),
            "--pretty=format:%H|%at|%ae|%an",
            "--numstat"
        ]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

        current_commit = None

        for line in proc.stdout:
            line = line.strip()

            # New commit header?
            if "|" in line:
                sha, timestamp, author_email, author_name = line.split("|")
                date = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
                author = author_email or author_name

                phase = "before" if date < self.boundary_date else "after"

                current_commit = {
                    "repo": self.repo_name,
                    "sha": sha,
                    "author": author,
                    "date": date,
                    "phase": phase,
                    "loc_added": 0,
                    "loc_deleted": 0,
                    "files_changed": 0,
                }
                continue

            # numstat line: "{added}\t{deleted}\t{filepath}"
            if current_commit and "\t" in line:
                added, deleted, _ = line.split("\t")
                if added.isdigit(): current_commit["loc_added"] += int(added)
                if deleted.isdigit(): current_commit["loc_deleted"] += int(deleted)
                current_commit["files_changed"] += 1

            # End of commit (empty line)
            if line == "" and current_commit:
                self.records.append(current_commit)
                current_commit = None

    # ------------------------------------------------------------
    # LOCAL PSEUDO-PRs? (Optional)
    # ------------------------------------------------------------
    def fetch_pull_requests(self):
        """
        Not needed: PR numeric metrics stay API-side.
        Kept empty for compatibility with UnifiedDeveloperAnalyzer.
        """
        return

    def fetch_issues(self):
        """Also not needed locally; textual API extractor handles these."""
        return

    # ------------------------------------------------------------
    # MAIN EXECUTION
    # ------------------------------------------------------------
    def run(self):
        print(f"\n🚀 Local numeric analysis for {self.repo_name}")
        print(f"   Range: {self.start_date.date()} → {self.end_date.date()}")
        print(f"   Boundary (agentic PR): {self.boundary_date}")

        self.fetch_commits()

        if not self.records:
            return pd.DataFrame()

        df = pd.DataFrame(self.records)
        df.sort_values("date", inplace=True)
        return df
