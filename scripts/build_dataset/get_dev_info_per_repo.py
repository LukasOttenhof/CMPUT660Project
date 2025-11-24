#!/usr/bin/env python3
from datetime import datetime, timezone
import pandas as pd
import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from scripts.build_dataset.numeric_repo_analysis import DeveloperAnalyzer     # now local git version
from scripts.build_dataset.text_repo_analysis import DeveloperTextCollector

AGENT_LABELS = {"Claude_Code", "Copilot", "Cursor", "Devin", "OpenAI_Codex"}

def load_boundary_dates(aidev_pr_path):
    print("\n=== Loading AIDEV agentic PRs ===")

    pr = pd.read_parquet(aidev_pr_path)
    pr["repo"] = pr["repo_url"].str.replace("https://api.github.com/repos/", "", regex=False)
    agentic_only = pr[pr["agent"].isin(AGENT_LABELS)].copy()

    boundary = (
        agentic_only.groupby("repo")["created_at"]
        .min()
        .reset_index()
        .rename(columns={"created_at": "boundary_date"})
    )

    boundary["boundary_date"] = pd.to_datetime(boundary["boundary_date"], utc=True)
    return dict(zip(boundary["repo"], boundary["boundary_date"]))

f
class UnifiedDeveloperAnalyzer:
    """Uses local numeric extractor + GitHub API text extractor."""

    def __init__(self, tokens, repo_name, start_date, end_date, boundary_date):
        self.numeric_analyzer = DeveloperAnalyzer(
            tokens=tokens,
            repo_name=repo_name,
            start_date=start_date,
            end_date=end_date,
            boundary_date=boundary_date,
        )

        self.text_analyzer = DeveloperTextCollector(
            tokens=tokens,
            repo_name=repo_name,
            start_date=start_date,
            end_date=end_date,
            boundary_date=boundary_date,
        )

    def run(self):
        numeric_df = self.numeric_analyzer.run()   # local git
        text_df = self.text_analyzer.run()         # GitHub API
        return numeric_df, text_df


if __name__ == "__main__":

    GITHUB_TOKENS = [
        "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",
        "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",
        "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"
    ]

    START_DATE = datetime(1970, 1, 1, tzinfo=timezone.utc)
    END_DATE   = datetime(2025, 12, 31, tzinfo=timezone.utc)

    repos = pd.read_csv("inputs/processed/final_repo_list.csv")["repo"].dropna().unique().tolist()

    BOUNDARIES = load_boundary_dates("inputs/raw/pull_request.parquet")

    os.makedirs("outputs", exist_ok=True)

    numeric_before_all = []
    numeric_after_all = []
    text_before_all = []
    text_after_all = []

    for repo_name in repos:
        print(f"\n=== Processing repo: {repo_name} ===")

        boundary_date = BOUNDARIES.get(repo_name)
        if boundary_date is None:
            print(f"⚠️ Skipping {repo_name} (no boundary)")
            continue

        analyzer = UnifiedDeveloperAnalyzer(
            tokens=GITHUB_TOKENS,
            repo_name=repo_name,
            start_date=START_DATE,
            end_date=END_DATE,
            boundary_date=boundary_date,
        )

        try:
            numeric_df, text_df = analyzer.run()

            nb = numeric_df[numeric_df["date"] < boundary_date].copy()
            na = numeric_df[numeric_df["date"] >= boundary_date].copy()
            tb = text_df[text_df["date"] < boundary_date].copy()
            ta = text_df[text_df["date"] >= boundary_date].copy()

            for df in (nb, na, tb, ta):
                df["repo"] = repo_name

            numeric_before_all.append(nb)
            numeric_after_all.append(na)
            text_before_all.append(tb)
            text_after_all.append(ta)

        except Exception as e:
            print(f"❌ Error processing {repo_name}: {e}")
            continue

    # Save output
    pd.concat(numeric_before_all).to_parquet("outputs/numeric_before.parquet", index=False)
    pd.concat(numeric_after_all).to_parquet("outputs/numeric_after.parquet", index=False)
    pd.concat(text_before_all).to_parquet("outputs/text_before.parquet", index=False)
    pd.concat(text_after_all).to_parquet("outputs/text_after.parquet", index=False)

    print("\n🎉 Done!")
