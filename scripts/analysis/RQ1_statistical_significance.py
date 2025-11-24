#!/usr/bin/env python3
"""
RQ1_statistical_tests.py

Runs statistical significance tests for RQ1:
- Signed-rank (Wilcoxon)
- Paired t-test
- Cliff's delta

On:
1) Author-level metrics
2) Repo-level metrics
"""

import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon, ttest_rel
import numpy as np

TABLES = Path("outputs/tables")
TABLES.mkdir(parents=True, exist_ok=True)

author = pd.read_csv(TABLES / "rq1_author_metrics.csv")
repo   = pd.read_csv(TABLES / "rq1_repo_metrics.csv")

# ----------------------------------------------------------------------------
# Cliff's delta
# ----------------------------------------------------------------------------
def cliffs_delta(before, after):
    diffs = []
    for b in before:
        for a in after:
            diffs.append(a - b)
    return sum(np.sign(diffs)) / len(diffs)


# ----------------------------------------------------------------------------
# Run tests on a metric
# ----------------------------------------------------------------------------
def run_tests(df, before_col, after_col):
    b = df[before_col]
    a = df[after_col]

    if (b == a).all():
        return {"metric": before_col.replace("_before",""), 
                "wilcoxon_p": "NA", 
                "t_p": "NA", 
                "cliffs_delta": 0}

    w = wilcoxon(b, a).pvalue
    t = ttest_rel(b, a).pvalue
    cd = cliffs_delta(b, a)

    return {
        "metric": before_col.replace("_before",""),
        "wilcoxon_p": w,
        "t_p": t,
        "cliffs_delta": cd
    }


# ----------------------------------------------------------------------------
# AUTHOR-LEVEL RESULTS
# ----------------------------------------------------------------------------
author_results = [
    run_tests(author, "commits_before", "commits_after"),
    run_tests(author, "prs_before", "prs_after"),
    run_tests(author, "reviews_before", "reviews_after")
]

pd.DataFrame(author_results).to_csv(TABLES / "rq1_author_stats.csv", index=False)


# ----------------------------------------------------------------------------
# REPO-LEVEL RESULTS
# ----------------------------------------------------------------------------
repo_results = [
    run_tests(repo, "commits_before", "commits_after"),
    run_tests(repo, "prs_before", "prs_after"),
    run_tests(repo, "reviews_before", "reviews_after")
]

pd.DataFrame(repo_results).to_csv(TABLES / "rq1_repo_stats.csv", index=False)

print("RQ1 statistical tests complete.")
print("Results saved to outputs/tables/")
