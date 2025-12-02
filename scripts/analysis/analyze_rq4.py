# scripts/analysis/analyze_rq3.py

from __future__ import annotations

from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT / "outputs" / "rq3" / "tables"
PLOTS_DIR = ROOT / "outputs" / "rq3" / "plots"

from data_loader import load_all, get_repo_language_mapping
from table_utils import summarize_before_after, save_table
from plot_utils import monthly_or_quarterly_boxplot

import pandas as pd


def add_lang_type(df: pd.DataFrame, lang_map: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.assign(lang_type=pd.Series(dtype=str))
    merged = df.merge(lang_map, on="repo", how="left")
    return merged


def text_length(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.split().str.len()


def main():
    data = load_all()

    issue_bodies_b = data["issue_bodies_before"]
    issue_bodies_a = data["issue_bodies_after"]
    review_comments_b = data["review_comments_before"]
    review_comments_a = data["review_comments_after"]
    pr_bodies_b = data["pr_bodies_before"]
    pr_bodies_a = data["pr_bodies_after"]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect repos
    repos = set()
    for df in (issue_bodies_b, issue_bodies_a,
               review_comments_b, review_comments_a,
               pr_bodies_b, pr_bodies_a):
        if not df.empty and "repo" in df.columns:
            repos.update(df["repo"].dropna().unique().tolist())

    lang_map = get_repo_language_mapping(repos)

    issue_bodies_b = add_lang_type(issue_bodies_b, lang_map)
    issue_bodies_a = add_lang_type(issue_bodies_a, lang_map)
    review_comments_b = add_lang_type(review_comments_b, lang_map)
    review_comments_a = add_lang_type(review_comments_a, lang_map)
    pr_bodies_b = add_lang_type(pr_bodies_b, lang_map)
    pr_bodies_a = add_lang_type(pr_bodies_a, lang_map)

    # ----------------------------------------------------------
    # Review comment text length before/after
    # ----------------------------------------------------------
    rc_text_len_b = text_length(review_comments_b["text"]) if not review_comments_b.empty else pd.Series(dtype=float)
    rc_text_len_a = text_length(review_comments_a["text"]) if not review_comments_a.empty else pd.Series(dtype=float)
    rc_text_table = summarize_before_after(rc_text_len_b, rc_text_len_a)
    save_table(rc_text_table, "rq3_review_comment_text_length", TABLES_DIR)

    # ----------------------------------------------------------
    # Combined communication volume per repo
    # ----------------------------------------------------------
    def comm_volume(df_issue, df_review, df_pr, period: str) -> pd.DataFrame:
        repos_local = set()
        for df in (df_issue, df_review, df_pr):
            if not df.empty:
                repos_local.update(df["repo"].dropna().unique().tolist())

        out = pd.DataFrame({"repo": sorted(repos_local)})
        if not df_issue.empty:
            out["issue_bodies"] = out["repo"].map(
                df_issue.groupby("repo").size()
            ).fillna(0).astype(int)
        else:
            out["issue_bodies"] = 0

        if not df_review.empty:
            out["review_comments"] = out["repo"].map(
                df_review.groupby("repo").size()
            ).fillna(0).astype(int)
        else:
            out["review_comments"] = 0

        if not df_pr.empty:
            out["pr_bodies"] = out["repo"].map(
                df_pr.groupby("repo").size()
            ).fillna(0).astype(int)
        else:
            out["pr_bodies"] = 0

        out["total_comm"] = out["issue_bodies"] + out["review_comments"] + out["pr_bodies"]
        out["period"] = period
        return out

    comm_b = comm_volume(issue_bodies_b, review_comments_b, pr_bodies_b, "before")
    comm_a = comm_volume(issue_bodies_a, review_comments_a, pr_bodies_a, "after")
    comm_all = pd.concat([comm_b, comm_a], ignore_index=True)
    comm_all.to_csv(TABLES_DIR / "rq3_comm_volume_per_repo.csv", index=False)

    save_table(
        summarize_before_after(comm_b["total_comm"], comm_a["total_comm"]),
        "rq3_total_comm_per_repo",
        TABLES_DIR,
    )

    # ----------------------------------------------------------
    # Static vs dynamic language: communication text length
    # ----------------------------------------------------------
    def lang_group_text_lengths(df: pd.DataFrame, label: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["lang_type", "period", "mean_len", "median_len", "count"])
        tmp = df.copy()
        tmp["len"] = text_length(tmp["text"])
        grouped = (
            tmp.groupby("lang_type")["len"]
            .agg(["mean", "median", "count"])
            .reset_index()
            .rename(columns={"mean": "mean_len", "median": "median_len"})
        )
        grouped["period"] = label
        return grouped

    issue_len_b = lang_group_text_lengths(issue_bodies_b, "before")
    issue_len_a = lang_group_text_lengths(issue_bodies_a, "after")
    issue_len_all = pd.concat([issue_len_b, issue_len_a], ignore_index=True)
    save_table(issue_len_all, "rq3_issue_text_langtype", TABLES_DIR)

    rc_len_b = lang_group_text_lengths(review_comments_b, "before")
    rc_len_a = lang_group_text_lengths(review_comments_a, "after")
    rc_len_all = pd.concat([rc_len_b, rc_len_a], ignore_index=True)
    save_table(rc_len_all, "rq3_review_text_langtype", TABLES_DIR)

    # Volume by language type
    def lang_group_volume(df: pd.DataFrame, label: str, source_name: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["lang_type", "period", "source", "count"])
        grouped = (
            df.groupby("lang_type")
            .size()
            .reset_index(name="count")
        )
        grouped["period"] = label
        grouped["source"] = source_name
        return grouped

    issue_vol_b = lang_group_volume(issue_bodies_b, "before", "issue_bodies")
    issue_vol_a = lang_group_volume(issue_bodies_a, "after", "issue_bodies")
    rc_vol_b = lang_group_volume(review_comments_b, "before", "review_comments")
    rc_vol_a = lang_group_volume(review_comments_a, "after", "review_comments")
    pr_vol_b = lang_group_volume(pr_bodies_b, "before", "pr_bodies")
    pr_vol_a = lang_group_volume(pr_bodies_a, "after", "pr_bodies")

    lang_vol_all = pd.concat(
        [issue_vol_b, issue_vol_a, rc_vol_b, rc_vol_a, pr_vol_b, pr_vol_a],
        ignore_index=True,
    )
    lang_vol_all.to_csv(TABLES_DIR / "rq3_comm_volume_by_langtype.csv", index=False)
    print("[rq3] Saved communication volume by language type.")

    # ----------------------------------------------------------
    # Time-series boxplots for communication events per repo
    # ----------------------------------------------------------
    for freq, tag in [("M", "monthly"), ("Q", "quarterly")]:
        monthly_or_quarterly_boxplot(
            issue_bodies_b,
            issue_bodies_a,
            date_col="date",
            group_col="repo",
            title=f"Issue communication per repo per {tag} (before vs. after agents)",
            outdir=PLOTS_DIR,
            filename=f"rq3_issues_comm_{tag}_boxplot.png",
            freq=freq,
        )

        monthly_or_quarterly_boxplot(
            review_comments_b,
            review_comments_a,
            date_col="date",
            group_col="repo",
            title=f"Review comments per repo per {tag} (before vs. after agents)",
            outdir=PLOTS_DIR,
            filename=f"rq3_review_comments_{tag}_boxplot.png",
            freq=freq,
        )

    print("[rq3] Analysis complete.")


if __name__ == "__main__":
    main()
