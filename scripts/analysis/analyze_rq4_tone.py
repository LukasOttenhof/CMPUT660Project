# scripts/analysis/analyze_rq4_tone.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_all  # same as in analyze_rq4

# If you don't have it yet: pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
TABLES_DIR = ROOT / "outputs" / "rq4" / "tables"
PLOTS_DIR = ROOT / "outputs" / "rq4" / "plots"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

analyzer = SentimentIntensityAnalyzer()


def score_tone(text: str) -> Dict[str, float]:
    if not isinstance(text, str) or not text.strip():
        return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
    scores = analyzer.polarity_scores(text)
    return scores  # keys: neg, neu, pos, compound


def categorize_compound(c: float) -> str:
    # Standard VADER thresholds
    if c >= 0.05:
        return "positive"
    elif c <= -0.05:
        return "negative"
    else:
        return "neutral"


def run() -> None:
    data = load_all()

    TEXT_TARGETS = {
        "commit_messages": ("commit_messages_before", "commit_messages_after", "text", "date"),
        "pr_bodies": ("pr_bodies_before", "pr_bodies_after", "text", "date"),
        "issue_bodies": ("issue_bodies_before", "issue_bodies_after", "text", "date"),
        "review_comments": ("review_comments_before", "review_comments_after", "text", "date"),
    }

    for name, (before_key, after_key, text_col, time_col) in TEXT_TARGETS.items():
        print(f"[rq4-tone] Sentiment analysis for {name}…")

        before_df = data.get(before_key)
        after_df = data.get(after_key)

        if before_df is None or after_df is None:
            print(f"[rq4-tone] {name}: missing data; skipping.")
            continue

        # Prepare + compute sentiment
        def prep(df: pd.DataFrame, group_label: str) -> pd.DataFrame:
            tmp = df[[text_col, time_col]].copy()
            tmp[text_col] = tmp[text_col].fillna("").astype(str)
            tmp["time"] = pd.to_datetime(tmp[time_col], errors="coerce", utc=True)
            tmp["group"] = group_label

            scores = tmp[text_col].apply(score_tone)
            tmp["neg"] = scores.apply(lambda d: d["neg"])
            tmp["neu"] = scores.apply(lambda d: d["neu"])
            tmp["pos"] = scores.apply(lambda d: d["pos"])
            tmp["compound"] = scores.apply(lambda d: d["compound"])
            tmp["sentiment_cat"] = tmp["compound"].apply(categorize_compound)
            return tmp

        b = prep(before_df, "before")
        a = prep(after_df, "after")

        combined = pd.concat([b, a], ignore_index=True)
        combined_path = TABLES_DIR / f"rq4_tone_{name}.csv"
        combined.to_csv(combined_path, index=False)
        print(f"[rq4-tone] Saved per-doc sentiment → {combined_path}")

        # Simple group-level summary
        summary = (
            combined.groupby(["group", "sentiment_cat"])
            .size()
            .reset_index(name="count")
        )
        total_per_group = summary.groupby("group")["count"].transform("sum")
        summary["share"] = summary["count"] / total_per_group

        summary_path = TABLES_DIR / f"rq4_tone_summary_{name}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[rq4-tone] Saved tone summary → {summary_path}")

        # ---- STACKED BARPLOT WITH FIXED COLORS ----
        pivot = (
            summary.pivot(index="group", columns="sentiment_cat", values="share")
            .reindex(["before", "after"])    # <-- FORCE ORDER HERE
            .fillna(0.0)
        )

        # Ensure consistent ordering
        sent_order = ["negative", "neutral", "positive"]

        # Reindex to enforce ordering
        pivot = pivot.reindex(columns=sent_order)

        # Color mapping
        colors = {
            "negative": "#cc3f3f",  # red
            "neutral":  "#d4c67a",  # blue
            "positive": "#5ca96d",  # green
        }

        plt.figure(figsize=(7, 5))

        bottom = np.zeros(len(pivot))
        x = np.arange(len(pivot.index))

        for sent in sent_order:
            plt.bar(
                x,
                pivot[sent].values,
                bottom=bottom,
                color=colors[sent],
                label=sent.capitalize()
            )
            bottom += pivot[sent].values

        plt.xticks(x, pivot.index)
        plt.ylim(0, 1.0)
        plt.ylabel("Proportion of documents")
        plt.title(f"{name}: sentiment distribution (before vs after)")
        plt.legend(title="Sentiment")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"rq4_tone_{name}.png", dpi=300)
        plt.close()


    print("[rq4-tone] Sentiment analysis complete.")
    

if __name__ == "__main__":
    run()
