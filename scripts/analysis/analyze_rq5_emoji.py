# =============================================================
# rq5 — Emoji Use Analysis by Topic & Before/After
# =============================================================
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_all


# -------------------------------------------------------------
# Output paths
# -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
OUT_TABLES = ROOT / "outputs" / "rq5" / "tables" / "emoji"
OUT_PLOTS = ROOT / "outputs" / "rq5" / "plots" / "emoji"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------
# Emoji detection
# -------------------------------------------------------------
EMOJI_REGEX = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport/map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental
    "\U0001FA70-\U0001FAFF"  # newer emoji
    "]+",
    flags=re.UNICODE
)

def count_emojis(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(EMOJI_REGEX.findall(text))


# -------------------------------------------------------------
# Process each dataset
# -------------------------------------------------------------
def run() -> None:
    data = load_all()

    # must match your existing RQ5 doc-topic outputs
    SOURCE_FILES = {
        "commit_messages": "rq5_doc_topics_commit_messages.csv",
        "pr_bodies": "rq5_doc_topics_pr_bodies.csv",
        "issue_bodies": "rq5_doc_topics_issue_bodies.csv",
        "review_comments": "rq5_doc_topics_review_comments.csv",
    }

    for name, topic_file in SOURCE_FILES.items():
        print(f"[rq5-emoji] Processing emoji use for {name} ...")

        topic_path = ROOT / "outputs" / "rq5" / "tables" / topic_file
        if not topic_path.exists():
            print(f"[rq5-emoji] Missing topic file for {name}: {topic_path}")
            continue

        df = pd.read_csv(topic_path)

        if "text" not in df.columns:
            print(f"[rq5-emoji] file missing 'text' column, skipping.")
            continue

        # ---------------------------------------------------------
        # Compute emoji counts
        # ---------------------------------------------------------
        df["emoji_count"] = df["text"].apply(count_emojis)
        df["emoji_density"] = df.apply(
            lambda row: row["emoji_count"] / max(len(row["text"]), 1),
            axis=1
        )

        # ---------------------------------------------------------
        # Save enriched document table
        # ---------------------------------------------------------
        out_doc_path = OUT_TABLES / f"rq5_emoji_docs_{name}.csv"
        df.to_csv(out_doc_path, index=False)
        print(f"[rq5-emoji] Saved per-doc emoji table → {out_doc_path}")

        # ---------------------------------------------------------
        # Aggregate: emoji usage per topic before/after
        # ---------------------------------------------------------
        summary = (
            df.groupby(["topic_label", "group"])
              .agg(
                  total_emojis=("emoji_count", "sum"),
                  avg_emojis=("emoji_count", "mean"),
                  avg_density=("emoji_density", "mean"),
                  n_docs=("emoji_count", "count"),
              )
              .reset_index()
        )

        # Compute delta columns (after - before)
        delta_rows = []
        for topic in summary["topic_label"].unique():
            try:
                b = summary[(summary["topic_label"] == topic) & (summary["group"] == "before")].iloc[0]
                a = summary[(summary["topic_label"] == topic) & (summary["group"] == "after")].iloc[0]
                delta_rows.append({
                    "topic_label": topic,
                    "delta_avg_emojis": a["avg_emojis"] - b["avg_emojis"],
                    "delta_avg_density": a["avg_density"] - b["avg_density"],
                    "delta_total": a["total_emojis"] - b["total_emojis"]
                })
            except IndexError:
                continue

        delta_df = pd.DataFrame(delta_rows)

        summary_path = OUT_TABLES / f"rq5_emoji_summary_{name}.csv"
        summary.to_csv(summary_path, index=False)

        delta_path = OUT_TABLES / f"rq5_emoji_summary_delta_{name}.csv"
        delta_df.to_csv(delta_path, index=False)

        print(f"[rq5-emoji] Saved summary → {summary_path}")
        print(f"[rq5-emoji] Saved delta summary → {delta_path}")

        # ---------------------------------------------------------
        # Plot: Emoji usage before/after by topic
        # ---------------------------------------------------------
        pivot = summary.pivot(
            index="topic_label",
            columns="group",
            values="avg_emojis"
        ).fillna(0)

        plt.figure(figsize=(12, 6))
        pivot[["before", "after"]].plot(
            kind="bar",
            figsize=(12, 6),
            width=0.8,
            color=["#87a7d6", "#5ca96d"]
        )
        plt.title(f"Emoji Use (Avg per Document) — {name}")
        plt.ylabel("Average emoji count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        out_plot = OUT_PLOTS / f"rq5_emoji_{name}.png"
        plt.savefig(out_plot, dpi=300)
        plt.close()

        print(f"[rq5-emoji] Saved plot → {out_plot}")

    print("[rq5-emoji] Done!")


if __name__ == "__main__":
    run()
