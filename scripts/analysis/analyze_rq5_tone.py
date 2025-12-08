from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def safe_filename(s: str) -> str:
    s = s.replace("/", "_")
    s = s.replace("\\", "_")
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_-]", "", s)
    return s

ROOT = Path(__file__).resolve().parents[2]
TOPIC_TABLES = ROOT / "outputs" / "rq5" / "tables"
TONE_TABLES = ROOT / "outputs" / "rq5" / "tables" / "tone_by_topic"
TONE_PLOTS  = ROOT / "outputs" / "rq5" / "plots"  / "tone_by_topic"
TONE_TABLES.mkdir(parents=True, exist_ok=True)
TONE_PLOTS.mkdir(parents=True, exist_ok=True)

analyzer = SentimentIntensityAnalyzer()


def score_tone(text: str):
    if not isinstance(text, str) or not text.strip():
        return {"neg":0.0, "neu":0.0, "pos":0.0, "compound":0.0}
    return analyzer.polarity_scores(text)


def categorize(c: float) -> str:
    if c >= 0.05: return "positive"
    if c <= -0.05: return "negative"
    return "neutral"


#Old sentiment colour palette
COLORS = {
    "negative": "#cc3f3f",
    "neutral":  "#d8cd79",
    "positive": "#5ca96d",
}

SENT_ORDER = ["negative", "neutral", "positive"]


def plot_topic_stacked(topic: str, df: pd.DataFrame, outpath: Path):
    """
    df has: group, sentiment_cat, share
    """
    pivot = (
        df.pivot(index="group", columns="sentiment_cat", values="share")
        .reindex(["before", "after"])
        .reindex(columns=SENT_ORDER, fill_value=0.0)
    )

    plt.figure(figsize=(6,5))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot.index))

    for s in SENT_ORDER:
        plt.bar(
            x, pivot[s].values,
            bottom=bottom,
            color=COLORS[s],
            label=s.capitalize()
        )
        bottom += pivot[s].values

    plt.xticks(x, pivot.index)
    plt.ylim(0,1)
    plt.ylabel("Proportion of docs")
    plt.title(f"Sentiment by Topic: {topic}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def run():
    TARGETS = [
        "commit_messages",
        "pr_bodies",
        "issue_bodies",
        "review_comments",
    ]

    for name in TARGETS:
        print(f"[rq5-tone] Processing sentiment by topic for {name}...")

        topic_file = TOPIC_TABLES / f"rq5_doc_topics_{name}.csv"
        if not topic_file.exists():
            print(f"[rq5-tone] MISSING topic file for {name}, skipping.")
            continue

        df = pd.read_csv(topic_file)

        #Sentiment computation
        tones = df["text"].astype(str).apply(score_tone)
        df["neg"]      = tones.apply(lambda d: d["neg"])
        df["neu"]      = tones.apply(lambda d: d["neu"])
        df["pos"]      = tones.apply(lambda d: d["pos"])
        df["compound"] = tones.apply(lambda d: d["compound"])
        df["sentiment_cat"] = df["compound"].apply(categorize)

        summary = (
            df.groupby(["topic_label", "group", "sentiment_cat"])
              .size()
              .reset_index(name="count")
        )

        summary["share"] = (
            summary.groupby(["topic_label", "group"])["count"]
                   .transform(lambda x: x / x.sum())
        )

        out_csv = TONE_TABLES / f"rq5_tone_by_topic_{name}.csv"
        summary.to_csv(out_csv, index=False)
        print(f"[rq5-tone] Saved → {out_csv}")

        #Stacked sentiment bars
        for topic in summary["topic_label"].unique():
            sdf = summary[summary["topic_label"] == topic]
            topic_file = safe_filename(topic)
            out_plot = TONE_PLOTS / f"rq5_tone_{name}_{topic_file}.png"
            plot_topic_stacked(topic, sdf, out_plot)

    print("[rq5-tone] Done!")


if __name__ == "__main__":
    run()
