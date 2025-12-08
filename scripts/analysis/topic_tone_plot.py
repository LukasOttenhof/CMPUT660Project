from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.colors as mc
import colorsys


plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

ROOT = Path(__file__).resolve().parents[2]

SUMMARY_DIR = ROOT / "outputs" / "rq5" / "tables"
TONE_TABLES = ROOT / "outputs" / "rq5" / "tables"
TONE_BY_TOPIC_DIR = ROOT / "outputs" / "rq5" / "tables" / "tone_by_topic"

PLOT_DIR = ROOT / "outputs" / "rq5" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SENTIMENTS = ["negative", "neutral", "positive"]

BASE_BEFORE = "#E74C3C"
BASE_AFTER  = "#3498DB"

def adjust_lightness(color, amount: float):
    """amount > 1 = lighter, amount < 1 = darker"""
    try:
        c = mc.cnames[color]
    except:
        c = color
    h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(c))
    l = max(0, min(1, l * amount))
    return colorsys.hls_to_rgb(h, l, s)

def build_palette(base):
    return {
        "positive": adjust_lightness(base, 1.35),
        "neutral":  base,
        "negative": adjust_lightness(base, 0.65),
    }

PALETTE_BEFORE = build_palette(BASE_BEFORE)
PALETTE_AFTER  = build_palette(BASE_AFTER)

DATASETS = {
    "Commit Messages": "rq5_topic_summary_commit_messages.csv",
    "Issues": "rq5_topic_summary_issue_bodies.csv",
    "Pull Request Bodies": "rq5_topic_summary_pr_bodies.csv",
    "Review Comments": "rq5_topic_summary_review_comments.csv",
}

TONE_FILES = {
    "Commit Messages": "rq5_tone_commit_messages.csv",
    "Issues": "rq5_tone_issue_bodies.csv",
    "Pull Request Bodies": "rq5_tone_pr_bodies.csv",
    "Review Comments": "rq5_tone_review_comments.csv",
}

#Stacked bars
def plot_stacked(df, xcol, title, path, before_palette, after_palette):
    """
    df must contain: xcol, group ("before"/"after"), sentiment_cat, share
    Stacks in NEG→NEU→POS order (negative bottom, positive top).
    """

    labels = sorted(df[xcol].unique())
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(14, 7))

    legend_handles = []
    legend_labels = []

    for i, group in enumerate(["before", "after"]):
        palette = before_palette if group == "before" else after_palette
        bottoms = np.zeros(len(labels))
        gx = x + (i - 0.5) * width  #specify after on the right

        for s in SENTIMENTS:
            subset = (
                df[(df.group == group) & (df.sentiment_cat == s)]
                .set_index(xcol)
                .reindex(labels)
                .fillna(0)
            )

            vals = subset["share"].values

            bar = plt.bar(
                gx,
                vals,
                bottom=bottoms,
                width=width,
                color=palette[s],
            )

            label = f"{group.title()} – {s.title()}"
            if label not in legend_labels:
                legend_handles.append(bar)
                legend_labels.append(label)

            bottoms += vals

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Share")
    plt.title(title)
    #Legend outside axis
    plt.legend(
        legend_handles,
        legend_labels,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 0.80, 1])  # Make room for legend on right
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()



#OLD TOPIC DISTRIBUTION BEFORE BERTOPIC IGNORE 
def plot_topic_distribution():
    dfs = [pd.read_csv(SUMMARY_DIR / f) for f in DATASETS.values()]
    full = pd.concat(dfs, ignore_index=True)

    combined = (
        full.groupby("topic_label")
        .agg(count_before=("count_before", "sum"),
             count_after=("count_after", "sum"))
        .reset_index()
    )

    total_b = combined["count_before"].sum()
    total_a = combined["count_after"].sum()
    combined["share_before"] = combined["count_before"] / total_b
    combined["share_after"]  = combined["count_after"]  / total_a

    topics = combined["topic_label"]
    x = np.arange(len(topics))
    width = 0.35

    #Counts
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, combined["count_before"], width, label="Before", color=BASE_BEFORE)
    plt.bar(x + width/2, combined["count_after"],  width, label="After",  color=BASE_AFTER)
    plt.xticks(x, topics, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Topic Distribution (Combined Dataset)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "topic_distribution_combined_counts.png", dpi=300)
    plt.close()

    #Shares
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, combined["share_before"], width, label="Before", color=BASE_BEFORE)
    plt.bar(x + width/2, combined["share_after"],  width, label="After",  color=BASE_AFTER)
    plt.xticks(x, topics, rotation=45, ha="right")
    plt.ylabel("Share")
    plt.title("Topic Distribution (Combined Dataset, Share)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "topic_distribution_combined_shares.png", dpi=300)
    plt.close()


#Sentiment per dataset
def plot_sentiment_per_dataset():
    dfs = []

    for ds_name, file in TONE_FILES.items():
        fp = TONE_TABLES / file
        if not fp.exists():
            print("[WARN] Missing:", fp)
            continue
        df = pd.read_csv(fp)
        df["dataset"] = ds_name
        if "count" not in df.columns:
            df["count"] = 1
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)

    full["share"] = full.groupby(["dataset", "group"])["count"].transform(lambda x: x / x.sum())

    tidy = (
        full.groupby(["dataset", "group", "sentiment_cat"])["share"]
        .sum()
        .reset_index()
    )


    plot_stacked(
        tidy,
        "dataset",
        "Sentiment Distribution Per Dataset (Before vs After)",
        PLOT_DIR / "sentiment_by_dataset.png",
        PALETTE_BEFORE,
        PALETTE_AFTER
    )



#OLD TOPIC DISTRIBUTION BEFORE BERTOPIC IGNORE 
def plot_sentiment_per_topic():
    csvs = list(TONE_BY_TOPIC_DIR.glob("rq5_tone_by_topic_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)

    agg = (
        df.groupby(["topic_label", "group", "sentiment_cat"])["count"]
        .sum()
        .reset_index()
    )

    agg["share"] = agg.groupby(["topic_label", "group"])["count"].transform(lambda x: x / x.sum())

    plot_stacked(
        agg,
        "topic_label",
        "Sentiment Distribution Per Topic (Before vs After)",
        PLOT_DIR / "sentiment_by_topic_combined.png",
        PALETTE_BEFORE,
        PALETTE_AFTER
    )


def run():
    print("[PLOTS] Topic distribution…")
    plot_topic_distribution()

    print("[PLOTS] Sentiment per dataset…")
    plot_sentiment_per_dataset()

    print("[PLOTS] Sentiment per topic…")
    plot_sentiment_per_topic()

    print("[OK] All plots written to:", PLOT_DIR)


if __name__ == "__main__":
    run()
