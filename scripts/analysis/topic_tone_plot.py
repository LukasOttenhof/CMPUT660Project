from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.colors as mc
import colorsys

# Updated global styling for much larger text
plt.rcParams.update({
    "font.size": 20,              # Increased base size
    "axes.titlesize": 26,         # Large title
    "axes.labelsize": 22,         # Large X/Y labels
    "xtick.labelsize": 20,        # Large tick labels
    "ytick.labelsize": 20,
    "legend.fontsize": 18,        # Large legend text
    "figure.titlesize": 28
})

ROOT = Path(__file__).resolve().parents[2]
TONE_TABLES = ROOT / "outputs" / "rq5" / "tables"
PLOT_DIR = ROOT / "outputs" / "rq5" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SENTIMENTS = ["negative", "neutral", "positive"]
GROUPS = ["before", "after_human", "after_agent"]

# Base Colors
BASE_BEFORE = "#FFDE21"  # Gold
BASE_HUMAN  = "#3498DB"  # Blue
BASE_AGENT  = "#2ECC71"  # Green

def adjust_lightness(color, amount: float):
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

PALETTES = {
    "before": build_palette(BASE_BEFORE),
    "after_human": build_palette(BASE_HUMAN),
    "after_agent": build_palette(BASE_AGENT)
}

TONE_FILES = {
    "Commit Messages": "rq5_tone_commit_messages.csv",
    "Issues": "rq5_tone_issue_bodies.csv",
    "Pull Request Bodies": "rq5_tone_pr_bodies.csv",
    "Review Comments": "rq5_tone_review_comments.csv",
}

def plot_stacked_3way(df, xcol, title, path):
    """
    Plots three stacked bars per X-axis category: Before, After Human, After Agent.
    """
    labels = sorted(df[xcol].unique())
    x = np.arange(len(labels))
    width = 0.25 

    fig, ax = plt.subplots(figsize=(18, 14)) # Larger canvas

    legend_handles = []
    legend_labels = []

    for i, group in enumerate(GROUPS):
        palette = PALETTES[group]
        bottoms = np.zeros(len(labels))
        gx = x + (i - 1) * width 

        for s in SENTIMENTS:
            subset = (
                df[(df.group == group) & (df.sentiment_cat == s)]
                .set_index(xcol)
                .reindex(labels)
                .fillna(0)
            )

            vals = subset["share"].values
            bar = ax.bar(
                gx, vals, bottom=bottoms, width=width,
                color=palette[s], edgecolor='white', linewidth=0.8
            )

            # Clean label for legend
            display_group = group.replace('before', 'Before Agents').replace('after_human', 'After (Human)').replace('after_agent', 'After (Agent)')
            label = f"{display_group} – {s.title()}"
            
            if label not in legend_labels:
                legend_handles.append(bar)
                legend_labels.append(label)

            bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=22) # Slanted for readability
    ax.set_ylabel("Share of Documents", labelpad=20)
    ax.set_title(title, pad=40, fontweight='bold')
    
    # Legend - larger font and better placement
    ax.legend(
        legend_handles, legend_labels,
        bbox_to_anchor=(0.5, -0.22), # Moved further down to avoid overlap
        loc="upper center",
        ncol=3, 
        frameon=True,
        fontsize=18,
        title="Group & Sentiment",
        title_fontsize=20
    )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_sentiment_per_dataset():
    dfs = []
    for ds_name, file in TONE_FILES.items():
        fp = TONE_TABLES / file
        if not fp.exists():
            continue
        
        df = pd.read_csv(fp)
        if df.empty: continue
            
        counts = df.groupby(['group', 'sentiment_cat']).size().reset_index(name='count')
        totals = df.groupby('group').size().reset_index(name='total')
        
        merged = counts.merge(totals, on='group')
        merged['share'] = merged['count'] / merged['total']
        merged['dataset'] = ds_name
        dfs.append(merged)

    if not dfs:
        print("[ERR] No sentiment data found to plot.")
        return

    full_tidy = pd.concat(dfs, ignore_index=True)

    plot_stacked_3way(
        full_tidy,
        "dataset",
        "Sentiment Distribution by Period and Contributor Type",
        PLOT_DIR / "sentiment_by_dataset_3way.png"
    )

def run():
    print("[PLOTS] Generating high-res 3-way sentiment comparison...")
    plot_sentiment_per_dataset()
    print(f"[OK] Plots saved to: {PLOT_DIR}")

if __name__ == "__main__":
    run()