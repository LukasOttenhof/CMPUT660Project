from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_loader import load_all
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT / "outputs" / "pokemon" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "commits",
    "prs_created",
    "prs_merged",
    "reviews_submitted",
    "review_comments",
    "issues_opened",
    "issues_closed",
]

METRIC_MAP = {
    "commits": ("commits", None),
    "prs_created": ("pull_requests", "pr_created"),
    "prs_merged": ("pull_requests", "pr_merged"),
    "reviews_submitted": ("reviews", None),
    "review_comments": ("review_comments", None),
    "issues_opened": ("issues", "issue_opened"),
    "issues_closed": ("issues", "issue_closed"),
}

def compute_monthly_avg_ratios(data, metric_name, scale_before=False):
    """
    Compute per-author-per-month sums scaled by active authors,
    then compute the sum across months and express as a fraction of the total actions.
    """
    key, subtype = METRIC_MAP[metric_name]
    df_before = data[f"{key}_before"].copy()
    df_after = data[f"{key}_after"].copy()

    if subtype:
        df_before = df_before[df_before["activity_type"] == subtype]
        df_after = df_after[df_after["activity_type"] == subtype]

    def per_month_scaled(df, scale=True):
        if df.empty:
            return 0.0
        df = df.copy()
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        grouped = df.groupby(["author", "month"]).size().reset_index(name="count")
        if not scale:
            return grouped["count"].sum()
        # scale by active authors per month
        month_totals = grouped.groupby("month")["count"].sum()
        month_authors = grouped.groupby("month")["author"].nunique()
        scaled_per_month = month_totals / month_authors
        # sum over months with activity
        return scaled_per_month.sum()

    total_before = sum(per_month_scaled(data[f"{k}_before"].copy(), scale=True)
                       for k, _ in METRIC_MAP.values())
    total_after = sum(per_month_scaled(data[f"{k}_after"].copy(), scale=True)
                      for k, _ in METRIC_MAP.values())

    before_val = per_month_scaled(df_before, scale=scale_before)
    after_val = per_month_scaled(df_after, scale=True)

    # Convert to ratio of total actions
    before_ratio = before_val / total_before if total_before > 0 else 0.0
    after_ratio = after_val / total_after if total_after > 0 else 0.0
    return before_ratio, after_ratio


def build_radar_data_ratios(data):
    before = {}
    after = {}

    for metric in METRICS:
        key, subtype = METRIC_MAP[metric]
        df_b = data[f"{key}_before"].copy()
        df_a = data[f"{key}_after"].copy()
        if subtype:
            df_b = df_b[df_b["activity_type"] == subtype]
            df_a = df_a[df_a["activity_type"] == subtype]

        # Sum total actions across all months and authors
        total_b = len(df_b)
        total_a = len(df_a)

        before[metric] = total_b
        after[metric] = total_a

    # Convert to ratios of totals
    sum_b = sum(before.values())
    sum_a = sum(after.values())

    for k in before:
        before[k] /= sum_b if sum_b > 0 else 1
        after[k]  /= sum_a  if sum_a  > 0 else 1

    return before, after


import seaborn as sns

import seaborn as sns

def plot_radar(before, after, title, filename):
    labels = list(before.keys())
    N = len(labels)

    values_before = list(before.values())
    values_after = list(after.values())

    # Close the loop
    values_before += values_before[:1]
    values_after += values_after[:1]
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Apply sqrt scaling for visual balance
    values_before_scaled = [np.sqrt(v) for v in values_before]
    values_after_scaled = [np.sqrt(v) for v in values_after]

    # Clean red and blue
    color_before_line = "#E74C3C" 
    color_before_fill = "#F1948A"  
    color_after_line = "#3498DB"   
    color_after_fill = "#85C1E9"  

    # Plot Before
    ax.plot(angles, values_before_scaled, color=color_before_line, linewidth=2, label="Before Agents")
    ax.fill(angles, values_before_scaled, color=color_before_fill, alpha=0.5)

    # Plot After
    ax.plot(angles, values_after_scaled, color=color_after_line, linewidth=2, label="After Agents")
    ax.fill(angles, values_after_scaled, color=color_after_fill, alpha=0.5)

    # Labels and title
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12, fontweight='bold')
    ax.set_title(title, size=18, pad=20, fontweight='bold')
    ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.1), fontsize=12)

    # Grid and ticks
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_rlabel_position(30)
    ax.tick_params(axis='y', labelsize=10)

    # Save
    outpath = PLOTS_DIR / filename
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[rq2] Saved radar chart -> {outpath}")

def main():
    data = load_all()
    before, after = build_radar_data_ratios(data)

    # Plot radar
    plot_radar(
        before, after,
        "Developer Activity Before vs After Agents (Ratios)",
        "rq2_radar_developer_activity_ratios.png"
    )

    # Print ratios as table
    ratio_df = pd.DataFrame({
        "Metric": list(before.keys()),
        "Before (Ratio)": [f"{v:.3f}" for v in before.values()],
        "After (Ratio)":  [f"{v:.3f}" for v in after.values()]
    })
    print("\nDeveloper Activity Ratios (Before vs After Agents):\n")
    print(ratio_df.to_string(index=False))


   


if __name__ == "__main__":
    main()
