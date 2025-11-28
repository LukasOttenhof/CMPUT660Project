# scripts/analysis/plot_utils.py

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")


def monthly_or_quarterly_boxplot(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    date_col: str,
    group_col: str,
    title: str,
    outdir: Path,
    filename: str,
    freq: Literal["M", "Q"] = "M",
    min_date: str | None = "2018-01-01",
):
    """
    Categorical seaborn boxplot with:
    - monthly/quarterly binning
    - linear + log versions
    - X-axis labels thinned to one label per YEAR
    - NO datetime ticks (prevents ConversionError)
    """

    outdir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # INTERNAL: Aggregate helper
    # ----------------------------
    def aggregate(df: pd.DataFrame, label: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["bin", group_col, "count", "period"])
        
        tmp = df.copy()
        tmp["bin"] = tmp[date_col].dt.to_period(freq).dt.to_timestamp()

        if min_date is not None:
            tmp = tmp[tmp["bin"] >= pd.Timestamp(min_date)]

        g = tmp.groupby(["bin", group_col]).size().reset_index(name="count")
        g["period"] = label
        return g

    # Aggregate
    before = aggregate(df_before, "before")
    after = aggregate(df_after, "after")

    data = pd.concat([before, after], ignore_index=True)
    if data.empty:
        return

    # ---------------------------------
    # Build categorical labels for x-axis
    # ---------------------------------
    data = data.sort_values("bin")
    data["bin_label"] = data["bin"].dt.strftime("%Y-%m")

    # Unique ordered bins
    bins = data["bin_label"].unique().tolist()

    # Map bin_label → categorical integer position
    data["bin_pos"] = data["bin_label"].apply(lambda x: bins.index(x))

    # Determine yearly tick positions (categorical index)
    years = sorted(data["bin"].dt.year.unique())

    year_positions = []
    year_labels = []
    for y in years:
        # first bin of that year
        idx = data.index[data["bin"].dt.year == y]
        if len(idx) == 0:
            continue
        first_pos = data.loc[idx[0], "bin_pos"]
        year_positions.append(first_pos)
        year_labels.append(str(y))

    # ---------------------------------------------
    # PLOT: linear & log versions
    # ---------------------------------------------
    for scale in ["linear", "log"]:
        plt.figure(figsize=(20, 5))

        ax = sns.boxplot(
            data=data,
            x="bin_pos",
            y="count",
            hue="period",
            showfliers=False,
        )

        if scale == "log":
            ax.set_yscale("log")

        # Set tick positions (categorical integers)
        ax.set_xticks(year_positions)
        ax.set_xticklabels(year_labels)

        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Activity count" + (" (log scale)" if scale == "log" else ""))
        ax.set_title(f"{title} ({scale} scale)")

        plt.tight_layout()
        out_path = outdir / f"{filename}_{scale}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

def stacked_activity_share_bar(
    activity_counts_before,
    activity_counts_after,
    outdir,
    filename,
    title="Activity Share Before vs After"
):
    """
    Automatically detects activity types from dictionary keys.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / filename

    # Auto-detect activity types
    activity_types = sorted(activity_counts_before.keys())

    # Totals
    total_before = sum(activity_counts_before.values())
    total_after = sum(activity_counts_after.values())

    # Proportions
    props_before = {a: activity_counts_before[a] / total_before for a in activity_types}
    props_after  = {a: activity_counts_after[a] / total_after  for a in activity_types}

    # Global average for ordering
    avg_props = {a: (props_before[a] + props_after[a]) / 2 for a in activity_types}

    # Sort descending so largest is at bottom
    sorted_acts = sorted(avg_props.keys(), key=lambda a: avg_props[a], reverse=True)

    # Colors
    cmap = plt.get_cmap("tab20")
    colors = {a: cmap(i) for i, a in enumerate(sorted_acts)}

    fig, ax = plt.subplots(figsize=(7, 6))

    x = np.array([0, 1])
    labels = ["before", "after"]

    bottom_b = 0.0
    bottom_a = 0.0

    for a in sorted_acts:
        pb = props_before[a]
        pa = props_after[a]

        ax.bar(x[0], pb, bottom=bottom_b, color=colors[a], label=a)
        ax.bar(x[1], pa, bottom=bottom_a, color=colors[a])

        bottom_b += pb
        bottom_a += pa

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Proportion of activity")
    ax.set_title(title)

    ax.legend(title="Activity Type", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
