# scripts/analysis/plot_utils.py

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import colorsys
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

sns.set(style="whitegrid")

# ----------------------------------------------------------------------
# Lightness-only hue-preserving shade generator
# ----------------------------------------------------------------------
def _generate_lightness_shades(base_hex: str, n: int):
    """
    Generate n shades of the SAME hue:
        darkest (bottom) → lightest (top)
    Only modifies HLS lightness; hue & saturation preserved.
    """
    r, g, b = mcolors.to_rgb(base_hex)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Good default lightness range: dark 0.35 → light 0.80
    lightness_values = np.linspace(0.35, 0.80, n)

    shades = []
    for L in lightness_values:
        r2, g2, b2 = colorsys.hls_to_rgb(h, L, s)
        shades.append(mcolors.to_hex((r2, g2, b2)))

    return shades


# ----------------------------------------------------------------------
# BOXPLOTS (RQ1): now using fixed red/blue palette
# ----------------------------------------------------------------------
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
    - custom before/after palette
    """

    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Internal aggregation
    # -------------------------
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

    before = aggregate(df_before, "before")
    after = aggregate(df_after, "after")

    data = pd.concat([before, after], ignore_index=True)
    if data.empty:
        return

    # Build categorical axis
    data = data.sort_values("bin")
    data["bin_label"] = data["bin"].dt.strftime("%Y-%m")
    bins = data["bin_label"].unique().tolist()
    data["bin_pos"] = data["bin_label"].apply(lambda x: bins.index(x))

    years = sorted(data["bin"].dt.year.unique())
    year_positions = []
    year_labels = []
    for y in years:
        idx = data.index[data["bin"].dt.year == y]
        if len(idx) > 0:
            first_pos = data.loc[idx[0], "bin_pos"]
            year_positions.append(first_pos)
            year_labels.append(str(y))

    # Palette
    PALETTE = {
        "before": "#E74C3C",  # red
        "after":  "#3498DB",  # blue
    }

    # ---------------------------------------
    # Plot (linear + log)
    # ---------------------------------------
    for scale in ["linear", "log"]:
        plt.figure(figsize=(20, 5))

        ax = sns.boxplot(
            data=data,
            x="bin_pos",
            y="count",
            hue="period",
            showfliers=False,
            palette=PALETTE,
        )

        if scale == "log":
            ax.set_yscale("log")

        ax.set_xticks(year_positions)
        ax.set_xticklabels(year_labels)

        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Activity count" + (" (log scale)" if scale == "log" else ""))
        ax.set_title(f"{title} ({scale} scale)")

        plt.tight_layout()
        out_path = outdir / f"{filename}_{scale}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()


# ----------------------------------------------------------------------
# STACKED BARPLOT (RQ1): Before = red gradient, After = blue gradient
# ----------------------------------------------------------------------
def stacked_activity_share_bar(
    activity_counts_before,
    activity_counts_after,
    outdir,
    filename,
    title="Activity Share Before vs After"
):
    """
    Stacked barplot:
    - before uses red gradients (dark → light)
    - after uses blue gradients (dark → light)
    - legend shows BOTH colours per activity type
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / filename

    # Detect activities
    activity_types = sorted(activity_counts_before.keys())

    # Totals
    total_before = sum(activity_counts_before.values())
    total_after = sum(activity_counts_after.values())

    # Proportions
    props_before = {a: activity_counts_before[a] / total_before for a in activity_types}
    props_after  = {a: activity_counts_after[a] / total_after  for a in activity_types}

    # Sort by global average contribution
    avg_props = {a: (props_before[a] + props_after[a]) / 2 for a in activity_types}
    sorted_acts = sorted(avg_props.keys(), key=lambda a: avg_props[a], reverse=True)

    # Generate gradients
    BASE_RED  = "#E74C3C"
    BASE_BLUE = "#3498DB"

    before_shades = _generate_lightness_shades(BASE_RED, len(sorted_acts))
    after_shades  = _generate_lightness_shades(BASE_BLUE, len(sorted_acts))

    before_colors = dict(zip(sorted_acts, before_shades))
    after_colors  = dict(zip(sorted_acts, after_shades))

    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.array([0, 1])
    labels = ["before", "after"]

    bottom_b = 0.0
    bottom_a = 0.0

    for act in sorted_acts:
        pb = props_before[act]
        pa = props_after[act]

        ax.bar(x[0], pb, bottom=bottom_b, color=before_colors[act])
        ax.bar(x[1], pa, bottom=bottom_a, color=after_colors[act])

        bottom_b += pb
        bottom_a += pa

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Proportion of activity")
    ax.set_title(title)

    # --------------------------
    # LEGEND: both colours shown
    # --------------------------
    legend_handles = []
    for act in sorted_acts:
        legend_handles.append(Patch(color=before_colors[act], label=f"{act} (before)"))
        legend_handles.append(Patch(color=after_colors[act],  label=f"{act} (after)"))

    ax.legend(
        handles=legend_handles,
        title="Activity Type",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
