from __future__ import annotations
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.ticker as ticker

# Ensure local imports work
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from data_loader import load_all

ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT / "outputs" / "rq12_final"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def process_before_with_cutoff(df):
    """Applies the 3-year cutoff and cleans the merge time metric."""
    if df.empty:
        return pd.Series(dtype=float)
    
    # 1. Date processing and Cutoff
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    cutoff = df["date"].max() - pd.DateOffset(years=3)
    df = df[df["date"] >= cutoff]
    
    # 2. Metric cleaning
    if "time_to_merge_hours" not in df.columns:
        return pd.Series(dtype=float)
    
    series = df["time_to_merge_hours"].dropna()
    return series[series > 0]

def get_clean_time_series(df):
    """Basic cleaning for 'After' groups (no cutoff needed)."""
    if df.empty or "time_to_merge_hours" not in df.columns:
        return pd.Series(dtype=float)
    
    series = df["time_to_merge_hours"].dropna()
    return series[series > 0]

def summarize(series):
    """Returns a list of summary statistics."""
    if series.empty:
        return [0, 0, 0, 0, 0, 0]
    return [
        len(series),
        series.mean(),
        series.median(),
        series.std(),
        series.min(),
        series.max()
    ]

def main():
    # 1. Load data
    data = load_all()

    # 2. Extract and clean time series with the specific logic for each group
    time_b = process_before_with_cutoff(data["pull_requests_before"])
    time_h = get_clean_time_series(data["pull_requests_after_human"])
    time_a = get_clean_time_series(data["pull_requests_after_agent"])

    if all(t.empty for t in [time_b, time_h, time_a]):
        print("No merge time data found.")
        return

    # 3. Create long-form DataFrame for plotting
    plot_data = pd.concat([
        pd.DataFrame({"Hours": time_b, "Period": "Pre-agent"}),
        pd.DataFrame({"Hours": time_h, "Period": "Post-agent Human"}),
        pd.DataFrame({"Hours": time_a, "Period": "Post-agent Agent"})
    ], ignore_index=True)

    # 4. Summary Statistics Table
    df_stats = pd.DataFrame({
        "Metric": ["Count", "Mean (h)", "Median (h)", "Std Dev", "Min", "Max"],
        "pre-": summarize(time_b),
        "After Human": summarize(time_h),
        "After Agent": summarize(time_a)
    })
    print("\n" + "="*20 + " MERGE TIME STATISTICS (3Y CUTOFF) " + "="*20)
    print(df_stats.to_string(index=False))
# 5. Plotting
    plt.figure(figsize=(14, 8))
    palette = ["#FFDE21", "#3498DB", "#2ECC71"] 
    
    ax = sns.boxplot(
        x="Period",
        y="Hours",
        data=plot_data,
        showfliers=False,
        palette=palette
    )

    # Set to Log Scale for visualization
    ax.set_yscale("log")
    
 
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

    # Labels and Styling
    plt.ylabel("Hours to Merge (Log Scale)", fontsize=24)
    plt.xlabel("Period", fontsize=24)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5, which="major")

    # Save
    outpath = PLOTS_DIR / "rq2_boxplot_merge_time_3groups_3y_cutoff.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"\n[rq2] Plot saved to: {outpath}")
    plt.show()

if __name__ == "__main__":
    main()