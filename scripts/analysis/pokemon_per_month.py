from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_loader import load_all
import seaborn as sns

# Suppress pandas warnings related to setting values on copies and TimeZone conversion
# Note: For production use, it's generally better to fix the root cause (which we did below) 
# but this line can be used as a general safety if other copy-related warnings appear.
pd.options.mode.chained_assignment = None

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

def build_monthly_dev_ratios(data):
    """
    Calculates the average activity per developer per month for 'before' and 'after' periods,
    then normalizes these averages into a ratio for the radar chart.
    
    Returns: (before_ratios, after_ratios, before_avg_per_dev, after_avg_per_dev)
    """
    before_avg_per_dev = {}
    after_avg_per_dev = {}

    for metric in METRICS:
        key, subtype = METRIC_MAP[metric]
        df_b = data[f"{key}_before"].copy()
        df_a = data[f"{key}_after"].copy()

        if subtype:
            df_b = df_b[df_b["activity_type"] == subtype]
            df_a = df_a[df_a["activity_type"] == subtype]
            
        # Ensure 'author' column exists for developer count
        if 'author' not in df_b.columns or 'author' not in df_a.columns:
             print(f"Warning: Skipping {metric} - 'author' column not found.")
             continue
        
        # --- Helper function for one period (Before or After) ---
        def calculate_avg_per_dev(df, period_label):
            df = df.copy()
            
            # 1. Group by Month
            # Explicitly remove timezone info to avoid UserWarning when using .to_period('M')
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None) 
            df['month'] = df['date'].dt.to_period('M')
            
            # Group by month to get total activity and unique dev count
            monthly_data = df.groupby('month').agg(
                activity_count=('author', 'size'),
                dev_count=('author', 'nunique')
            ).reset_index()

            # 2. Calculate Activity Per Developer Per Month
            # Safety check: avoid division by zero
            monthly_data['activity_per_dev'] = np.where(
                monthly_data['dev_count'] > 0,
                monthly_data['activity_count'] / monthly_data['dev_count'],
                0
            )

            # 3. Average across all months in the period
            avg_activity = monthly_data['activity_per_dev'].mean()
            return avg_activity

        # Calculate averages for both periods
        avg_b = calculate_avg_per_dev(df_b, "Before")
        avg_a = calculate_avg_per_dev(df_a, "After")

        before_avg_per_dev[metric] = avg_b
        after_avg_per_dev[metric] = avg_a
        
        # Original print statement (now clean)
        print(f"[rq2] Metric: {metric} | Before Avg/Dev/Month: {avg_b:.4f} | After Avg/Dev/Month: {avg_a:.4f}")

    # --- Normalize to Ratios ---
    sum_b = sum(before_avg_per_dev.values())
    sum_a = sum(after_avg_per_dev.values())

    before_ratios = {}
    after_ratios = {}

    for k in before_avg_per_dev:
        before_ratios[k] = before_avg_per_dev[k] / (sum_b if sum_b > 0 else 1)
        after_ratios[k]  = after_avg_per_dev[k]  / (sum_a if sum_a > 0 else 1)

    # Returning both ratios and raw averages
    return before_ratios, after_ratios, before_avg_per_dev, after_avg_per_dev


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

    # Apply sqrt scaling for visual balance (optional but often helps ratios)
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

    # =========================================================
    # ✂️ TIME SLICING: LIMIT 'BEFORE' DATA TO LAST 3 YEARS
    # =========================================================
    print("Applying 3-year filter to 'Before' datasets...")
    
    # Iterate over all keys in the data dictionary
    for key in data:
        if key.endswith("_before"):
            df = data[key]
            
            if df.empty or "date" not in df.columns:
                continue

            df["date"] = pd.to_datetime(df["date"])
            before_end = df["date"].max()
            cutoff_date = before_end - pd.DateOffset(years=3)

            data[key] = df[df["date"] >= cutoff_date]
    
    # =========================================================

    # --- USE THE NEW FUNCTION HERE TO GET AVGS AND RATIOS ---
    before_ratios, after_ratios, before_avg, after_avg = build_monthly_dev_ratios(data)

    # Plot radar
    plot_radar(
        before_ratios, after_ratios,
        "Developer Activity Before vs After Agents (Per Dev Per Month Ratios)",
        "rq2_radar_developer_activity_per_dev_per_month_ratios.png"
    )

    # =========================================================
    # 📊 Print Raw Averages Per Month Per Dev
    # =========================================================
    avg_df = pd.DataFrame({
        "Metric": list(before_avg.keys()),
        "Before (Avg/Dev/Month)": [f"{v:.2f}" for v in before_avg.values()],
        "After (Avg/Dev/Month)":  [f"{v:.2f}" for v in after_avg.values()]
    })
    print("\n================ RAW ACTIVITY PER DEVELOPER PER MONTH ================\n")
    print(avg_df.to_string(index=False))

    # Print ratios as table
    ratio_df = pd.DataFrame({
        "Metric": list(before_ratios.keys()),
        "Before (Ratio)": [f"{v:.3f}" for v in before_ratios.values()],
        "After (Ratio)":  [f"{v:.3f}" for v in after_ratios.values()]
    })
    print("\n================ NORMALIZED ACTIVITY RATIOS (FOR RADAR) ================\n")
    print(ratio_df.to_string(index=False))


if __name__ == "__main__":
    main()