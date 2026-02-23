from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_loader import load_all
import seaborn as sns

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
    # 1. Clean labels
    raw_labels = list(before.keys())
    labels = [l.replace('_', ' ').title().replace('Prs', 'PRs') for l in raw_labels]
    
    N = len(labels)
    values_before = list(before.values())
    values_after = list(after.values())

    # Close the loop
    values_before += values_before[:1]
    values_after += values_after[:1]
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # 2. Setup Figure
    fig = plt.figure(figsize=(14, 14)) 
    # Centering the plot area
    ax = fig.add_axes([0.1, 0.15, 0.8, 0.7], polar=True)

    color_before_line = "#E74C3C" 
    color_before_fill = "#F1948A"  
    color_after_line = "#3498DB"   
    color_after_fill = "#85C1E9"  

    ax.plot(angles, values_before, color=color_before_line, linewidth=4, label="Before Agents")
    ax.fill(angles, values_before, color=color_before_fill, alpha=0.4)

    ax.plot(angles, values_after, color=color_after_line, linewidth=4, label="After Agents")
    ax.fill(angles, values_after, color=color_after_fill, alpha=0.4)

    # 3. Anchoring Labels to the Spokes
    # Using 'va' and 'ha' here ensures they are centered on the coordinate
    ax.set_thetagrids(
        np.degrees(angles[:-1]), 
        labels, 
        fontsize=28, 
        fontweight='bold',
        ha='center', 
        va='center'
    )
    
    # 4. Uniform Padding
    # This moves all labels out by a fixed amount of points from the edge
    ax.tick_params(axis='x', which='major', pad=45) 

    # Title & Legend
    ax.set_title(title, size=32, pad=70, fontweight='bold')
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), fontsize=28, ncol=2)

    # Radial grid cleanup
    ax.grid(True, linestyle='--', linewidth=1.5, alpha=0.5)
    ax.tick_params(axis='y', labelsize=18, labelcolor='gray')

    # Save
    outpath = PLOTS_DIR / filename
    plt.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=1.0)
    plt.close()
    print(f"[rq2] Saved anchored radar chart -> {outpath}")

def main():
    data = load_all()

    #Time slicing
    print("Applying 3-year filter to 'Before' datasets...")
    
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

    before_ratios, after_ratios, before_avg, after_avg = build_monthly_dev_ratios(data)

    plot_radar(
        before_ratios, after_ratios,
        "",
        "permonth.png"
    )

    #Monthly averages
    avg_df = pd.DataFrame({
        "Metric": list(before_avg.keys()),
        "Before (Avg/Dev/Month)": [f"{v:.2f}" for v in before_avg.values()],
        "After (Avg/Dev/Month)":  [f"{v:.2f}" for v in after_avg.values()]
    })
    print("\n================ RAW ACTIVITY PER DEVELOPER PER MONTH ================\n")
    print(avg_df.to_string(index=False))

    #Ratios as table
    ratio_df = pd.DataFrame({
        "Metric": list(before_ratios.keys()),
        "Before (Ratio)": [f"{v:.3f}" for v in before_ratios.values()],
        "After (Ratio)":  [f"{v:.3f}" for v in after_ratios.values()]
    })
    print("\n================ NORMALIZED ACTIVITY RATIOS (FOR RADAR) ================\n")
    print(ratio_df.to_string(index=False))


if __name__ == "__main__":
    main()