from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.ticker as ticker

# ==========================================
# ⚙️ CONFIGURATION & PATHS
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = Path("inputs/processed")
PLOTS_DIR = ROOT / "outputs" / "rq12_final"

# Create output directory if it doesn't exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_process(period):
    file_name = f"pull_requests_{period}.parquet"
    fpath = INPUT_DIR / file_name
    
    if not fpath.exists():
        print(f"⚠️ Missing file: {file_name}")
        return pd.DataFrame()

    df = pd.read_parquet(fpath)
    date_col = 'date' if 'date' in df.columns else 'created_at'
    if date_col in df.columns:
        df["date"] = pd.to_datetime(df[date_col])
    
    return df

def main():
    # 1. Load Data
    print("Loading data...")
    df_before = load_and_process("before")
    df_after = load_and_process("after")

    if df_before.empty or df_after.empty:
        print("❌ Error: Could not load data.")
        return

    # 2. ✂️ Apply 3-Year Filter to 'Before' Data
    if "date" in df_before.columns:
        before_end = df_before["date"].max()
        cutoff_date = before_end - pd.DateOffset(years=3)
        print(f"Filtering 'Before' data to last 3 years (Start: {cutoff_date.date()})")
        df_before = df_before[df_before["date"] >= cutoff_date]

    # 3. Extract 'time_to_merge_hours'
    # Filter out 0s or negative numbers for Log Scale safety
    time_b = df_before["time_to_merge_hours"].dropna()
    time_b = time_b[time_b > 0]
    
    time_a = df_after["time_to_merge_hours"].dropna()
    time_a = time_a[time_a > 0]

    print(f"Data Points -> Before: {len(time_b)}, After: {len(time_a)}")

    # 4. Statistical Test (Mann-Whitney U)
    u_stat, p_value = stats.mannwhitneyu(time_a, time_b, alternative='two-sided')
    
    # Effect Size (Cohen's d on Log-Transformed Data is often more robust, 
    # but we'll stick to standard d on raw data for consistency with previous steps)
    n1, n2 = len(time_b), len(time_a)
    var1, var2 = np.var(time_b, ddof=1), np.var(time_a, ddof=1)
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (time_a.mean() - time_b.mean()) / pooled_se if pooled_se > 0 else 0

    # 5. Create Summary Table
    summary_df = pd.DataFrame({
        "Metric": ["Count", "Mean (Hours)", "Median (Hours)", "Std Dev", "Min", "Max"],
        "Before (Last 3y)": [
            len(time_b), time_b.mean(), time_b.median(), time_b.std(), time_b.min(), time_b.max()
        ],
        "After (Agents)": [
            len(time_a), time_a.mean(), time_a.median(), time_a.std(), time_a.min(), time_a.max()
        ]
    })

    print("\n================ TIME TO MERGE SUMMARY ================\n")
    print(summary_df.to_string(index=False, float_format="%.2f"))
    print(f"\nMann-Whitney P-Value: {p_value:.6f}")
    print(f"Cohen's d Effect Size: {cohens_d:.4f}")

    # 6. Plot Box Plots (Log Scale)
    plot_data = pd.DataFrame({
        "Time to Merge (Hours)": np.concatenate([time_b, time_a]),
        "Period": ["Before (Last 3y)"] * len(time_b) + ["After Agents"] * len(time_a)
    })

    plt.figure(figsize=(10, 6))
    
    # We turn 'showfliers' ON now because Log Scale usually handles outliers better,
    # and they are interesting to see. If it's still too messy, set it back to False.
    ax = sns.boxplot(
        x="Period", 
        y="Time to Merge (Hours)", 
        data=plot_data, 
        showfliers=False, 
        palette=["#E74C3C", "#3498DB"]
    )

    # -------------------------------------------------------
    # 📉 LOG SCALE MAGIC
    # -------------------------------------------------------
    ax.set_yscale("log")
    
    # Format the Y-axis ticks to be readable numbers (1, 10, 100) instead of 10^0, 10^1
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    
    plt.title("Time to Merge Distribution (Log Scale)", fontsize=20, fontweight='bold')
    plt.ylabel("Hours to Merge (Log Scale)", fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5, which="major")
    
    # Save Plot
    outpath = PLOTS_DIR / "rq2_boxplot_time_to_merge_log.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"\n[Plot] Saved log-scale boxplot to -> {outpath}")
    
    plt.show()

if __name__ == "__main__":
    main()