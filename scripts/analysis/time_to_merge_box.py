from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = Path("inputs/processed")
PLOTS_DIR = ROOT / "outputs" / "rq12_final"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_process(period):
    file_name = f"pull_requests_{period}.parquet"
    fpath = INPUT_DIR / file_name
    if not fpath.exists():
        print(f"Missing file: {file_name}")
        return pd.DataFrame()

    df = pd.read_parquet(fpath)
    date_col = 'date' if 'date' in df.columns else 'created_at'
    if date_col in df.columns:
        df["date"] = pd.to_datetime(df[date_col])
    
    return df

def main():
    df_before = load_and_process("before")
    df_after = load_and_process("after")
    if df_before.empty or df_after.empty:
        return

    if "date" in df_before.columns:
        before_end = df_before["date"].max()
        cutoff_date = before_end - pd.DateOffset(years=3)
        df_before = df_before[df_before["date"] >= cutoff_date]

    time_b = df_before["time_to_merge_hours"].dropna()
    time_b = time_b[time_b > 0]
    time_a = df_after["time_to_merge_hours"].dropna()
    time_a = time_a[time_a > 0]

    #Statistical test
    u_stat, p_value = stats.mannwhitneyu(time_a, time_b, alternative='two-sided')
    n1, n2 = len(time_b), len(time_a)
    var1, var2 = np.var(time_b, ddof=1), np.var(time_a, ddof=1)
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (time_a.mean() - time_b.mean()) / pooled_se if pooled_se > 0 else 0

    #Boxplot
    plot_data = pd.DataFrame({
        "Time to Merge (Hours)": np.concatenate([time_b, time_a]),
        "Period": ["Before (Last 3y)"] * len(time_b) + ["After Agents"] * len(time_a)
    })

    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(
        x="Period",
        y="Time to Merge (Hours)",
        data=plot_data,
        showfliers=False,
        palette=["#E74C3C", "#3498DB"]
    )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)

    plt.ylabel("Hours to Merge (Log Scale)", fontsize=24)
    plt.xlabel("Period", fontsize=24)
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5, which="major")

    outpath = PLOTS_DIR / "rq2_boxplot_time_to_merge_log.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
