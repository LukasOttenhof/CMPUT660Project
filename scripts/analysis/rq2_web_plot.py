from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_loader import load_all
import seaborn as sns
import matplotlib.ticker as ticker
from scipy.stats import chi2_contingency

pd.options.mode.chained_assignment = None

ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT / "outputs" / "RQ2" / "plots"
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

def calculate_avg_per_dev(df):
    """Calculates the average activity per developer per month for a given dataframe."""
    if df.empty or 'author' not in df.columns:
        return 0
    
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None) 
        df['month'] = df['date'].dt.to_period('M')
    
    monthly_data = df.groupby('month').agg(
        activity_count=('author', 'size'),
        dev_count=('author', 'nunique')
    ).reset_index()

    monthly_data['activity_per_dev'] = np.where(
        monthly_data['dev_count'] > 0,
        monthly_data['activity_count'] / monthly_data['dev_count'],
        0
    )

    return monthly_data['activity_per_dev'].mean()

def build_three_way_ratios(data):
    results = {
        "before": {},
        "after_h": {},
        "after_a": {}
    }

    for metric in METRICS:
        key, subtype = METRIC_MAP[metric]
        
        df_b = data[f"{key}_before"].copy()
        df_h = data[f"{key}_after_human"].copy()
        df_a = data[f"{key}_after_agent"].copy()

        if subtype:
            df_b = df_b[df_b["activity_type"] == subtype]
            df_h = df_h[df_h["activity_type"] == subtype]
            df_a = df_a[df_a["activity_type"] == subtype]

        results["before"][metric] = calculate_avg_per_dev(df_b)
        results["after_h"][metric] = calculate_avg_per_dev(df_h)
        results["after_a"][metric] = calculate_avg_per_dev(df_a)

    ratios = {}
    for group in results:
        total_val = sum(results[group].values())
        ratios[group] = {k: (v / total_val * 100 if total_val > 0 else 0) for k, v in results[group].items()}

    return ratios, results

def calculate_cramers_v(chi2, n, shape):
    """Calculates Cramer's V for a contingency table."""
    phi2 = chi2 / n
    r, k = shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

def perform_stat_analysis(raw_avgs, group1, group2, label):
    """Performs Chi-Square and Cramer's V on the distribution of activities."""
    # Create contingency table from raw average counts
    obs1 = [raw_avgs[group1][m] for m in METRICS]
    obs2 = [raw_avgs[group2][m] for m in METRICS]
    
    contingency_table = np.array([obs1, obs2])
    
    # chi2_contingency handles the expected frequencies automatically
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # N is the sum of all observations in the table
    n = np.sum(contingency_table)
    v = calculate_cramers_v(chi2, n, contingency_table.shape)
    
    return {
        "Comparison": label,
        "Chi2": f"{chi2:.4f}",
        "p-value": f"{p:.4e}",
        "Cramer's V": f"{v:.4f}",
        "Significant": "Yes" if p < 0.05 else "No"
    }

def plot_radar_three_way(ratios, title, filename):
    raw_labels = list(ratios["before"].keys())
    labels = [l.replace('_', ' ').title().replace('Prs', 'PRs') for l in raw_labels]
    
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(16, 16)) 
    ax = fig.add_axes([0.1, 0.15, 0.8, 0.7], polar=True)

    groups = [
        ("before", "Before (Full)", "#FFDE21", "#FDEE9A"), 
        ("after_h", "After (Human)", "#3498DB", "#85C1E9"), 
        ("after_a", "After (Agent)", "#2ECC71", "#A9DFBF"), 
    ]

    max_val = 0
    for key, label, l_color, f_color in groups:
        values = list(ratios[key].values())
        max_val = max(max_val, max(values))
        values += values[:1]
        
        ax.plot(angles, values, color=l_color, linewidth=4, label=label)
        ax.fill(angles, values, color=f_color, alpha=0.3)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=22, fontweight='bold')
    ax.tick_params(axis='x', which='major', pad=35) 

    ax.set_yticks(np.linspace(0, max_val, 5))
    ax.set_yticklabels([f"{int(tick)}%" for tick in ax.get_yticks()], fontsize=14)

    ax.set_title(title, size=32, pad=70, fontweight='bold')
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), fontsize=22, ncol=3)
    ax.grid(True, linestyle='--', alpha=0.6)

    outpath = PLOTS_DIR / filename
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[rq2] Saved 3-way radar chart -> {outpath}")

def main():
    data = load_all()
    ratios, raw_avgs = build_three_way_ratios(data)

    plot_radar_three_way(ratios, "", "permonth.png")

    # 1. Raw Averages Table
    summary_df = pd.DataFrame({
        "Metric": METRICS,
        "Before": [f"{raw_avgs['before'][m]:.2f}" for m in METRICS],
        "After Human": [f"{raw_avgs['after_h'][m]:.2f}" for m in METRICS],
        "After Agent": [f"{raw_avgs['after_a'][m]:.2f}" for m in METRICS],
    })
    
    print("\n" + "="*20 + " RAW AVG ACTIVITY PER DEV PER MONTH " + "="*20)
    print(summary_df.to_string(index=False))

    # 2. Statistical Analysis (Chi-Square + Cramer's V)
    stats_results = []
    stats_results.append(perform_stat_analysis(raw_avgs, "before", "after_a", "Before vs After Agent"))
    stats_results.append(perform_stat_analysis(raw_avgs, "before", "after_h", "Before vs After Human"))
    stats_results.append(perform_stat_analysis(raw_avgs, "after_h", "after_a", "After Human vs After Agent"))
    
    stats_df = pd.DataFrame(stats_results)
    print("\n" + "="*20 + " STATISTICAL SIGNIFICANCE (RATIO SHIFTS) " + "="*20)
    print(stats_df.to_string(index=False))

    # 3. Ratios (Percentages) Table
    percent_df = pd.DataFrame({
        "Metric": METRICS,
        "Before (%)": [f"{ratios['before'][m]:.2f}%" for m in METRICS],
        "After Human (%)": [f"{ratios['after_h'][m]:.2f}%" for m in METRICS],
        "After Agent (%)": [f"{ratios['after_a'][m]:.2f}%" for m in METRICS],
    })

    print("\n" + "="*20 + " ACTIVITY PERCENTAGE DISTRIBUTION " + "="*20)
    print(percent_df.to_string(index=False))

if __name__ == "__main__":
    main()