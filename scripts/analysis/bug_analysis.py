import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
BASE = Path("inputs/processed")
PLOTS_DIR = BASE.parent / "outputs" / "pokemon" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Define categories and their keywords
CATEGORIES = {
    "Bug": [
        "fix", "bug", "repair", "resolve",
        "broken", "error", "fail", "hotfix", "issue", "crash",
        "correct", "debug", "restore"
    ],
    "Refactor": [
        "refactor", "clean", "structure", "style", 
        "rename", "move", "format", "lint", "tidy", "simplify", "optimize"
    ],
    "Revert": [
        "revert", "rollback", "undo"
    ]
}

def load_and_filter(file_type, period):
    """
    Loads a specific file type (commit_messages or pr_bodies) for a period.
    Does NOT apply any 3-year filtering (uses full history).
    """
    fpath = BASE / f"{file_type}_{period}.parquet"
    if not fpath.exists():
        print(f"⚠️ Missing {fpath}")
        return pd.DataFrame()
    
    df = pd.read_parquet(fpath)
    
    # 1. Normalize Date Column
    date_col = 'date' if 'date' in df.columns else 'created_at'
    if date_col in df.columns:
        df["date"] = pd.to_datetime(df[date_col])
    else:
        return pd.DataFrame()

    # 2. Normalize Text Column
    if "text" not in df.columns:
        return pd.DataFrame()
        
    return df[["text"]].copy() 

def get_category_stats(df, keywords):
    """Helper to calculate counts and ratios for a specific keyword list."""
    if df.empty:
        return 0, 0, 0.0
    
    # Use non-capturing group (?:...) to silence UserWarning about match groups
    pattern = r"\b(?:" + "|".join(keywords) + r")\b"
    is_match = df["text"].str.contains(pattern, case=False, na=False)
    count = is_match.sum()
    total = len(df)
    ratio = count / total if total > 0 else 0.0
    return count, total, ratio

def analyze_fixes():
    print("Loading and analyzing Text Artifacts (Commits + PR Bodies)...")
    
    # 1. Load Data
    commits_b = load_and_filter("commit_messages", "before")
    commits_a = load_and_filter("commit_messages", "after")
    
    prs_b = load_and_filter("pr_bodies", "before")
    prs_a = load_and_filter("pr_bodies", "after")
    
    # 2. Combine Data Sources
    df_b = pd.concat([commits_b, prs_b])
    df_a = pd.concat([commits_a, prs_a])
    
    # Totals for denominators
    tot_cb = len(commits_b)
    tot_ca = len(commits_a)
    tot_pb = len(prs_b)
    tot_pa = len(prs_a)
    total_b = len(df_b)
    total_a = len(df_a)
    
    print(f"Total Artifacts (Before): {total_b:,}")
    print(f"Total Artifacts (After Agents): {total_a:,}")

    if df_b.empty or df_a.empty:
        print("❌ Error: Insufficient data after loading/filtering.")
        return

    # 3. Analyze Categories
    plot_data = []
    
    print("\n================ MAINTENANCE TYPE ANALYSIS (RQ3) ================")
    print(f"{'Category / Source':<35} | {'Before':<25} | {'After (Agents)':<25}")
    print("-" * 90)

    for cat_name, keywords in CATEGORIES.items():
        # --- Commits ---
        cb_cnt, _, cb_ratio = get_category_stats(commits_b, keywords)
        ca_cnt, _, ca_ratio = get_category_stats(commits_a, keywords)
        
        # --- PRs ---
        pb_cnt, _, pb_ratio = get_category_stats(prs_b, keywords)
        pa_cnt, _, pa_ratio = get_category_stats(prs_a, keywords)
        
        # --- Combined ---
        b_count, _, b_ratio = get_category_stats(df_b, keywords)
        a_count, _, a_ratio = get_category_stats(df_a, keywords)
        
        # Print Group Header
        print(f"🔹 {cat_name}")
        
        # Print Rows
        print(f"   {'Commits Only':<32} | {cb_ratio:.2%} ({cb_cnt}/{tot_cb})    | {ca_ratio:.2%} ({ca_cnt}/{tot_ca})")
        print(f"   {'PR Bodies Only':<32} | {pb_ratio:.2%} ({pb_cnt}/{tot_pb})    | {pa_ratio:.2%} ({pa_cnt}/{tot_pa})")
        print(f"   {'COMBINED':<32} | {b_ratio:.2%} ({b_count}/{total_b})    | {a_ratio:.2%} ({a_count}/{total_a})")
        print("-" * 90)
        
        # Store Combined for plotting
        plot_data.append({"Period": "Before Agents", "Category": cat_name, "Ratio": b_ratio})
        plot_data.append({"Period": "After Agents", "Category": cat_name, "Ratio": a_ratio})

    # 4. Plot Grouped Bar Chart
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(10, 6))
    
    # Create grouped bar chart
    ax = sns.barplot(
        x="Category", 
        y="Ratio", 
        hue="Period", 
        data=df_plot, 
        palette=["#E74C3C", "#3498DB"]
    )
    
    
    plt.title("Shift in Maintenance Activity Types (Commits + PRs)", fontsize=20, fontweight='bold', pad=20)
    plt.ylabel("Proportion of Text Artifacts", fontsize=18)
    plt.xlabel("")
    plt.legend(title="Time Period")
    
    # Add labels - Explicitly format float ratio (0.12) to percentage string (12.2%)
    for container in ax.containers:
        # Check if datavalues attribute exists (matplotlib > 3.4), otherwise use get_height
        if hasattr(container, 'datavalues'):
            labels = [f'{val:.1%}' for val in container.datavalues]
        else:
            labels = [f'{bar.get_height():.1%}' for bar in container]
            
        ax.bar_label(container, labels=labels, padding=3, fontsize=13)

    outpath = PLOTS_DIR / "rq3_maintenance_types_combined.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"\n[Plot] Saved chart to -> {outpath}")
    plt.show()

if __name__ == "__main__":
    analyze_fixes()