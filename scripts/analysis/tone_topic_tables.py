from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.colors as mc
import colorsys

# Updated global styling
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
    "figure.titlesize": 28
})

ROOT = Path(__file__).resolve().parents[2]
# Added a specific directory for the LaTeX output
TABLE_OUT_DIR = ROOT / "outputs" / "rq5" / "latex_tables"
TONE_TABLES = ROOT / "outputs" / "rq5" / "tables"
PLOT_DIR = ROOT / "outputs" / "rq5" / "plots"

TABLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SENTIMENTS = ["negative", "neutral", "positive"]
GROUPS = ["before", "after_human", "after_agent"]

# ... [Keep your existing color/palette functions here] ...

TONE_FILES = {
    "Commit Messages": "rq5_tone_commit_messages.csv",
    "Issues": "rq5_tone_issue_bodies.csv",
    "Pull Request Bodies": "rq5_tone_pr_bodies.csv",
    "Review Comments": "rq5_tone_review_comments.csv",
}

def generate_latex_tables(full_df: pd.DataFrame):
    """Generates LaTeX tables for each sentiment category."""
    
    for sentiment in SENTIMENTS:
        table_lines = []
        table_lines.append("\\begin{table}[t]")
        table_lines.append("\\small\n\\setlength{\\tabcolsep}{3pt}\n\\centering")
        table_lines.append(f"\\caption{{{sentiment.title()} sentiment distribution across text types.}}")
        table_lines.append("\\resizebox{\\columnwidth}{!}{")
        table_lines.append("\\begin{tabular}{lrrrrrrr}")
        table_lines.append("\\toprule")
        # Header: B=Before, AH=After Human, AA=After Agent
        table_lines.append("\\textbf{Text Source} & \\textbf{B} & \\textbf{AH} & \\textbf{AA} & \\textbf{B\\%} & \\textbf{AH\\%} & \\textbf{AA\\%} & \\textbf{$\\Delta$ AA} \\\\")
        table_lines.append("\\midrule")

        # Prepare summary row (Combined)
        summary_data = {g: {'count': 0, 'total': 0} for g in GROUPS}

        for ds_name in TONE_FILES.keys():
            row_bits = [ds_name]
            ds_df = full_df[full_df['dataset'] == ds_name]
            
            vals = {}
            for g in GROUPS:
                subset = ds_df[ds_df['group'] == g]
                # Filter for specific sentiment
                s_subset = subset[subset['sentiment_cat'] == sentiment]
                
                count = s_subset['count'].sum() if not s_subset.empty else 0
                total = subset['total'].iloc[0] if not subset.empty else 0
                share = (count / total * 100) if total > 0 else 0
                
                vals[g] = {'count': count, 'share': share}
                summary_data[g]['count'] += count
                summary_data[g]['total'] += total

            # Add counts and percentages to row
            row_bits.extend([f"{vals['before']['count']:,}", f"{vals['after_human']['count']:,}", f"{vals['after_agent']['count']:,}"])
            row_bits.extend([f"{vals['before']['share']:.2f}\\%", f"{vals['after_human']['share']:.2f}\\%", f"{vals['after_agent']['share']:.2f}\\%"])
            
            # Delta calculation (After Agent vs Before)
            delta = vals['after_agent']['share'] - vals['before']['share']
            row_bits.append(f"{'+' if delta > 0 else ''}{delta:.2f}\\%")
            
            table_lines.append(" & ".join(row_bits) + " \\\\")

        # Add Combined Row
        table_lines.append("\\midrule")
        comb_row = ["Combined"]
        c_vals = {}
        for g in GROUPS:
            cnt = summary_data[g]['count']
            tot = summary_data[g]['total']
            shr = (cnt / tot * 100) if tot > 0 else 0
            c_vals[g] = shr
            comb_row.append(f"{cnt:,}")
        
        for g in GROUPS:
            comb_row.append(f"{c_vals[g]:.2f}\\%")
            
        c_delta = c_vals['after_agent'] - c_vals['before']
        comb_row.append(f"{'+' if c_delta > 0 else ''}{c_delta:.2f}\\%")
        table_lines.append(" & ".join(comb_row) + " \\\\")

        table_lines.append("\\bottomrule\n\\end{tabular}\n}")
        table_lines.append(f"\\label{{tab:rq5_tone_{sentiment}}}")
        table_lines.append("\\end{table}")

        # Save to file
        with open(TABLE_OUT_DIR / f"table_{sentiment}.tex", "w") as f:
            f.write("\n".join(table_lines))

def plot_stacked_3way(df, xcol, title, path):
    # ... [Keep your existing plotting logic here] ...
    # (No changes needed to this function)
    pass

def run():
    print("[PROCESS] Loading data and preparing tables...")
    dfs = []
    for ds_name, file in TONE_FILES.items():
        fp = TONE_TABLES / file
        if not fp.exists(): continue
        
        df = pd.read_csv(fp)
        if df.empty: continue
            
        counts = df.groupby(['group', 'sentiment_cat']).size().reset_index(name='count')
        totals = df.groupby('group').size().reset_index(name='total')
        
        merged = counts.merge(totals, on='group')
        merged['share'] = merged['count'] / merged['total']
        merged['dataset'] = ds_name
        dfs.append(merged)

    if not dfs:
        print("[ERR] No sentiment data found.")
        return

    full_tidy = pd.concat(dfs, ignore_index=True)

    # 1. Generate Plots
    plot_stacked_3way(
        full_tidy, "dataset",
        "Sentiment Distribution by Period and Contributor Type",
        PLOT_DIR / "sentiment_by_dataset_3way.png"
    )

    # 2. Generate LaTeX Tables
    generate_latex_tables(full_tidy)
    
    print(f"[OK] Plots saved to: {PLOT_DIR}")
    print(f"[OK] LaTeX tables saved to: {TABLE_OUT_DIR}")

if __name__ == "__main__":
    run()