import pandas as pd
from pathlib import Path

# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parents[2]   # adjust if needed
INPUT_SUMMARY = ROOT / "outputs" / "rq5" / "tables"
OUT_LATEX = ROOT / "outputs" / "rq5" / "tables" / "mean_weights_table.tex"
OUT_LATEX.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load summary files (topic summaries for each text type)
# We only need *combined* dataset version. You may change this
# to load only the dataset you care about.
# ============================================================

# If you generated a "combined" summary, specify it here.
# If not, script will concatenate all summaries and average them.
summary_files = list(INPUT_SUMMARY.glob("rq5_topic_summary_*.csv"))

if not summary_files:
    raise FileNotFoundError("No topic summary files found in rq5/tables")


dfs = [pd.read_csv(f) for f in summary_files]

# Combine by weighted mean (optional)
combined_df = pd.concat(dfs, ignore_index=True)

# ============================================================
# Compute mean across all text types for each topic
# ============================================================
grouped = (
    combined_df.groupby("topic_label")[["mean_weight_before", "mean_weight_after"]]
    .mean()
    .reset_index()
)

# Round for LaTeX
grouped = grouped.round(4)

# ============================================================
# Build LaTeX table
# ============================================================

latex_table = r"""\begin{table}[h]
\centering
\caption{Mean semantic similarity weights for each topic before and after the introduction of coding agents.}
\label{tab:mean_weights}
\begin{tabular}{lcc}
\toprule
\textbf{Topic} & \textbf{Mean Weight (Before)} & \textbf{Mean Weight (After)} \\
\midrule
"""

for _, row in grouped.iterrows():
    latex_table += f"{row['topic_label']} & {row['mean_weight_before']} & {row['mean_weight_after']} \\\\\n"

latex_table += r"""\bottomrule
\end{tabular}
\end{table}
"""

# ============================================================
# Write LaTeX file
# ============================================================
with open(OUT_LATEX, "w", encoding="utf-8") as f:
    f.write(latex_table)

print(f"✔ LaTeX table saved → {OUT_LATEX}")
