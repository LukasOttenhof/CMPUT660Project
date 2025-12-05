import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARIES = ROOT / "outputs" / "rq5" / "tables"

OUT_LATEX = ROOT / "outputs" / "rq5" / "combined_topic_table.tex"

# Escape LaTeX characters
def fmt_topic(t: str) -> str:
    if not isinstance(t, str):
        return ""
    return t.replace("&", r"\&")


def build_latex():
    latex_rows = []

    datasets = {
        "Commit Messages": "rq5_topic_summary_commit_messages.csv",
        "Issues": "rq5_topic_summary_issue_bodies.csv",
        "Pull Request Bodies": "rq5_topic_summary_pr_bodies.csv",
        "Review Comments": "rq5_topic_summary_review_comments.csv",
    }

    # ------------------------------------------------------------
    # LOAD ALL DATASETS & store per-dataset DataFrames
    # ------------------------------------------------------------
    dfs = {}
    for ds, filename in datasets.items():
        fp = SUMMARIES / filename
        if not fp.exists():
            raise RuntimeError(f"Missing summary file: {fp}")
        dfs[ds] = pd.read_csv(fp)

    # ------------------------------------------------------------
    # COMPUTE COMBINED DATASET
    # ------------------------------------------------------------
    combined_df = pd.concat(dfs.values(), ignore_index=True)

    combined_grouped = (
        combined_df.groupby("topic_label")
        .agg(
            count_before=("count_before", "sum"),
            count_after=("count_after", "sum"),
        )
        .reset_index()
    )

    total_before = combined_grouped["count_before"].sum()
    total_after = combined_grouped["count_after"].sum()

    combined_grouped["share_before"] = combined_grouped["count_before"] / total_before
    combined_grouped["share_after"] = combined_grouped["count_after"] / total_after

    # Insert into dictionary so formatting loop handles it like others
    dfs["Combined"] = combined_grouped

    # ------------------------------------------------------------
    # Begin LaTeX table
    # ------------------------------------------------------------
    header = r"""
\begin{table*}[t]
\centering
\small
\begin{tabular}{llrrrrrr}
\toprule
\textbf{Dataset} & \textbf{Topic} &
\textbf{Count (B)} & \textbf{Count (A)} &
\textbf{$\Delta$ Count} &
\textbf{Share (B)} & \textbf{Share (A)} &
\textbf{$\Delta$ Share (\%)} \\
\midrule
"""
    latex_rows.append(header)

    # ------------------------------------------------------------
    # Write each dataset block (including Combined)
    # ------------------------------------------------------------
    for ds_name, df in dfs.items():
        df = df.sort_values("topic_label")  # consistent ordering

        latex_rows.append(f"\\multirow{{{len(df)}}}{{*}}{{{ds_name}}}")

        for i, row in df.iterrows():
            topic = fmt_topic(row["topic_label"])

            b = int(row["count_before"])
            a = int(row["count_after"])
            d_count = a - b

            sb = row["share_before"]
            sa = row["share_after"]
            delta_share_pct = (sa - sb) * 100

            # First row of dataset block (multirow already printed)
            if i == 0:
                line = (
                    f" & {topic} & {b} & {a} & "
                    f"{d_count:+d} & "
                    f"{sb:.4f} & {sa:.4f} & {delta_share_pct:+.2f}\\% \\\\"
                )
            else:
                line = (
                    f"& {topic} & {b} & {a} & "
                    f"{d_count:+d} & "
                    f"{sb:.4f} & {sa:.4f} & {delta_share_pct:+.2f}\\% \\\\"
                )

            latex_rows.append(line)

        latex_rows.append("\\midrule")

    # ------------------------------------------------------------
    # Close table
    # ------------------------------------------------------------
    footer = r"""
\bottomrule
\end{tabular}
\caption{Combined topic classification results across all datasets, including a unified ``Combined'' dataset.}
\label{tab:combined_topics}
\end{table*}
"""
    latex_rows.append(footer)

    OUT_LATEX.write_text("\n".join(latex_rows), encoding="utf8")
    print(f"[OK] Wrote LaTeX table to: {OUT_LATEX}")


if __name__ == "__main__":
    build_latex()
