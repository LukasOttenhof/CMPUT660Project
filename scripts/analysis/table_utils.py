from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


STATS = ["mean", "median", "p25", "p75", "variance", "std"]

def compute_stats(series: pd.Series) -> dict:
    if series.empty:
        return {s: np.nan for s in STATS}

    return {
        "mean": series.mean(),
        "median": series.median(),
        "p25": series.quantile(0.25),
        "p75": series.quantile(0.75),
        "variance": series.var(),
        "std": series.std(),
    }


def summarize_before_after(before: pd.Series, after: pd.Series) -> pd.DataFrame:
    before = pd.to_numeric(before, errors='coerce').dropna()
    after = pd.to_numeric(after, errors='coerce').dropna()

    num_before = before.sum()
    num_after = after.sum()
    denom_before = before.count()
    denom_after = after.count()

    stats_b = compute_stats(before)
    stats_a = compute_stats(after)
    stats_d = {k: stats_a[k] - stats_b[k] for k in STATS}

    before_row = {
        "n_numerator": num_before,
        "n_denominator": denom_before,
        **stats_b
    }

    after_row = {
        "n_numerator": num_after,
        "n_denominator": denom_after,
        **stats_a
    }

    diff_row = {
        "n_numerator": num_after - num_before,
        "n_denominator": denom_after - denom_before,
        **stats_d
    }

    df = pd.DataFrame(
        [before_row, after_row, diff_row],
        index=["before", "after", "diff"]
    )

    return df

#CSV and Latex
def save_table(df: pd.DataFrame, name: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.csv"
    df.to_csv(path, float_format="%.4f")
    print(f"[table_utils] Saved table → {name}")
    path = outdir / f"{name}.tex"

    df_fmt = df.copy()
    df_fmt = df_fmt.applymap(
        lambda x: f"{x:.3f}" if isinstance(x, (float, int)) else x
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l" + "r" * len(df.columns) + "}\n")
        f.write("\\toprule\n")

        f.write("Period & " + " & ".join(df.columns) + " \\\\\n")
        f.write("\\midrule\n")

        for idx, row in df_fmt.iterrows():
            f.write(f"{idx} & " + " & ".join(row.values) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(f"\\caption{{Statistical summary for {name}.}}")
        f.write(f"\\label{{tab:{name}}}\n")
        f.write("\\end{table}\n")

    print(f"[table_utils] Saved LaTeX table → {name}.tex")