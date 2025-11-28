# scripts/analysis/table_utils.py

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


# ----------------------------------------------------------------------
# BEFORE/AFTER SUMMARY TABLE
# ----------------------------------------------------------------------
def summarize_before_after(before: pd.Series, after: pd.Series) -> pd.DataFrame:
    before = pd.Series(before).dropna()
    after = pd.Series(after).dropna()

    def stats(x: pd.Series):
        return pd.Series({
            "mean": x.mean(),
            "median": x.median(),
            "q1": x.quantile(0.25),
            "q3": x.quantile(0.75),
            "variance": x.var(),
            "std": x.std(),
            "count": len(x),
        })

    df = pd.DataFrame({
        "before": stats(before),
        "after": stats(after)
    })

    df["diff"] = df["after"] - df["before"]

    return df

# ----------------------------------------------------------------------
# SAVE TABLES (CSV + LaTeX ACM STYLE)
# ----------------------------------------------------------------------
def save_table(df: pd.DataFrame, name: str, outdir: Path) -> None:
    r"""
    Save a DataFrame as:
      - CSV   (unmodified)
      - LaTeX (ACM-style using booktabs)

    Parameters
    ----------
    df : pandas DataFrame
    name : str
        Base filename (no extension)
    outdir : Path
        Directory to save into (e.g., ROOT/outputs/tables)
    """
    if df is None or df.empty:
        print(f"[table_utils] Skipping empty table: {name}")
        return

    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"{name}.csv"
    tex_path = outdir / f"{name}.tex"

    # --- Save CSV ---
    df.to_csv(csv_path, index=False)

    # --- Save LaTeX (ACM style) ---
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="lrrrrr",  # left, right, right, right, right, right
        longtable=False,
        caption=f"{name.replace('_', ' ').title()}",
        label=f"tab:{name}",
        bold_rows=False,
        multicolumn=False,
    )

    # Insert booktabs manually if needed
    latex = latex.replace("\\toprule", "\\hline")
    latex = latex.replace("\\midrule", "\\hline")
    latex = latex.replace("\\bottomrule", "\\hline")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"[table_utils] Saved table → {name}")
