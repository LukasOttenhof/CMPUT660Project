# scripts/analysis/analyze_rq4_rq5_stats.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
from scipy.stats import fisher_exact


# ----------------------------
# Paths
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]
TABLES_RQ5 = ROOT / "outputs" / "rq5" / "tables"
OUTDIR = ROOT / "outputs" / "rq_stats"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Helper functions
# ----------------------------

def cramers_v(chi2, n, r, k):
    return np.sqrt(chi2 / (n * (min(r - 1, k - 1))))

def cohens_h(p1, p2):
    p1 = np.clip(p1, 1e-9, 1 - 1e-9)
    p2 = np.clip(p2, 1e-9, 1 - 1e-9)
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

# ----------------------------
# 1. Topic distribution before/after
# ----------------------------
def analyze_topic_rates():
    csvs = list(TABLES_RQ5.glob("rq5_topic_summary_*.csv"))
    records = []

    for path in csvs:
        text_type = path.stem.replace("rq5_topic_summary_", "")
        df = pd.read_csv(path)

        topics = sorted(df["topic_label"].unique())

        # build contingency table
        table = []
        for topic in topics:
            row = df[df.topic_label == topic]

            b = int(row["count_before"].iloc[0])
            a = int(row["count_after"].iloc[0])

            table.append([b, a])

        table_np = np.array(table)
        chi2, p, dof, exp = chi2_contingency(table_np)
        n = table_np.sum()
        v = cramers_v(chi2, n, r=len(topics), k=2)

        records.append({
            "text_type": text_type,
            "chi2": chi2,
            "p_value": p,
            "cramers_v": v,
            "n": n
        })

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTDIR / "topic_stats.csv", index=False)
    print("[stats] Wrote topic_stats.csv")
    return df_out

# ----------------------------
# 2. Tone distribution before/after (overall and per text type)
# ----------------------------
def analyze_tone_rates():
    csvs = list(TABLES_RQ5.glob("rq5_tone_summary_*.csv"))
    records = []

    for path in csvs:
        text_type = path.stem.replace("rq5_tone_summary_", "")
        df = pd.read_csv(path)

        # reorder to ensure neg, neu, pos
        sentiments = ["negative", "neutral", "positive"]
        table = []
        for s in sentiments:
            b = df[(df.sentiment_cat == s) & (df.group == "before")]["count"].sum()
            a = df[(df.sentiment_cat == s) & (df.group == "after")]["count"].sum()
            table.append([b, a])

        table_np = np.array(table)
        chi2, p, dof, exp = chi2_contingency(table_np)
        n = table_np.sum()
        v = cramers_v(chi2, n, r=3, k=2)

        records.append({
            "text_type": text_type,
            "chi2": chi2,
            "p_value": p,
            "cramers_v": v,
            "n": n
        })

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTDIR / "tone_stats.csv", index=False)
    print("[stats] Wrote tone_stats.csv")
    return df_out

def analyze_tone_by_topic():
    csvs = list(TABLES_RQ5.glob("tone_by_topic/rq5_tone_by_topic_*.csv"))
    print("[stats] tone-by-topic CSVs found:", csvs)

    if len(csvs) == 0:
        raise RuntimeError("NO tone-by-topic CSVs found. Check folder path.")

    required_cols = {"topic_label", "group", "sentiment_cat", "count"}

    records = []

    for path in csvs:
        text_type = path.stem.replace("rq5_tone_by_topic_", "")
        df = pd.read_csv(path)

        print(f"[stats] Loaded {path}, cols={df.columns}")

        # Validate columns
        if not required_cols.issubset(df.columns):
            raise RuntimeError(
                f"File {path} missing required columns. "
                f"Expected {required_cols}, got {set(df.columns)}"
            )

        sentiments = ["negative", "neutral", "positive"]
        topics = sorted(df["topic_label"].unique())

        for topic in topics:
            sub = df[df.topic_label == topic]

            before_counts = []
            after_counts = []

            for s in sentiments:
                b = int(sub[(sub.group == "before") & (sub.sentiment_cat == s)]["count"].sum())
                a = int(sub[(sub.group == "after") & (sub.sentiment_cat == s)]["count"].sum())
                before_counts.append(b)
                after_counts.append(a)

            table = np.array([before_counts, after_counts])  # shape = 2 × 3

            if table.sum() == 0:
                print(f"[stats] WARNING: Topic '{topic}' has zero total count, skipping.")
                continue

            # Determine test type
            use_fisher = False

            # Compute chi-square expected frequencies
            try:
                chi2, p_chi, dof, expected = chi2_contingency(table)
                if (expected < 5).any() or (expected == 0).any():
                    use_fisher = True
            except ValueError:
                use_fisher = True

            # Perform statistical test
            if use_fisher:
                # Fisher exact ONLY works for 2×2 tables. Ours is 2×3.
                # → fallback: collapse to negative vs non-negative
                collapsed = np.array([
                    [before_counts[0], sum(before_counts[1:])],
                    [after_counts[0], sum(after_counts[1:])]
                ])

                # If still invalid, skip
                if collapsed.sum() == 0 or collapsed.shape != (2, 2):
                    print(f"[stats] Skipping Fisher for {topic} (invalid table).")
                    continue

                odds_ratio, p_val = fisher_exact(collapsed)
                chi2_stat = None
                effect = odds_ratio  # report OR
                effect_name = "odds_ratio"

            else:
                chi2_stat = chi2
                p_val = p_chi
                effect = cramers_v(chi2_stat, table.sum(), r=2, k=3)
                effect_name = "cramers_v"

            records.append({
                "text_type": text_type,
                "topic": topic,
                "test": "fisher_exact" if use_fisher else "chi_square",
                "chi2": chi2_stat,
                "p_value": p_val,
                effect_name: effect,
                "before_neg": before_counts[0],
                "after_neg": after_counts[0],
                "before_neu": before_counts[1],
                "after_neu": after_counts[1],
                "before_pos": before_counts[2],
                "after_pos": after_counts[2],
                "total_n": table.sum()
            })

    df_out = pd.DataFrame(records)

    print("\nDEBUG — df_out BEFORE FDR correction:\n", df_out)

    if df_out.empty:
        raise RuntimeError("df_out is EMPTY. No statistical rows were generated.")

    # FDR correction
    reject, p_adj, _, _ = multipletests(df_out["p_value"], method="fdr_bh")
    df_out["p_adj"] = p_adj
    df_out["reject_H0"] = reject

    # Save final CSV
    outpath = OUTDIR / "tone_topic_stats.csv"
    df_out.to_csv(outpath, index=False)
    print(f"[stats] Wrote tone_topic_stats.csv → {outpath}")

    return df_out

# ----------------------------
# Combine into LaTeX summary
# ----------------------------
def write_latex_summary(topic_df, tone_df, tone_topic_df):
    out = OUTDIR / "stats_summary_latex.tex"

    with open(out, "w") as f:
        f.write("% Auto-generated statistical results\n")
        f.write("\\section{Statistical Tests for Topic, Tone, and Tone-by-Topic}\n")

        def write_table(df, title):
            f.write(f"\\subsection*{{{title}}}\n")
            f.write("\\begin{tabular}{lccc}\n")
            f.write("\\toprule\n")
            f.write("Text Type & $\\chi^2$ & p-value & Cramer's V\\\\\n\\midrule\n")
            for _, row in df.iterrows():
                f.write(f"{row['text_type']} & {row['chi2']:.3f} & {row['p_value']:.3e} & {row['cramers_v']:.3f}\\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\n")

        write_table(topic_df, "Topic Distribution Before vs After")
        write_table(tone_df, "Tone Distribution Before vs After")

        # tone-by-topic section
        f.write("\\subsection*{Tone by Topic}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\nTopic & Text Type & $\\chi^2$ & p (FDR) & Cramer's V\\\\\n\\midrule\n")
        for _, row in tone_topic_df.iterrows():
            f.write(
                f"{row['topic']} & {row['text_type']} & "
                f"{row['chi2']:.3f} & {row['p_adj']:.3e} & {row['cramers_v']:.3f}\\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"[stats] Wrote LaTeX summary → {out}")

# ----------------------------
# Main
# ----------------------------
def run():
    print("[stats] Running topic distribution tests...")
    topic_df = analyze_topic_rates()

    print("[stats] Running tone distribution tests...")
    tone_df = analyze_tone_rates()

    print("[stats] Running tone-per-topic distribution tests...")
    tone_topic_df = analyze_tone_by_topic()

    print("[stats] Writing LaTeX summary...")
    write_latex_summary(topic_df, tone_df, tone_topic_df)

    print("[stats] All statistical tests complete!")

if __name__ == "__main__":
    run()
