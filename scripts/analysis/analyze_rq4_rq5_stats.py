# scripts/analysis/analyze_rq4_rq5_stats.py
from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, fisher_exact
from statsmodels.stats.multitest import multipletests

def cliffs_delta(before, after):
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)

    n1 = len(before)
    n2 = len(after)
    if n1 == 0 or n2 == 0:
        return np.nan

    greater = 0
    less = 0
    for b in before:
        greater += np.sum(b > after)
        less += np.sum(b < after)
    return float((greater - less) / (n1 * n2))


ROOT = Path(__file__).resolve().parents[2]
TABLES_RQ5 = ROOT / "outputs" / "rq5" / "tables"
OUTDIR = ROOT / "outputs" / "rq_stats"
OUTDIR.mkdir(parents=True, exist_ok=True)


def cramers_v(chi2, n, r, k):
    return np.sqrt(chi2 / (n * (min(r - 1, k - 1))))


def cohens_h(p1, p2):
    p1 = np.clip(p1, 1e-9, 1 - 1e-9)
    p2 = np.clip(p2, 1e-9, 1 - 1e-9)
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def cohens_d_indep(before, after):
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    n1, n2 = len(before), len(after)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = before.mean(), after.mean()
    s1, s2 = before.std(ddof=1), after.std(ddof=1)
    df = n1 + n2 - 2
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df)
    if pooled <= 0:
        return np.nan
    return (m2 - m1) / pooled


CATEGORY_LIST = [
    "Feature Development",
    "UI / UX",
    "Bug Fixing & Debugging",
    "Testing & CI",
    "Refactoring & Cleanup",
    "Documentation & Communication",
    "Deployment",
    "Code Review & Collaboration",
]

def slugify(label: str) -> str:
    s = label.lower().replace("&", "and")
    for ch in ["/", "-", " "]:
        s = s.replace(ch, "_")
    return re.sub(r"[^a-z0-9_]", "", s)

WEIGHT_COLS = {label: f"w_{slugify(label)}" for label in CATEGORY_LIST}


# ---------------------------------------------------------------------------
# 1. Topic rates per text type
# ---------------------------------------------------------------------------
def analyze_topic_rates():
    csvs = list(TABLES_RQ5.glob("rq5_topic_summary_*.csv"))
    out = []

    for path in csvs:
        text_type = path.stem.replace("rq5_topic_summary_", "")
        df = pd.read_csv(path)

        topics = sorted(df["topic_label"].unique())
        table = []
        for topic in topics:
            row = df[df.topic_label == topic].iloc[0]
            table.append([int(row["count_before"]), int(row["count_after"])])

        table = np.array(table)
        chi2, p, dof, exp = chi2_contingency(table)
        n = table.sum()
        v = cramers_v(chi2, n, r=len(topics), k=2)

        out.append({
            "text_type": text_type,
            "chi2": chi2,
            "p_value": p,
            "cramers_v": v,
            "n": n
        })

    df_out = pd.DataFrame(out)
    df_out.to_csv(OUTDIR / "topic_stats.csv", index=False)
    return df_out


# ---------------------------------------------------------------------------
# 1B. NEW — Topic rates combined across ALL text types
# ---------------------------------------------------------------------------
def analyze_topic_rates_combined():
    csvs = list(TABLES_RQ5.glob("rq5_topic_summary_*.csv"))

    all_data = []
    for path in csvs:
        df = pd.read_csv(path)
        all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)

    topics = sorted(df_all["topic_label"].unique())
    table = []

    for topic in topics:
        sub = df_all[df_all.topic_label == topic]
        b = int(sub["count_before"].sum())
        a = int(sub["count_after"].sum())
        table.append([b, a])

    table = np.array(table)
    chi2, p, dof, exp = chi2_contingency(table)
    n = table.sum()
    v = cramers_v(chi2, n, r=len(topics), k=2)

    df_out = pd.DataFrame([{
        "text_type": "ALL_COMBINED",
        "chi2": chi2,
        "p_value": p,
        "cramers_v": v,
        "n": n
    }])

    df_out.to_csv(OUTDIR / "topic_stats_combined.csv", index=False)
    return df_out


# ---------------------------------------------------------------------------
# 2. Tone distribution per text type
# ---------------------------------------------------------------------------
def analyze_tone_rates():
    csvs = list(TABLES_RQ5.glob("rq5_tone_summary_*.csv"))
    out = []
    sentiments = ["negative", "neutral", "positive"]

    for path in csvs:
        text_type = path.stem.replace("rq5_tone_summary_", "")
        df = pd.read_csv(path)

        table = []
        for s in sentiments:
            b = df[(df.sentiment_cat == s) & (df.group == "before")]["count"].sum()
            a = df[(df.sentiment_cat == s) & (df.group == "after")]["count"].sum()
            table.append([b, a])

        table = np.array(table)
        chi2, p, dof, exp = chi2_contingency(table)
        n = table.sum()
        v = cramers_v(chi2, n, r=3, k=2)

        out.append({
            "text_type": text_type,
            "chi2": chi2,
            "p_value": p,
            "cramers_v": v,
            "n": n
        })

    df_out = pd.DataFrame(out)
    df_out.to_csv(OUTDIR / "tone_stats.csv", index=False)
    return df_out


# ---------------------------------------------------------------------------
# 2B. NEW — Tone distribution combined across ALL text types
# ---------------------------------------------------------------------------
def analyze_tone_rates_combined():
    csvs = list(TABLES_RQ5.glob("rq5_tone_summary_*.csv"))
    sentiments = ["negative", "neutral", "positive"]

    totals = {s: {"before": 0, "after": 0} for s in sentiments}

    for path in csvs:
        df = pd.read_csv(path)
        for s in sentiments:
            totals[s]["before"] += df[(df.sentiment_cat == s) & (df.group == "before")]["count"].sum()
            totals[s]["after"] += df[(df.sentiment_cat == s) & (df.group == "after")]["count"].sum()

    table = np.array([[totals[s]["before"], totals[s]["after"]] for s in sentiments])
    chi2, p, dof, exp = chi2_contingency(table)
    n = table.sum()
    v = cramers_v(chi2, n, r=3, k=2)

    df_out = pd.DataFrame([{
        "text_type": "ALL_COMBINED",
        "chi2": chi2,
        "p_value": p,
        "cramers_v": v,
        "n": n
    }])

    df_out.to_csv(OUTDIR / "tone_stats_combined.csv", index=False)
    return df_out


# ---------------------------------------------------------------------------
# 3. Tone by topic per text type (unchanged)
# ---------------------------------------------------------------------------
def analyze_tone_by_topic():
    csvs = list(TABLES_RQ5.glob("tone_by_topic/rq5_tone_by_topic_*.csv"))
    if len(csvs) == 0:
        raise RuntimeError("No tone-by-topic CSVs found.")

    sentiments = ["negative", "neutral", "positive"]
    records = []

    for path in csvs:
        text_type = path.stem.replace("rq5_tone_by_topic_", "")
        df = pd.read_csv(path)
        topics = sorted(df["topic_label"].unique())

        for topic in topics:
            sub = df[df.topic_label == topic]
            before = [int(sub[(sub.group == "before") & (sub.sentiment_cat == s)]["count"].sum()) for s in sentiments]
            after  = [int(sub[(sub.group == "after") & (sub.sentiment_cat == s)]["count"].sum())  for s in sentiments]

            table = np.array([before, after])
            if table.sum() == 0:
                continue

            try:
                chi2, p_chi, dof, exp = chi2_contingency(table)
                fisher_needed = (exp < 5).any()
            except ValueError:
                fisher_needed = True

            if fisher_needed:
                collapsed = np.array([
                    [before[0], sum(before[1:])],
                    [after[0], sum(after[1:])]
                ])
                odds, p_val = fisher_exact(collapsed)
                records.append({
                    "topic": topic,
                    "text_type": text_type,
                    "test": "fisher_exact",
                    "chi2": np.nan,
                    "p_value": p_val,
                    "effect": odds
                })

            else:
                chi2, p_val, _, _ = chi2_contingency(table)
                v = cramers_v(chi2, table.sum(), r=2, k=3)
                records.append({
                    "topic": topic,
                    "text_type": text_type,
                    "test": "chi_square",
                    "chi2": chi2,
                    "p_value": p_val,
                    "effect": v
                })

    df_out = pd.DataFrame(records)
    reject, p_adj, _, _ = multipletests(df_out["p_value"], method="fdr_bh")
    df_out["p_adj"] = p_adj
    df_out["reject_H0"] = reject
    df_out.to_csv(OUTDIR / "tone_topic_stats.csv", index=False)
    return df_out


# ---------------------------------------------------------------------------
# 3B. NEW — Tone by topic across ALL text types combined
# ---------------------------------------------------------------------------
def analyze_tone_by_topic_combined():
    csvs = list(TABLES_RQ5.glob("tone_by_topic/rq5_tone_by_topic_*.csv"))
    sentiments = ["negative", "neutral", "positive"]

    # aggregate across files
    agg = {}

    for path in csvs:
        df = pd.read_csv(path)
        for topic in df["topic_label"].unique():
            if topic not in agg:
                agg[topic] = {s: {"before": 0, "after": 0} for s in sentiments}

            sub = df[df.topic_label == topic]
            for s in sentiments:
                agg[topic][s]["before"] += int(sub[(sub.group == "before") & (sub.sentiment_cat == s)]["count"].sum())
                agg[topic][s]["after"]  += int(sub[(sub.group == "after") & (sub.sentiment_cat == s)]["count"].sum())

    records = []

    for topic, sentiment_data in agg.items():
        before = [sentiment_data[s]["before"] for s in sentiments]
        after =  [sentiment_data[s]["after"]  for s in sentiments]

        table = np.array([before, after])

        try:
            chi2, p_chi, dof, exp = chi2_contingency(table)
            fisher_needed = (exp < 5).any()
        except ValueError:
            fisher_needed = True

        if fisher_needed:
            collapsed = np.array([
                [before[0], sum(before[1:])],
                [after[0], sum(after[1:])]
            ])
            odds, p_val = fisher_exact(collapsed)
            records.append({
                "topic": topic,
                "test": "fisher_exact",
                "chi2": np.nan,
                "p_value": p_val,
                "effect": odds
            })
        else:
            chi2, p_val, _, _ = chi2_contingency(table)
            v = cramers_v(chi2, table.sum(), r=2, k=3)
            records.append({
                "topic": topic,
                "test": "chi_square",
                "chi2": chi2,
                "p_value": p_val,
                "effect": v
            })

    df_out = pd.DataFrame(records)
    reject, p_adj, _, _ = multipletests(df_out["p_value"], method="fdr_bh")
    df_out["p_adj"] = p_adj
    df_out["reject_H0"] = reject
    df_out.to_csv(OUTDIR / "tone_topic_stats_combined.csv", index=False)
    return df_out


# ---------------------------------------------------------------------------
# 4. Mean weight tests — already include pooled data
# ---------------------------------------------------------------------------
def analyze_mean_weights():
    DOC_FILES = {
        "commit_messages": TABLES_RQ5 / "rq5_doc_topics_commit_messages.csv",
        "pr_bodies": TABLES_RQ5 / "rq5_doc_topics_pr_bodies.csv",
        "issue_bodies": TABLES_RQ5 / "rq5_doc_topics_issue_bodies.csv",
        "review_comments": TABLES_RQ5 / "rq5_doc_topics_review_comments.csv",
    }

    records_types = []
    combined_store = {topic: {"before": [], "after": []} for topic in CATEGORY_LIST}

    for text_type, path in DOC_FILES.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)

        for topic in CATEGORY_LIST:
            w_col = WEIGHT_COLS[topic]
            if w_col not in df.columns:
                continue

            before_vals = df[df["group"] == "before"][w_col].dropna().values
            after_vals = df[df["group"] == "after"][w_col].dropna().values
            if len(before_vals) < 2 or len(after_vals) < 2:
                continue

            mean_b = float(before_vals.mean())
            mean_a = float(after_vals.mean())
            delta = mean_a - mean_b
            t_stat, p_val = ttest_ind(before_vals, after_vals, equal_var=False)
            d = cohens_d_indep(before_vals, after_vals)

            records_types.append({
                "topic": topic,
                "text_type": text_type,
                "mean_before": mean_b,
                "mean_after": mean_a,
                "delta": delta,
                "p_value": p_val,
                "cohens_d": d,
                "n_before": len(before_vals),
                "n_after": len(after_vals)
            })

            combined_store[topic]["before"].append(before_vals)
            combined_store[topic]["after"].append(after_vals)

    weight_df_types = pd.DataFrame(records_types)
    if not weight_df_types.empty:
        reject, p_adj, _, _ = multipletests(weight_df_types["p_value"], method="fdr_bh")
        weight_df_types["p_adj"] = p_adj
        weight_df_types["reject_H0"] = reject

    combined_records = []
    for topic in CATEGORY_LIST:
        before_list = combined_store[topic]["before"]
        after_list = combined_store[topic]["after"]
        if not before_list or not after_list:
            continue

        before_vals = np.concatenate(before_list)
        after_vals = np.concatenate(after_list)

        mean_b = float(before_vals.mean())
        mean_a = float(after_vals.mean())
        delta = mean_a - mean_b
        t_stat, p_val = ttest_ind(before_vals, after_vals, equal_var=False)
        d = cohens_d_indep(before_vals, after_vals)

        combined_records.append({
            "topic": topic,
            "mean_before": mean_b,
            "mean_after": mean_a,
            "delta": delta,
            "p_value": p_val,
            "cohens_d": d,
            "n_before": len(before_vals),
            "n_after": len(after_vals)
        })

    weight_df_combined = pd.DataFrame(combined_records)
    if not weight_df_combined.empty:
        reject, p_adj, _, _ = multipletests(weight_df_combined["p_value"], method="fdr_bh")
        weight_df_combined["p_adj"] = p_adj
        weight_df_combined["reject_H0"] = reject

    weight_df_types.to_csv(OUTDIR / "weights_by_texttype.csv", index=False)
    weight_df_combined.to_csv(OUTDIR / "weights_pooled.csv", index=False)

    return weight_df_types, weight_df_combined


# ---------------------------------------------------------------------------
# LaTeX Summary (unchanged)
# ---------------------------------------------------------------------------
def write_latex_summary(topic_df, tone_df, tone_topic_df,
                        weight_df_combined, weight_df_types,
                        topic_df_combined,
                        tone_df_combined,
                        tone_topic_df_combined):

    out = OUTDIR / "stats_summary_latex.tex"

    with open(out, "w") as f:
        f.write("% Auto-generated RQ4/RQ5 statistical results\n")
        f.write("\\section{Statistical Tests for Topics, Tone, and Semantic Weights}\n")

        def write_table(df, title):
            f.write(f"\\subsection*{{{title}}}\n")
            f.write("\\begin{tabular}{lccc}\n")
            f.write("\\toprule\nText Type & $\\chi^2$ & p-value & Cramer's V\\\\\n")
            f.write("\\midrule\n")
            for _, row in df.iterrows():
                f.write(
                    f"{row['text_type']} & {row['chi2']:.3f} & "
                    f"{row['p_value']:.3e} & {row['cramers_v']:.3f}\\\\\n"
                )
            f.write("\\bottomrule\n\\end{tabular}\n\n")

        write_table(topic_df, "Topic Distribution Before vs After (Per Text Type)")
        write_table(topic_df_combined, "Topic Distribution Before vs After (All Combined)")

        write_table(tone_df, "Tone Distribution Before vs After (Per Text Type)")
        write_table(tone_df_combined, "Tone Distribution Before vs After (All Combined)")

        f.write("\\subsection*{Tone by Topic (Per Text Type)}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\nTopic & Text Type & $\\chi^2$ & p(FDR) & Effect\\\\\n")
        f.write("\\midrule\n")
        for _, row in tone_topic_df.iterrows():
            eff = row.get("effect", np.nan)
            chi_str = "-" if pd.isna(row["chi2"]) else f"{row['chi2']:.3f}"
            f.write(
                f"{row['topic']} & {row['text_type']} & {chi_str} & "
                f"{row['p_adj']:.3e} & {eff:.3f}\\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\n")

        f.write("\\subsection*{Tone by Topic (All Combined)}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\nTopic & $\\chi^2$ & p(FDR) & Effect\\\\\n")
        f.write("\\midrule\n")
        for _, row in tone_topic_df_combined.iterrows():
            eff = row.get("effect", np.nan)
            chi_str = "-" if pd.isna(row["chi2"]) else f"{row['chi2']:.3f}"
            f.write(
                f"{row['topic']} & {chi_str} & "
                f"{row['p_adj']:.3e} & {eff:.3f}\\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\n")

        f.write("\\subsection*{Mean Topic Weights (All Text Types Pooled)}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\nTopic & Before & After & $\\Delta$ & p(FDR) & d\\\\\n")
        f.write("\\midrule\n")
        for _, row in weight_df_combined.iterrows():
            f.write(
                f"{row['topic']} & {row['mean_before']:.3f} & "
                f"{row['mean_after']:.3f} & {row['delta']:.3f} & "
                f"{row['p_adj']:.3e} & {row['cohens_d']:.3f}\\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\n")

        f.write("\\subsection*{Mean Topic Weights by Text Type}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\nTopic & Text Type & Before & After & $\\Delta$ & p(FDR) & d\\\\\n")
        f.write("\\midrule\n")
        for _, row in weight_df_types.iterrows():
            f.write(
                f"{row['topic']} & {row['text_type']} & {row['mean_before']:.3f} & "
                f"{row['mean_after']:.3f} & {row['delta']:.3f} & "
                f"{row['p_adj']:.3e} & {row['cohens_d']:.3f}\\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\n")

    print(f"[stats] Wrote LaTeX summary → {out}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run():
    print("[stats] Topic Rates (per type)...")
    topic_df = analyze_topic_rates()

    print("[stats] Topic Rates (combined)...")
    topic_df_combined = analyze_topic_rates_combined()

    print("[stats] Tone Rates (per type)...")
    tone_df = analyze_tone_rates()

    print("[stats] Tone Rates (combined)...")
    tone_df_combined = analyze_tone_rates_combined()

    print("[stats] Tone by Topic (per type)...")
    tone_topic_df = analyze_tone_by_topic()

    print("[stats] Tone by Topic (combined)...")
    tone_topic_df_combined = analyze_tone_by_topic_combined()

    print("[stats] Mean Weight Tests...")
    weight_df_types, weight_df_combined = analyze_mean_weights()

    print("[stats] Writing LaTeX summary...")
    write_latex_summary(
        topic_df, tone_df, tone_topic_df,
        weight_df_combined, weight_df_types,
        topic_df_combined,
        tone_df_combined,
        tone_topic_df_combined,
    )

    print("[stats] Done!")


if __name__ == "__main__":
    run()
