from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# ==========================================================
# PATHS / OUTPUT DIRS
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]

TABLES = ROOT / "outputs" / "rq5" / "tables"
PLOTS = ROOT / "outputs" / "rq5" / "plots"
LATEX = ROOT / "outputs" / "rq5" / "latex"
STATS = ROOT / "outputs" / "rq_stats"

PLOTS.mkdir(parents=True, exist_ok=True)
LATEX.mkdir(parents=True, exist_ok=True)
STATS.mkdir(parents=True, exist_ok=True)

# ==========================================================
# IO / HELPERS
# ==========================================================


def load_topic_mapping(path: Path) -> Dict[int, str]:
    df = pd.read_csv(path)
    if "topic_id" not in df.columns or "category" not in df.columns:
        raise ValueError(f"[mapping] Expected columns topic_id,category in {path}")
    df = df.dropna(subset=["topic_id", "category"])
    df["topic_id"] = df["topic_id"].astype(int)
    df["category"] = df["category"].astype(str)
    return dict(zip(df["topic_id"], df["category"]))


def load_and_map_docs(doc_csv: Path, mapping_csv: Path) -> pd.DataFrame:
    docs = pd.read_csv(doc_csv)

    # expected minimal schema
    required = {"topic_id", "group"}
    missing = required - set(docs.columns)
    if missing:
        raise ValueError(f"[docs] Missing columns {sorted(missing)} in {doc_csv}")

    mapping = load_topic_mapping(mapping_csv)

    docs["topic_id"] = pd.to_numeric(docs["topic_id"], errors="coerce")
    docs = docs.dropna(subset=["topic_id"])
    docs["topic_id"] = docs["topic_id"].astype(int)

    docs["group"] = docs["group"].astype(str)

    # drop noise topics if present
    docs = docs[docs["topic_id"] != -1].copy()

    docs["category"] = docs["topic_id"].map(mapping)
    docs = docs.dropna(subset=["category"]).copy()

    # enforce the 3-group system
    valid_groups = {"before", "after_human", "after_agent"}
    docs = docs[docs["group"].isin(valid_groups)].copy()

    return docs


import re

_LATEX_SPECIALS = [
    ("&",  r"\&"),
    ("%",  r"\%"),
    ("$",  r"\$"),
    ("#",  r"\#"),
    ("_",  r"\_"),
    ("{",  r"\{"),
    ("}",  r"\}"),
]

def latex_escape(text: str) -> str:
    """
    Escape LaTeX special chars, but DO NOT double-escape things that are already escaped
    (e.g., keeps '\\_' as '\\_' instead of turning it into '\\\\_').
    """
    if text is None:
        return ""
    text = str(text)

    # Escape only if the char is NOT already preceded by a backslash.
    # Example: _  -> \_
    #          \_ -> \_  (unchanged)
    for ch, repl in _LATEX_SPECIALS:
        pattern = rf"(?<!\\){re.escape(ch)}"
        text = re.sub(pattern, repl, text)

    # Optional: escape ~ and ^ only if not already escaped
    text = re.sub(r"(?<!\\)~", r"\\textasciitilde{}", text)
    text = re.sub(r"(?<!\\)\^", r"\\textasciicircum{}", text)

    # IMPORTANT: do NOT escape backslashes globally here,
    # or you'll break existing escapes and LaTeX commands.
    return text


def load_bert_topic_names(path: Path) -> Dict[int, str]:
    """
    Expects a BERTopic overview CSV with columns like: Topic, Name
    (your sample shows 'Topic' and 'Name').
    """
    df = pd.read_csv(path)
    if "Topic" not in df.columns:
        raise ValueError(f"[bert] Missing 'Topic' column in {path}")
    if "Name" not in df.columns:
        raise ValueError(f"[bert] Missing 'Name' column in {path}")

    out: Dict[int, str] = {}
    for _, row in df.iterrows():
        if pd.isna(row["Topic"]) or pd.isna(row["Name"]):
            continue
        try:
            tid = int(row["Topic"])
        except Exception:
            continue
        out[tid] = latex_escape(str(row["Name"]))
    return out


# ==========================================================
# SUMMARY (3-GROUP)
# ==========================================================


def build_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per category with:
      count_before, count_after_human, count_after_agent
      share_before, share_after_human, share_after_agent
    """
    grouped = (
        df.groupby(["group", "category"])
        .size()
        .unstack(fill_value=0)  # columns=category
    )

    # Ensure all three groups exist
    for g in ["before", "after_human", "after_agent"]:
        if g not in grouped.index:
            grouped.loc[g] = 0

    grouped = grouped.sort_index()

    # categories as rows
    cat_table = grouped.T.copy()
    cat_table["count_before"] = cat_table.get("before", 0)
    cat_table["count_after_human"] = cat_table.get("after_human", 0)
    cat_table["count_after_agent"] = cat_table.get("after_agent", 0)

    total_before = int(cat_table["count_before"].sum())
    total_human = int(cat_table["count_after_human"].sum())
    total_agent = int(cat_table["count_after_agent"].sum())

    # safe shares (avoid division by zero)
    cat_table["share_before"] = (
        100 * cat_table["count_before"] / total_before if total_before > 0 else 0.0
    )
    cat_table["share_after_human"] = (
        100 * cat_table["count_after_human"] / total_human if total_human > 0 else 0.0
    )
    cat_table["share_after_agent"] = (
        100 * cat_table["count_after_agent"] / total_agent if total_agent > 0 else 0.0
    )

    out = (
        cat_table.reset_index()
        .rename(columns={"index": "category"})
        .sort_values("category")
        .reset_index(drop=True)
    )
    return out


# ==========================================================
# CHI-SQUARE STATS
# ==========================================================


def _cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    if n <= 0:
        return float("nan")
    denom = n * (min(r, c) - 1)
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(chi2 / denom))


def compute_chi2_3group(summary: pd.DataFrame, label: str) -> dict:
    """
    3-group chi-square over categories x groups (3 columns).
    """
    table = summary[["count_before", "count_after_human", "count_after_agent"]].values
    chi2, p, _, _ = chi2_contingency(table)

    n = int(table.sum())
    v = _cramers_v(chi2, n, r=table.shape[0], c=table.shape[1])

    return {
        "dataset": label,
        "comparison": "before vs after_human vs after_agent",
        "chi2": float(chi2),
        "p_value": float(p),
        "cramers_v": float(v),
        "n": n,
    }


def compute_pairwise_chi2(summary: pd.DataFrame, label: str) -> List[dict]:
    """
    Pairwise chi-square across the same categories for:
      before vs after_human
      before vs after_agent
      after_human vs after_agent
    """
    pairs = [
        ("before", "after_human"),
        ("before", "after_agent"),
        ("after_human", "after_agent"),
    ]

    results: List[dict] = []

    colmap = {
        "before": "count_before",
        "after_human": "count_after_human",
        "after_agent": "count_after_agent",
    }

    for g1, g2 in pairs:
        c1, c2 = colmap[g1], colmap[g2]
        table = summary[[c1, c2]].values
        chi2, p, _, _ = chi2_contingency(table)

        n = int(table.sum())
        v = _cramers_v(chi2, n, r=table.shape[0], c=table.shape[1])

        results.append(
            {
                "dataset": label,
                "comparison": f"{g1} vs {g2}",
                "chi2": float(chi2),
                "p_value": float(p),
                "cramers_v": float(v),
                "n": n,
            }
        )

    return results


# ==========================================================
# PROVENANCE TABLE (TOPICS -> META-CATEGORY)
# ==========================================================


def build_topic_provenance(
    df: pd.DataFrame,
    topic_to_category: Dict[int, str],
    bert_labels: Dict[int, str],
    min_docs: int = 100,
) -> pd.DataFrame:
    """
    Outputs:
      bert_topic_label, documents, category
    Filter: topic_id != -1, docs >= min_docs, has category.
    """
    df = df[df["topic_id"] != -1].copy()

    counts = df.groupby("topic_id").size().reset_index(name="documents")
    counts["bert_topic_label"] = counts["topic_id"].map(bert_labels)
    counts["category"] = counts["topic_id"].map(topic_to_category)

    # drop unmapped topics/categories
    counts = counts.dropna(subset=["category"]).copy()

    # keep label even if missing; fill with topic id
    counts["bert_topic_label"] = counts["bert_topic_label"].fillna(
        counts["topic_id"].astype(str)
    )

    counts = counts[counts["documents"] >= int(min_docs)].copy()

    return (
        counts.sort_values("documents", ascending=False)
        .reset_index(drop=True)[["bert_topic_label", "documents", "category"]]
    )


# ==========================================================
# LATEX OUTPUT
# ==========================================================


def write_three_group_distribution_table(
    summary: pd.DataFrame,
    caption: str,
    label: str,
    out: Path,
):

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Category} & "
        r"\textbf{Share(BA)} & "
        r"\textbf{Share(AH)} & "
        r"\textbf{Share(AA)} & "
        r"\textbf{$\Delta$(B→H)} & "
        r"\textbf{$\Delta$(B→A)} \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():

        delta_bh = row["share_after_human"] - row["share_before"]
        delta_ba = row["share_after_agent"] - row["share_before"]

        lines.append(
            f"{row['category']} & "
            f"{row['share_before']:.2f}\\% & "
            f"{row['share_after_human']:.2f}\\% & "
            f"{row['share_after_agent']:.2f}\\% & "
            f"{delta_bh:+.2f}\\% & "
            f"{delta_ba:+.2f}\\% \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        f"\\label{{{label}}}",
        r"\end{table}",
    ]

    out.write_text("\n".join(lines), encoding="utf8")


# ==========================================================
# PLOTS (3 GROUPS)
# ==========================================================


def plot_grouped_three(summary: pd.DataFrame, out: Path, title: str):
    """
    Grouped bar chart: share_before, share_after_human, share_after_agent
    """
    x = np.arange(len(summary))
    w = 0.25

    plt.figure(figsize=(14, 6))

    plt.bar(x - w, summary["share_before"], w, label="Before")
    plt.bar(x, summary["share_after_human"], w, label="After (Human)")
    plt.bar(x + w, summary["share_after_agent"], w, label="After (Agent)")

    plt.xticks(x, summary["category"], rotation=45, ha="right")
    plt.ylabel("Share of documents")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


# ==========================================================
# PER TEXT TYPE PIPELINE
# ==========================================================


def process_text_type(
    name: str,
    doc_csv: Path,
    map_csv: Path,
    bert_csv: Path,
    min_topic_docs: int = 100,
) -> pd.DataFrame:
    """
    Builds:
      - CSV summary (3 groups)
      - LaTeX distribution table (3 groups)
      - LaTeX provenance table
      - plot
    Returns summary for stats.
    """
    topic_to_category = load_topic_mapping(map_csv)
    bert_labels = load_bert_topic_names(bert_csv)

    df = load_and_map_docs(doc_csv, map_csv)
    summary = build_category_summary(df)

    summary.to_csv(TABLES / f"rq5_manual_topic_summary_{name}.csv", index=False)

    write_three_group_distribution_table(
        summary=summary,
        caption=(
            f"Category share distribution before agent introduction, "
            f"after human introduction, and after agent introduction for "
            f"{name.replace('_', ' ')}."
        ),
        label=f"tab:{name}_combined_distribution",
        out=LATEX / f"{name}_combined_distribution.tex",
    )

    return summary


# ==========================================================
# MAIN RUNNER
# ==========================================================


def run():
    configs = {
        "commit_messages": {
            "docs": TABLES / "rq5_doc_topics_commit_messages.csv",
            "map": ROOT / "manual_commit_message_topic_mapping.csv",
            "bert": TABLES / "rq5_topics_overview_commit_messages.csv",
        },
        "pr_bodies": {
            "docs": TABLES / "rq5_doc_topics_pr_bodies.csv",
            "map": ROOT / "manual_pr_body_topic_mapping.csv",
            "bert": TABLES / "rq5_topics_overview_pr_bodies.csv",
        },
    }

    stats_rows: List[dict] = []

    for name, cfg in configs.items():
        summary = process_text_type(
            name=name,
            doc_csv=cfg["docs"],
            map_csv=cfg["map"],
            bert_csv=cfg["bert"],
            min_topic_docs=100,
        )

        # 3-group + pairwise stats
        if summary is not None and not summary.empty:
            stats_rows.append(compute_chi2_3group(summary, name))
            stats_rows.extend(compute_pairwise_chi2(summary, name))

    pd.DataFrame(stats_rows).to_csv(STATS / "rq5_manual_topic_stats.csv", index=False)
    print("[OK] RQ5 manual topic analysis complete.")


if __name__ == "__main__":
    run()