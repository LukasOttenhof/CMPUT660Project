from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

TABLES = ROOT / "outputs" / "rq5" / "tables"
PLOTS  = ROOT / "outputs" / "rq5" / "plots"
LATEX  = ROOT / "outputs" / "rq4" / "latex"
STATS  = ROOT / "outputs" / "rq_stats"
BERT_OVERVIEW = TABLES / "rq5_topics_overview_commit_messages.csv"
BERT_OVERVIEW_PRs = TABLES / "rq5_topics_overview_pr_bodies.csv"

PLOTS.mkdir(parents=True, exist_ok=True)
LATEX.mkdir(parents=True, exist_ok=True)
STATS.mkdir(parents=True, exist_ok=True)

#All colour palette
PALETTE = ["#E74C3C", "#3498DB"]


def load_topic_mapping(path: Path) -> dict[int, str]:
    df = pd.read_csv(path)
    return dict(zip(df["topic_id"], df["category"]))


def load_and_map_docs(doc_csv: Path, mapping_csv: Path) -> pd.DataFrame:
    docs = pd.read_csv(doc_csv)
    mapping = load_topic_mapping(mapping_csv)

    docs["category"] = docs["topic_id"].map(mapping)
    docs = docs.dropna(subset=["category"])
    return docs

def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&":  r"\&",
        "%":  r"\%",
        "$":  r"\$",
        "#":  r"\#",
        "_":  r"\_",
        "{":  r"\{",
        "}":  r"\}",
        "~":  r"\textasciitilde{}",
        "^":  r"\textasciicircum{}",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def load_bert_topic_names(path: Path) -> dict[int, str]:
    df = pd.read_csv(path)

    return {
        int(row["Topic"]): latex_escape(str(row["Name"]))
        for _, row in df.iterrows()
        if pd.notna(row["Name"])
    }


def build_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["group", "category"])
          .size()
          .unstack(fill_value=0)
          .T
    )

    grouped["count_before"] = grouped.get("before", 0)
    grouped["count_after"]  = grouped.get("after", 0)
    grouped["delta_count"]  = grouped["count_after"] - grouped["count_before"]

    total_before = grouped["count_before"].sum()
    total_after  = grouped["count_after"].sum()

    grouped["share_before"] = 100 * grouped["count_before"] / total_before
    grouped["share_after"]  = 100 * grouped["count_after"] / total_after
    grouped["delta_share"]  = grouped["share_after"] - grouped["share_before"]

    return (
        grouped
        .reset_index()
        .rename(columns={"index": "category"})
        .sort_values("category")
    )


def compute_chi2(summary: pd.DataFrame, label: str) -> dict:
    table = summary[["count_before", "count_after"]].values
    chi2, p, _, _ = chi2_contingency(table)

    n = table.sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))

    return {
        "dataset": label,
        "chi2": chi2,
        "p_value": p,
        "cramers_v": v,
        "n": n
    }


def build_topic_provenance(
    df: pd.DataFrame,
    topic_to_category: dict[int, str],
    bert_labels: dict[int, str],
    min_docs: int = 100
) -> pd.DataFrame:

    df = df[df.topic_id != -1]

    counts = (
        df.groupby("topic_id")
          .size()
          .reset_index(name="documents")
    )

    counts["bert_topic_label"] = counts["topic_id"].map(bert_labels)
    counts["category"] = counts["topic_id"].map(topic_to_category)

    return (
        counts
        .dropna(subset=["category"])
        .query("documents >= @min_docs")
        .sort_values("documents", ascending=False)
        .reset_index(drop=True)
    )


#Latex
def write_latex_table(df, cols, caption, label, out):
    percent_cols = {
        "share_before",
        "share_after",
        "delta_share",
        "delta_share_pct"
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + "l" * len(cols) + "}",
        r"\toprule",
        " & ".join(c.replace("_", " ").title() for c in cols) + r" \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]

            if isinstance(v, float):
                if c in percent_cols:
                    vals.append(f"{v:.2f}\\%")
                else:
                    vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))

        lines.append(" & ".join(vals) + r" \\")
    
    lines += [
        r"\bottomrule",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out.write_text("\n".join(lines), encoding="utf8")

def write_combined_distribution_table(
    summary: pd.DataFrame,
    caption: str,
    label: str,
    out: Path,
):

    cols = [
        "category",
        "count_before",
        "count_after",
        "share_before",
        "share_after",
        "delta_share",
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        "Category & Count Before & Count After & Share Before & Share After & $\\Delta$ Share \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"{row['category']} & "
            f"{int(row['count_before'])} & "
            f"{int(row['count_after'])} & "
            f"{row['share_before']:.2f}\\% & "
            f"{row['share_after']:.2f}\\% & "
            f"{row['delta_share']:+.2f}\\% \\\\"
        )

    lines += [
        r"\bottomrule",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out.write_text("\n".join(lines), encoding="utf8")


#Plotting
def plot_grouped(summary, out, title):
    x = np.arange(len(summary))
    w = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(x - w/2, summary["share_before"], w,
            color=PALETTE[0], label="Before")
    plt.bar(x + w/2, summary["share_after"], w,
            color=PALETTE[1], label="After")

    plt.xticks(x, summary["category"], rotation=45, ha="right")
    plt.ylabel("Share of documents")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def plot_stacked_before_after(summary, out, title):
    categories = summary.sort_values("category")

    before = categories["share_before"].values
    after  = categories["share_after"].values

    x = [0, 1]
    bottom_before = 0
    bottom_after = 0

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab20").colors

    for i, cat in enumerate(categories["category"]):
        plt.bar(x[0], before[i], 0.6,
                bottom=bottom_before, color=cmap[i % len(cmap)], label=cat)
        plt.bar(x[1], after[i], 0.6,
                bottom=bottom_after, color=cmap[i % len(cmap)])

        bottom_before += before[i]
        bottom_after  += after[i]

    plt.xticks(x, ["Before", "After"])
    plt.ylabel("Share of documents")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


#Per text type
def process_text_type(name, doc_csv, map_csv, bert_csv):
    mapping = load_topic_mapping(map_csv)
    bert_labels = load_bert_topic_names(bert_csv)

    df = load_and_map_docs(doc_csv, map_csv)
    summary = build_category_summary(df)

    summary.to_csv(TABLES / f"rq5_manual_topic_summary_{name}.csv", index=False)

    write_latex_table(
        summary,
        ["category", "count_before", "count_after", "delta_count"],
        f"{name}: Category Counts",
        f"tab:{name}_counts",
        LATEX / f"{name}_counts.tex"
    )

    write_latex_table(
        summary,
        ["category", "share_before", "share_after", "delta_share"],
        f"{name}: Category Shares",
        f"tab:{name}_shares",
        LATEX / f"{name}_shares.tex"
    )
    write_combined_distribution_table(
        summary,
        f"{name}: Category Counts, Shares, and Share Change",
        f"tab:{name}_combined_distribution",
        LATEX / f"{name}_combined_distribution.tex",
    )

    provenance = build_topic_provenance(df, mapping, bert_labels)
    write_latex_table(
        provenance,
        ["topic_id", "bert_topic_label", "documents", "category"],
        f"{name}: Topic Provenance and Manual Category Assignment",
        f"tab:{name}_topics",
        LATEX / f"{name}_topics.tex"
    )

    plot_grouped(
        summary,
        PLOTS / f"rq5_manual_grouped_{name}.png",
        f"{name}: Category Distribution (Before vs After)"
    )

    plot_stacked_before_after(
        summary,
        PLOTS / f"rq4_{name}_stacked_before_after.png",
        f"{name.replace('_', ' ').title()}: Category Distribution"
    )

    return summary


def run():
    configs = {
        "commit_messages": {
            "docs": TABLES / "rq5_doc_topics_commit_messages.csv",
            "map":  ROOT / "manual_commit_message_topic_mapping.csv",
            "bert": TABLES / "rq5_topics_overview_commit_messages.csv",
        },
        "pr_bodies": {
            "docs": TABLES / "rq5_doc_topics_pr_bodies.csv",
            "map":  ROOT / "manual_pr_topic_mapping.csv",
            "bert": TABLES / "rq5_topics_overview_pr_bodies.csv",
        },
    }

    stats = []

    for name, cfg in configs.items():
        summary = process_text_type(
            name,
            cfg["docs"],
            cfg["map"],
            cfg["bert"],
        )
        stats.append(compute_chi2(summary, name))

    pd.DataFrame(stats).to_csv(
        STATS / "rq5_manual_topic_stats.csv", index=False
    )

    print("[OK] RQ4 manual topic analysis complete.")


if __name__ == "__main__":
    run()
