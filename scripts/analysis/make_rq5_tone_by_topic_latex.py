#THIS FILE IS NO LONGER APPLICABLE. WAS USED FOR MEANS PRIOR TO BERTOPIC TOPIC MODELLING
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TONE_TABLES = ROOT / "outputs" / "rq5" / "tables" / "tone_by_topic"

OUT_LATEX = ROOT / "outputs" / "rq5" / "tone_three_tables.tex"

SENTIMENTS = ["pos", "neu", "neg"]

def pct(x):
    return f"{x*100:.2f}\\%"

def build():
    csvs = list(TONE_TABLES.glob("rq5_tone_by_topic_*.csv"))
    if not csvs:
        raise RuntimeError("No tone_by_topic CSVs found.")

    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)

    agg = (
        df.groupby(["topic_label", "group", "sentiment_cat"])["count"]
        .sum()
        .reset_index()
    )

    agg["share"] = agg.groupby(["topic_label", "group"])["count"].transform(
        lambda x: x / x.sum()
    )

    topics = sorted(agg["topic_label"].unique())

    def make_table(sent):
        header = """
\\begin{table*}[t]
\\centering
\\small
\\begin{tabular}{lrrrrr}
\\toprule
\\textbf{Topic} &
\\textbf{%(sent)s (B)} &
\\textbf{%(sent)s (A)} &
\\textbf{%(sent)s \\%% (B)} &
\\textbf{%(sent)s \\%% (A)} &
\\textbf{$\\Delta$ \\%%} \\\\
\\midrule
""" % {"sent": sent.title()}

        rows = []

        for topic in topics:
            #Before
            b = agg[
                (agg.topic_label == topic)
                & (agg.group == "before")
                & (agg.sentiment_cat == sent)
            ]
            #After
            a = agg[
                (agg.topic_label == topic)
                & (agg.group == "after")
                & (agg.sentiment_cat == sent)
            ]

            b_count = int(b["count"].sum()) if not b.empty else 0
            a_count = int(a["count"].sum()) if not a.empty else 0

            b_share = float(b["share"].sum()) if not b.empty else 0.0
            a_share = float(a["share"].sum()) if not a.empty else 0.0

            delta = a_share - b_share

            row = f"{topic} & {b_count} & {a_count} & {pct(b_share)} & {pct(a_share)} & {pct(delta)} \\\\"
            rows.append(row)

        footer = """
\\bottomrule
\\end{tabular}
\\caption{%s sentiment distribution across topics (combined datasets).}
\\label{tab:rq5_tone_%s}
\\end{table*}
""" % (sent.title(), sent)

        return header + "\n".join(rows) + footer

    #All 3 tables together for now
    final_text = (
        make_table("positive")
        + "\n\n"
        + make_table("neutral")
        + "\n\n"
        + make_table("negative")
    )

    OUT_LATEX.write_text(final_text, encoding="utf8")
    print(f"[OK] Wrote → {OUT_LATEX}")


if __name__ == "__main__":
    build()
