from __future__ import annotations
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Correct directory for tone-by-texttype CSVs
TONE_DIR = ROOT / "outputs" / "rq5" / "tables"

OUT_POS = ROOT / "outputs" / "rq5" / "tone_positive_table.tex"
OUT_NEU = ROOT / "outputs" / "rq5" / "tone_neutral_table.tex"
OUT_NEG = ROOT / "outputs" / "rq5" / "tone_negative_table.tex"

FILES = {
    "Commit Messages": "rq5_tone_commit_messages.csv",
    "Pull Request Bodies": "rq5_tone_pr_bodies.csv",
    "Issues": "rq5_tone_issue_bodies.csv",
    "Review Comments": "rq5_tone_review_comments.csv",
}

SENTIMENTS = ["positive", "neutral", "negative"]


def esc(s: str) -> str:
    return s.replace("&", r"\&")


def pct(v, total):
    return (v / total * 100) if total else 0.0


def load_stats():
    """Compute sentiment counts, shares, and deltas for each dataset + combined."""
    results = {}

    for ds_name, fname in FILES.items():
        fp = TONE_DIR / fname
        if not fp.exists():
            print(f"[WARN] Missing {fp}")
            continue

        df = pd.read_csv(fp)
        stats = {}

        for s in SENTIMENTS:
            b = len(df[(df.group == "before") & (df.sentiment_cat == s)])
            a = len(df[(df.group == "after") & (df.sentiment_cat == s)])

            total_b = len(df[df.group == "before"])
            total_a = len(df[df.group == "after"])

            pct_b = pct(b, total_b)
            pct_a = pct(a, total_a)

            stats[s] = {
                "b": b, "a": a,
                "pct_b": pct_b,
                "pct_a": pct_a,
                "delta": pct_a - pct_b,
            }

        results[ds_name] = stats

    # ------------------ COMBINED ------------------
    combined = {s: {"b": 0, "a": 0} for s in SENTIMENTS}

    for ds_stats in results.values():
        for s in SENTIMENTS:
            combined[s]["b"] += ds_stats[s]["b"]
            combined[s]["a"] += ds_stats[s]["a"]

    total_b = sum(combined[s]["b"] for s in SENTIMENTS)
    total_a = sum(combined[s]["a"] for s in SENTIMENTS)

    for s in SENTIMENTS:
        b = combined[s]["b"]
        a = combined[s]["a"]
        pct_b = pct(b, total_b)
        pct_a = pct(a, total_a)

        combined[s].update({
            "pct_b": pct_b,
            "pct_a": pct_a,
            "delta": pct_a - pct_b
        })

    results["Combined"] = combined
    return results


# ===============================================
# LATEX GENERATION
# ===============================================

def make_table(stats, sentiment, outfile):
    label = {
        "positive": "Pos",
        "neutral": "Neu",
        "negative": "Neg"
    }[sentiment]

    title = {
        "positive": "Positive sentiment distribution across text types",
        "neutral": "Neutral sentiment distribution across text types",
        "negative": "Negative sentiment distribution across text types",
    }[sentiment]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        rf"\textbf{{Dataset}} & \textbf{{{label}(B)}} & \textbf{{{label}(A)}} & "
        rf"\textbf{{{label} \% (B)}} & \textbf{{{label} \% (A)}} & \textbf{{$\Delta$ \%}} \\"
    )
    lines.append(r"\midrule")

    for ds, st in stats.items():
        row = st[sentiment]
        lines.append(
            f"{esc(ds)} & "
            f"{row['b']} & {row['a']} & "
            f"{row['pct_b']:.2f}\\% & {row['pct_a']:.2f}\\% & "
            f"{row['delta']:+.2f}\\% \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{title}.}}")
    lines.append(r"\end{table}")

    outfile.write_text("\n".join(lines), encoding="utf8")
    print(f"[OK] wrote {outfile}")


def build():
    stats = load_stats()
    make_table(stats, "positive", OUT_POS)
    make_table(stats, "neutral", OUT_NEU)
    make_table(stats, "negative", OUT_NEG)


if __name__ == "__main__":
    build()
