import pandas as pd
from pathlib import Path
import sys
import numpy as np
from scipy.stats import chi2_contingency


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from data_loader import load_all

CATEGORIES = {
    "Bug": [
        "bug", "bugs", "fix", "fixed", "fixes", "fixing", "resolves", "issue",
        "issues", "bugfix", "bugfixes", "closes", "hotfix", "hotfixes", "typo",
        "typos", "correct", "correction", "incorrect"
    ],
    "Refactor": [
        "refactor", "clean", "structure", "style", "rename", "move",
        "format", "lint", "tidy", "simplify", "optimize"
    ],
    "Revert": [
        "revert", "rollback", "undo"
    ]
}

def get_stats(df, keywords):
    if df is None or df.empty or "text" not in df.columns:
        return "0.00% (0/0)", 0, 0

    pattern = r"\b(?:" + "|".join(keywords) + r")\b"
    is_match = df["text"].str.contains(pattern, case=False, na=False)

    count = is_match.sum()
    total = len(df)
    ratio = (count / total * 100) if total > 0 else 0.0

    return f"{ratio:.2f}% ({count:,}/{total:,})", count, total


def chi_square_test(count1, total1, count2, total2):
    table = np.array([
        [count1, total1 - count1],
        [count2, total2 - count2]
    ])

    chi2, p, _, _ = chi2_contingency(table)

    n = table.sum()
    r, k = table.shape
    cramer_v = np.sqrt(chi2 / (n * (min(r - 1, k - 1))))

    return chi2, p, cramer_v


def main():
    data = load_all()

    periods = {
        "Before": {
            "commits": data["commit_messages_before"],
            "prs": data["pr_bodies_before"]
        },
        "After (Human)": {
            "commits": data["commit_messages_after_human"],
            "prs": data["pr_bodies_after_human"]
        },
        "After (Agent)": {
            "commits": data["commit_messages_after_agent"],
            "prs": data["pr_bodies_after_agent"]
        }
    }

    print(f"{'Text Source':<20} | {'Before':<25} | {'After (Human)':<25} | {'After (Agent)':<25}")
    print("-" * 105)

    for cat_name, keywords in CATEGORIES.items():
        print(f" {cat_name.upper()}")

        counts = {"commits": [], "prs": [], "combined": []}
        totals = {"commits": [], "prs": [], "combined": []}

        for p_label, datasets in periods.items():
            c_df = datasets["commits"]
            p_df = datasets["prs"]
            comb_df = pd.concat([c_df, p_df], ignore_index=True)

            c_str, c_count, c_total = get_stats(c_df, keywords)
            p_str, p_count, p_total = get_stats(p_df, keywords)
            comb_str, comb_count, comb_total = get_stats(comb_df, keywords)

            counts["commits"].append(c_count)
            counts["prs"].append(p_count)
            counts["combined"].append(comb_count)

            totals["commits"].append(c_total)
            totals["prs"].append(p_total)
            totals["combined"].append(comb_total)

            if p_label == "Before":
                before_strings = (c_str, p_str, comb_str)
            elif p_label == "After (Human)":
                human_strings = (c_str, p_str, comb_str)
            else:
                agent_strings = (c_str, p_str, comb_str)

        print(f"  Commits only    | {before_strings[0]:<25} | {human_strings[0]:<25} | {agent_strings[0]:<25}")
        print(f"  PR bodies only  | {before_strings[1]:<25} | {human_strings[1]:<25} | {agent_strings[1]:<25}")
        print(f"  Combined        | {before_strings[2]:<25} | {human_strings[2]:<25} | {agent_strings[2]:<25}")

        print("\n  Chi-square tests (Combined):")

        pairs = [
            ("Before vs Human", 0, 1),
            ("Before vs Agent", 0, 2),
            ("Human vs Agent", 1, 2)
        ]

        for label, i, j in pairs:
            chi2, p, v = chi_square_test(
                counts["combined"][i],
                totals["combined"][i],
                counts["combined"][j],
                totals["combined"][j]
            )

            print(f"   {label:<16} χ²={chi2:.2f}, p={p:.4f}, V={v:.3f}")

        print("-" * 105)


if __name__ == "__main__":
    main()