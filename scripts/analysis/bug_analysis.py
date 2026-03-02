import pandas as pd
from pathlib import Path
import sys
import numpy as np

# Ensure local imports work
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from data_loader import load_all

# Configuration
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
    """Returns (ratio_string, count, total)"""
    if df is None or df.empty or "text" not in df.columns:
        return "0.00% (0/0)", 0, 0
    
    pattern = r"\b(?:" + "|".join(keywords) + r")\b"
    is_match = df["text"].str.contains(pattern, case=False, na=False)
    count = is_match.sum()
    total = len(df)
    ratio = (count / total * 100) if total > 0 else 0.0
    return f"{ratio:.2f}% ({count:,}/{total:,})", count, total

def main():
    data = load_all()
    
    # Organize data by period
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
        
        row_data = {"commits": [], "prs": [], "combined": []}
        
        for p_label, datasets in periods.items():
            c_df = datasets["commits"]
            p_df = datasets["prs"]
            comb_df = pd.concat([c_df, p_df], ignore_index=True)
            
            c_str, _, _ = get_stats(c_df, keywords)
            p_str, _, _ = get_stats(p_df, keywords)
            comb_str, _, _ = get_stats(comb_df, keywords)
            
            row_data["commits"].append(c_str)
            row_data["prs"].append(p_str)
            row_data["combined"].append(comb_str)

        print(f"  Commits only    | {row_data['commits'][0]:<25} | {row_data['commits'][1]:<25} | {row_data['commits'][2]:<25}")
        print(f"  PR bodies only  | {row_data['prs'][0]:<25} | {row_data['prs'][1]:<25} | {row_data['prs'][2]:<25}")
        print(f"  Combined        | {row_data['combined'][0]:<25} | {row_data['combined'][1]:<25} | {row_data['combined'][2]:<25}")
        print("-" * 105)

if __name__ == "__main__":
    main()