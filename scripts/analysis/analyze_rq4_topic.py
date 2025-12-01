# =============================================================
# RQ4 — Semi-Supervised Topic Modelling with Anchored CorEx
# =============================================================
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer

from data_loader import load_all


# -------------------------------------------------------------
# Output paths
# -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
OUT_TABLES = ROOT / "outputs" / "rq4" / "tables"
OUT_PLOTS = ROOT / "outputs" / "rq4" / "plots"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------
# Terms to remove (LLM-related / AI-related)
# -------------------------------------------------------------
AI_TERMS = [
    "ai", "agent", "model", "models", "llm", "gpt", "openai",
    "anthropic", "deepseek", "assistant", "automated", "auto",
    "llm-generated"
]


# -------------------------------------------------------------
# Text cleaner
# -------------------------------------------------------------
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""

    s = s.lower()

    # Remove AI terms
    for t in AI_TERMS:
        s = re.sub(rf"\b{t}\b", " ", s)

    # Remove URLs
    s = re.sub(r"http\S+|www\.\S+", " ", s)

    # Code blocks
    s = re.sub(r"`+[^`]*`+", " ", s)
    s = re.sub(r"```[\s\S]*?```", " ", s)

    # Hex strings
    s = re.sub(r"\b[0-9a-f]{7,40}\b", " ", s)

    # Paths
    s = re.sub(r"/[^ \t\n\r\f\v]+", " ", s)

    # Non-alphanumeric
    s = re.sub(r"[^a-z0-9\s]", " ", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -------------------------------------------------------------
# 8 Anchored Topic Categories
# -------------------------------------------------------------
ANCHORS_SOURCE = {
    0: ("Bug Fixing & Debugging", [
        "bug", "crash", "error", "exception", "fault", "regression",
        "stacktrace", "segfault", "nullpointer", "panic", "hang",
        "freeze", "timeout", "deadlock", "race", "memory leak",
        "overflow", "underflow", "corruption", "mismatch",
        "debug", "debugging", "diagnose", "investigate", "traceback"
    ]),
    1: ("Testing & CI", [
        "test", "tests", "testing", "unittest", "pytest", "jest", "mocha",
        "ci", "pipeline", "workflow", "verification", "validate",
        "integration test", "coverage", "mock", "fixture", "assert",
        "qa", "quality"
    ]),
    2: ("Refactoring & Cleanup", [
        "refactor", "cleanup", "reorganize", "restructure",
        "reformat", "formatting", "lint", "tidy", "rename", "extract",
        "optimize", "consolidate", "dead code", "remove unused",
        "modernize", "migrate", "tech debt", "quality improvement"
    ]),
    3: ("Feature Development", [
        "feature", "implement", "add", "create", "introduce", "support",
        "enable", "mvp", "prototype", "functionality", "endpoint", "api",
        "handler", "module", "component", "extend", "enhancement",
        "route", "controller", "schema", "field", "flag", "settings",
        "plugin"
    ]),
    4: ("UI / UX", [
        "ui", "ux", "interface", "layout", "design", "styling",
        "css", "scss", "html", "jsx", "tsx", "components",
        "button", "menu", "dropdown", "modal", "dialog",
        "theme", "color", "alignment", "responsive", "a11y",
        "frontend", "animation", "icon", "typography"
    ]),
    5: ("Documentation & Communication", [
        "doc", "docs", "documentation", "readme", "guide",
        "comment", "comments", "changelog", "tutorial", "examples",
        "clarify", "description", "spelling", "grammar",
        "markdown", "overview", "communication"
    ]),
    6: ("Build / Tooling / DevOps", [
        "build", "docker", "compose", "deployment", "deploy",
        "pipeline", "ci", "cd", "workflow", "action", "makefile",
        "config", "script", "dependency", "package", "version bump",
        "infrastructure", "env", "container", "kubernetes",
        "bash", "shell", "automation"
    ]),
    7: ("Code Review & Collaboration", [
        "review", "reviews", "cr", "approve", "approved", "lgtm",
        "commented", "feedback", "revise", "discussion", "thread",
        "collaboration", "merge", "conflict", "resolve conflict",
        "pull request", "pr", "code owner", "nit", "nitpick"
    ])
}

N_TOPICS = 8

# -------------------------------------------------------------
# RANDOMIZE anchors + labels while keeping pairs aligned
# -------------------------------------------------------------
anchor_pairs = [(ANCHORS_SOURCE[i][1], ANCHORS_SOURCE[i][0]) for i in range(N_TOPICS)]
random.shuffle(anchor_pairs)

ANCHOR_GROUPS = [p[0] for p in anchor_pairs]       # list[list[str]]
ANCHOR_LABELS_LIST = [p[1] for p in anchor_pairs]  # list[str]

def topic_label_from_id(tid: int) -> str:
    return ANCHOR_LABELS_LIST[tid]


# -------------------------------------------------------------
# CorEx Topic Modelling
# -------------------------------------------------------------
def corex_topics(df_before: pd.DataFrame,
                 df_after: pd.DataFrame,
                 text_col: str,
                 time_col: str,
                 text_kind: str):

    def prep(df):
        df = df[[text_col, time_col]].copy()
        df["clean"] = df[text_col].fillna("").astype(str).map(clean_text)
        df["time"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df[df["clean"].str.len() > 0]
        df = df[df["time"].notna()]
        return df.reset_index(drop=True)

    before = prep(df_before)
    after = prep(df_after)

    if before.empty or after.empty:
        print(f"[rq4] {text_kind}: one group empty — skip.")
        return None

    combined = pd.concat([before, after], ignore_index=True)

    # -----------------------------
    # Vectorize
    # -----------------------------
    vectorizer = CountVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=5,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(combined["clean"].tolist())
    vocab = vectorizer.get_feature_names_out()

    # -----------------------------
    # Build anchor indices
    # -----------------------------
    anchor_indices = []
    for wordlist in ANCHOR_GROUPS:
        indices = [np.where(vocab == w)[0][0] for w in wordlist if w in vocab]
        anchor_indices.append(indices)

    # -----------------------------
    # Fit CorEx
    # -----------------------------
    corex = ct.Corex(n_hidden=N_TOPICS, seed=42)
    corex.fit(
        X,
        words=vocab,
        anchors=anchor_indices,
        anchor_strength=3,
    )

    # Topic activations
    topic_matrix = corex.transform(X)
    topic_id = topic_matrix.argmax(axis=1)

    combined["topic_id"] = topic_id
    combined["topic_label"] = combined["topic_id"].map(topic_label_from_id)

    # Split back into before/after
    before_out = combined.iloc[:len(before)].copy()
    before_out["group"] = "before"

    after_out = combined.iloc[len(before):].copy()
    after_out["group"] = "after"

    # Summaries
    summary = []
    for tid in range(N_TOPICS):
        label = topic_label_from_id(tid)

        b_count = (before_out["topic_id"] == tid).sum()
        a_count = (after_out["topic_id"] == tid).sum()

        b_share = b_count / len(before_out)
        a_share = a_count / len(after_out)

        summary.append({
            "topic_id": tid,
            "topic_label": label,
            "count_before": b_count,
            "count_after": a_count,
            "share_before": b_share,
            "share_after": a_share,
            "delta_share": a_share - b_share,
        })

    summary_df = pd.DataFrame(summary)
    return before_out, after_out, summary_df


# -------------------------------------------------------------
# Plotting
# -------------------------------------------------------------
def plot_stacked(summary_df: pd.DataFrame, out_path: Path, title: str):
    summary_df = summary_df.sort_values("topic_id")

    labels = summary_df["topic_label"].tolist()
    before = summary_df["share_before"].tolist()
    after = summary_df["share_after"].tolist()

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, before, width, label="Before")
    plt.bar(x + width/2, after, width, label="After")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Share of documents")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def run():
    data = load_all()

    TARGETS = {
        "commit_messages": ("commit_messages_before", "commit_messages_after"),
        "pr_bodies": ("pr_bodies_before", "pr_bodies_after"),
        "issue_bodies": ("issue_bodies_before", "issue_bodies_after"),
        "review_comments": ("review_comments_before", "review_comments_after"),
    }

    for name, (bkey, akey) in TARGETS.items():
        print(f"[rq4] Processing {name}...")

        before_df = data.get(bkey)
        after_df = data.get(akey)

        if before_df is None or after_df is None:
            print(f"[rq4] Missing data for {name}")
            continue

        result = corex_topics(before_df, after_df,
                              text_col="text",
                              time_col="date",
                              text_kind=name)

        if result is None:
            continue

        before_out, after_out, summary_df = result

        # Save documents
        pd.concat([before_out, after_out], ignore_index=True).to_csv(
            OUT_TABLES / f"rq4_doc_topics_{name}.csv", index=False
        )

        # Save summary
        summary_df.to_csv(
            OUT_TABLES / f"rq4_topic_summary_{name}.csv", index=False
        )

        # Plot
        plot_stacked(
            summary_df,
            OUT_PLOTS / f"rq4_stacked_{name}.png",
            f"{name}: topic distribution before/after"
        )

    print("[rq4] Done.")


if __name__ == "__main__":
    run()
