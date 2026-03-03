from __future__ import annotations

"""
FAST BERTopic runner for 3-group comparison:
    - before
    - after_human
    - after_agent

Optimized for:
    - Large corpora
    - Empirical software engineering research
    - Faster clustering
    - Stable topic structures

Removes:
    - Probability calculation (very slow)
    - Sentence-level segmentation (unnecessary)
"""

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from data_loader import load_all


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
OUT_TABLES = ROOT / "outputs" / "rq5" / "tables"
OUT_TABLES.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Text Cleaning
# ------------------------------------------------------------------

AI_TERMS = [
    "ai", "agent", "agents", "model", "models", "llm", "gpt",
    "openai", "anthropic", "assistant", "automated", "auto"
]

def clean_text(s: str) -> str:
    """
    Clean GitHub artifact text.

    Removes:
        - URLs
        - Code blocks
        - Commit hashes
        - AI-related terms (to avoid trivial clustering)
    """
    if not isinstance(s, str):
        return ""

    s = s.lower()

    for t in AI_TERMS:
        s = re.sub(rf"\b{t}\b", " ", s)

    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"`+[^`]*`+", " ", s)
    s = re.sub(r"```[\s\S]*?```", " ", s)
    s = re.sub(r"\b[0-9a-f]{7,40}\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


# ------------------------------------------------------------------
# Optimized Topic Model Builder
# ------------------------------------------------------------------

def build_fast_topic_model() -> BERTopic:
    """
    Returns a faster BERTopic model suitable for
    large-scale empirical analysis.
    """

    # Lower-dimensional manifold learning (faster clustering)
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,   # reduce to 5D for faster HDBSCAN
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    # Larger clusters = fewer micro-fragments
    hdbscan_model = HDBSCAN(
        min_cluster_size=50,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False
    )

    return BERTopic(
        embedding_model=None,  # we supply embeddings manually
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,  # HUGE speed boost
        verbose=True
    )


# ------------------------------------------------------------------
# Core 3-Group Topic Runner
# ------------------------------------------------------------------

def run_bertopic_three_groups(
    df_before: pd.DataFrame,
    df_after_human: pd.DataFrame,
    df_after_agent: pd.DataFrame,
    text_col: str,
    date_col: str
) -> Tuple[pd.DataFrame, BERTopic]:

    def prep(df, label):
        if df is None or df.empty:
            return pd.DataFrame(columns=[text_col, date_col, "clean", "group"])

        df = df[[text_col, date_col]].copy()
        df["clean"] = df[text_col].fillna("").astype(str).map(clean_text)
        df["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        df["group"] = label

        df = df[df["clean"].str.len() > 5]
        df = df[df["date"].notna()]

        return df.reset_index(drop=True)

    before = prep(df_before, "before")
    human  = prep(df_after_human, "after_human")
    agent  = prep(df_after_agent, "after_agent")

    combined = pd.concat([before, human, agent], ignore_index=True)

    if combined.empty:
        return combined, None

    print("Total documents:", len(combined))

    # ------------------------------------------------------------------
    # Embed ONCE (no segmentation — much faster)
    # ------------------------------------------------------------------

    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    embeddings = embedder.encode(
        combined["clean"].tolist(),
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # ------------------------------------------------------------------
    # Fit Topic Model
    # ------------------------------------------------------------------

    topic_model = build_fast_topic_model()

    topics, _ = topic_model.fit_transform(
        combined["clean"].tolist(),
        embeddings
    )

    combined["topic_id"] = topics

    return combined, topic_model


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run():

    data = load_all()

    TARGETS = {
        "commit_messages": (
            "commit_messages_before",
            "commit_messages_after_human",
            "commit_messages_after_agent",
        ),
        "pr_bodies": (
            "pr_bodies_before",
            "pr_bodies_after_human",
            "pr_bodies_after_agent",
        ),
        "issue_bodies": (
            "issue_bodies_before",
            "issue_bodies_after_human",
            "issue_bodies_after_agent",
        ),
        "review_comments": (
            "review_comments_before",
            "review_comments_after_human",
            "review_comments_after_agent",
        ),
    }

    for name, (bkey, hkey, akey) in TARGETS.items():

        print(f"\n[rq5-fast] Processing {name}")

        df_b = data.get(bkey, pd.DataFrame())
        df_h = data.get(hkey, pd.DataFrame())
        df_a = data.get(akey, pd.DataFrame())

        if df_b.empty and df_h.empty and df_a.empty:
            print("No data — skipping.")
            continue

        doc_df, tm = run_bertopic_three_groups(
            df_b,
            df_h,
            df_a,
            text_col="text",
            date_col="date"
        )

        if doc_df.empty:
            print("Empty after cleaning — skipping.")
            continue

        # Save document-topic assignments
        doc_df.to_csv(
            OUT_TABLES / f"rq5_doc_topics_{name}.csv",
            index=False
        )

        # Save topic overview
        topic_info = tm.get_topic_info()
        topic_info.to_csv(
            OUT_TABLES / f"rq5_topics_overview_{name}.csv",
            index=False
        )

        # Save per-group topic distribution
        dist = (
            doc_df
            .groupby(["group", "topic_id"])
            .size()
            .reset_index(name="count")
        )

        dist.to_csv(
            OUT_TABLES / f"rq5_topic_distribution_{name}.csv",
            index=False
        )

        print(f"[rq5-fast] Finished {name}")

    print("\n[rq5-fast] All done.")


if __name__ == "__main__":
    run()