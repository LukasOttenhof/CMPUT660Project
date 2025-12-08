from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from data_loader import load_all


ROOT = Path(__file__).resolve().parents[2]
OUT_TABLES = ROOT / "outputs" / "rq5" / "tables"
OUT_PLOTS  = ROOT / "outputs" / "rq5" / "plots"

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)


#Clean text and filter some LLM AI terms
AI_TERMS = [
    "ai", "agent", "agents", "model", "models", "llm", "gpt", "openai",
    "anthropic", "assistant", "automated", "auto"
]

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()

    for t in AI_TERMS:
        s = re.sub(rf"\b{t}\b", " ", s)

    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"`+[^`]*`+", " ", s)
    s = re.sub(r"```[\s\S]*?```", " ", s)
    s = re.sub(r"\b[0-9a-f]{7,40}\b", " ", s)
    s = re.sub(r"/[^ \t\n\r\f\v]+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_into_segments(text: str):
    parts = re.split(r"[.!?;\n]+", text)
    segs = [p.strip() for p in parts if p.strip()]
    return segs if segs else [text.strip()]


#Embedding model
EMBED_MODEL = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

topic_model = BERTopic(
    embedding_model=EMBED_MODEL,
    calculate_probabilities=True,
    verbose=True
)


#Classification
def run_bertopic(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    text_col: str,
    date_col: str
) -> Tuple[pd.DataFrame, BERTopic]:

    def prep(df, group):
        df = df[[text_col, date_col]].copy()
        df["clean"] = df[text_col].fillna("").astype(str).map(clean_text)
        df["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        df["group"] = group
        df = df[df["clean"].str.len() > 0]
        df = df[df["date"].notna()]
        return df.reset_index(drop=True)

    before = prep(df_before, "before")
    after  = prep(df_after, "after")

    combined = pd.concat([before, after], ignore_index=True)

    #Text segmenting
    all_segments = []
    spans = []

    for text in combined["clean"]:
        segs = split_into_segments(text)
        start = len(all_segments)
        all_segments.extend(segs)
        spans.append((start, start + len(segs)))

    seg_embeddings = EMBED_MODEL.encode(
        all_segments,
        batch_size=64,
        show_progress_bar=True
    )

    doc_embeddings = []
    for start, end in spans:
        doc_embeddings.append(seg_embeddings[start:end].mean(axis=0))

    doc_embeddings = np.vstack(doc_embeddings)

    topics, probs = topic_model.fit_transform(
        combined["clean"].tolist(),
        embeddings=doc_embeddings
    )

    combined["topic_id"] = topics

    if probs is not None:
        combined["topic_max_prob"] = probs.max(axis=1)

    return combined, topic_model





def run():
    data = load_all()

    TARGETS = {
        "commit_messages": ("commit_messages_before", "commit_messages_after"),
        "pr_bodies": ("pr_bodies_before", "pr_bodies_after"),
        "issue_bodies": ("issue_bodies_before", "issue_bodies_after"),
        "review_comments": ("review_comments_before", "review_comments_after"),
    }

    topic_models: Dict[str, BERTopic] = {}

    for name, (bkey, akey) in TARGETS.items():
        print(f"[rq5-bertopic] Processing {name}...")

        df_b = data.get(bkey)
        df_a = data.get(akey)

        if df_b is None or df_a is None:
            print(f"[rq5-bertopic] Missing data for {name}, skipping.")
            continue

        doc_df, tm = run_bertopic(
            df_b,
            df_a,
            text_col="text",
            date_col="date"
        )

        topic_models[name] = tm

        out_docs = OUT_TABLES / f"rq5_doc_topics_{name}.csv"
        doc_df.to_csv(out_docs, index=False)
        print(f"[rq5-bertopic] Saved → {out_docs}")

        topic_info = tm.get_topic_info()
        topic_info.to_csv(
            OUT_TABLES / f"rq5_topics_overview_{name}.csv",
            index=False
        )

        rows = []
        for tid in topic_info["Topic"].unique():
            if tid == -1:
                continue
            words = [w for w, _ in tm.get_topic(tid)]
            rows.append({
                "topic_id": tid,
                "top_words": ", ".join(words[:10])
            })

        pd.DataFrame(rows).to_csv(
            OUT_TABLES / f"rq5_topic_keywords_{name}.csv",
            index=False
        )

        print(f"[rq5-bertopic] Exported topic keywords for {name}")

    print("[rq5-bertopic] Done.")


if __name__ == "__main__":
    run()
