# =============================================================
# rq5 — BERT-Based Semi-Supervised Topic Classification
# =============================================================
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from data_loader import load_all


# -------------------------------------------------------------
# Output paths
# -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
OUT_TABLES = ROOT / "outputs" / "rq5" / "tables"
OUT_PLOTS = ROOT / "outputs" / "rq5" / "plots"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------
# Terms to remove (LLM-related)
# -------------------------------------------------------------
AI_TERMS = [
    "ai", "agent", "agents", "model", "models", "llm", "gpt", "openai",
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

    # URLs
    s = re.sub(r"http\S+|www\.\S+", " ", s)

    # Code spans
    s = re.sub(r"`+[^`]*`+", " ", s)
    s = re.sub(r"```[\s\S]*?```", " ", s)

    # Git hashes / long hex
    s = re.sub(r"\b[0-9a-f]{7,40}\b", " ", s)

    # File paths (rough)
    s = re.sub(r"/[^ \t\n\r\f\v]+", " ", s)

    # Non-alphanumeric
    s = re.sub(r"[^a-z0-9\s]", " ", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -------------------------------------------------------------
# Simple sentence/segment splitter for short dev texts
# -------------------------------------------------------------
def split_into_segments(text: str) -> List[str]:
    """
    Split commit messages / PR bodies into smaller clauses / sentences.
    This helps BERT by avoiding one giant, mixed sentence.
    """
    parts = re.split(r"[\.!?;\n]+", text)
    segments = [p.strip() for p in parts if p.strip()]
    if not segments:
        segments = [text.strip()]
    return segments


# -------------------------------------------------------------
# 8 Topic Categories — sentence-style anchors
# (natural English sentences => better sentence-transformer behaviour)
# -------------------------------------------------------------

FEATURE_DEV_ANCHORS = [
    "add new feature",
    "implement new api endpoint",
    "add backend route",
    "implement business logic",
    "add service layer",
    "introduce new module",
    "create controller method",
    "add functionality",
    "extend api",
    "add option to user settings",
    "add new config flag",
    "support new workflow",
    "implement handler for request",
    "add new serialization logic",
    "add database logic",
    "add query support",
    "add feature toggle",
    "extend model class",
    "add command handler",
    "feature implementation",
    "initial feature commit",
    "add new job worker",
    "add background task",
    "add permission checks",
    "add processing pipeline",
    "add input validation",
    "implement routing",
    "add pagination support",
    "add filtering options",
    "add new action in controller",
    "implement batch operation",
    "add state machine logic",
]

UI_UX_ANCHORS = [
    "update ui",
    "fix alignment issue",
    "update css",
    "adjust padding",
    "update layout",
    "restyle button",
    "improve page layout",
    "fix responsive styles",
    "modify styles",
    "update theme colors",
    "adjust margins",
    "refine ui look",
    "fix icon size",
    "revamp ui elements",
    "change font size",
    "css cleanup",
    "adjust grid layout",
    "fix dark mode styles",
    "fix mobile view",
    "add loading spinner",
    "ui tweak",
    "css fix",
    "update react component style",
    "adjust modal layout",
    "update header layout",
    "update navigation ui",
    "update form layout",
    "adjust spacing",
    "update color palette",
    "improve button styling",
    "update component props for ui",
]

BUG_FIX_ANCHORS = [
    "fix bug",
    "fix issue",
    "bugfix",
    "resolve crash",
    "fix exception",
    "handle null pointer",
    "fix npe",
    "fix edge case",
    "fix failing logic",
    "fix error handling",
    "fix regression",
    "fix incorrect output",
    "fix typo in code",
    "debug failing test",
    "fix missing dependency",
    "fix broken build",
    "fix crash on startup",
    "fix boundary case",
    "fix race condition",
    "fix deadlock",
    "fix off-by-one",
    "resolve merge issue",
    "fix infinite loop",
    "fix unexpected behavior",
    "fix state inconsistency",
    "fix validation error",
    "fix parsing bug",
    "fix broken url",
    "fix wrong variable",
    "fix broken reference",
    "fix serialization error",
    "fix threading bug",
]

TESTING_ANCHORS = [
    "add unit tests",
    "fix failing tests",
    "update tests",
    "improve test coverage",
    "add integration tests",
    "mock service in test",
    "ci config update",
    "update github actions",
    "fix workflow pipeline",
    "fix build pipeline",
    "disable flaky test",
    "update test fixtures",
    "improve test stability",
    "refactor tests",
    "add regression test",
    "fix ci timeout",
    "setup test environment",
    "update docker for ci",
    "fix test data",
    "add test case",
    "mock db in tests",
    "test cleanup",
    "ci optimizations",
    "update runner config",
]

REFACTOR_ANCHORS = [
    "refactor code",
    "cleanup code",
    "remove unused code",
    "rename variable",
    "rename file",
    "simplify logic",
    "extract method",
    "inline function",
    "code reorganization",
    "remove clutter",
    "remove dead code",
    "format code",
    "apply lint fixes",
    "rename package",
    "remove unused import",
    "split file",
    "merge duplicate logic",
    "optimize algorithm",
    "reduce complexity",
    "improve readability",
    "small refactor",
    "code style update",
    "update naming",
    "move file",
]

DOC_ANCHORS = [
    "update readme",
    "update documentation",
    "improve docs",
    "add usage instructions",
    "update comments",
    "add docstring",
    "clarify behavior",
    "update changelog",
    "update contributing guide",
    "fix documentation typo",
    "expand docs",
    "add example",
    "add inline docs",
    "update project description",
    "update setup guide",
]

DEPLOY_ANCHORS = [
    "update dockerfile",
    "update deployment script",
    "update build config",
    "upgrade dependency",
    "update requirements",
    "update lockfile",
    "modify helm chart",
    "modify terraform",
    "ci/cd pipeline update",
    "optimize docker image",
    "update environment vars",
    "modify k8s config",
    "version bump",
    "update makefile",
    "infra updates",
]

REVIEW_ANCHORS = [
    "address review comments",
    "review feedback changes",
    "resolve merge conflict",
    "approve changes",
    "apply pr feedback",
    "review fixes",
    "update per reviewer request",
    "merge branch",
    "pr updates",
    "collaboration changes",
    "team feedback changes",
]

ANCHOR_SENTENCES = {
    "Feature Development": FEATURE_DEV_ANCHORS,
    "UI / UX": UI_UX_ANCHORS,
    "Bug Fixing & Debugging": BUG_FIX_ANCHORS,
    "Testing & CI": TESTING_ANCHORS,
    "Refactoring & Cleanup": REFACTOR_ANCHORS,
    "Documentation & Communication": DOC_ANCHORS,
    "Deployment": DEPLOY_ANCHORS,
    "Code Review & Collaboration": REVIEW_ANCHORS,
}

CATEGORY_LIST: List[str] = list(ANCHOR_SENTENCES.keys())
N_TOPICS = len(CATEGORY_LIST)


# -------------------------------------------------------------
# BERT Embedding Model
# -------------------------------------------------------------
# Stronger model than MiniLM. Requires more memory but much better semantics.
BERT = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


# -------------------------------------------------------------
# Compute anchor embeddings (batched, averaged per category)
# -------------------------------------------------------------
def compute_anchor_embeddings() -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    all_sents: List[str] = []
    cat_index: List[str] = []

    for cat, sents in ANCHOR_SENTENCES.items():
        for s in sents:
            all_sents.append(s)
            cat_index.append(cat)

    Z = BERT.encode(all_sents, batch_size=64, show_progress_bar=False)
    Z = np.asarray(Z)

    # Aggregate by category
    cat_embeds: Dict[str, List[np.ndarray]] = {c: [] for c in CATEGORY_LIST}
    for emb, cat in zip(Z, cat_index):
        cat_embeds[cat].append(emb)

    anchor_embeds: Dict[str, np.ndarray] = {
        cat: np.mean(v, axis=0) for cat, v in cat_embeds.items()
    }

    anchor_matrix = np.stack([anchor_embeds[cat] for cat in CATEGORY_LIST], axis=0)
    return anchor_embeds, anchor_matrix


ANCHOR_EMBEDS, ANCHOR_MATRIX = compute_anchor_embeddings()


# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------
def slugify(label: str) -> str:
    s = label.lower()
    s = s.replace("&", "and")
    for ch in ["/", "-", " "]:
        s = s.replace(ch, "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


WEIGHT_COLS = {label: f"w_{slugify(label)}" for label in CATEGORY_LIST}


def softmax(x: np.ndarray, temp: float = 0.4) -> np.ndarray:
    """Temperature-scaled softmax for one 1D vector."""
    z = x / temp
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def embed_docs_with_segments(texts: List[str]) -> np.ndarray:
    """
    Split each text into smaller segments, embed each segment, and
    average per document.
    """
    all_segments: List[str] = []
    doc_spans: List[Tuple[int, int]] = []

    for t in texts:
        segs = split_into_segments(t)
        start = len(all_segments)
        all_segments.extend(segs)
        end = len(all_segments)
        doc_spans.append((start, end))

    seg_embs = BERT.encode(all_segments, batch_size=64, show_progress_bar=False)
    seg_embs = np.asarray(seg_embs)

    doc_embs = []
    for start, end in doc_spans:
        doc_embs.append(seg_embs[start:end].mean(axis=0))
    return np.vstack(doc_embs)


# -------------------------------------------------------------
# Core classification using soft mixture weights
# -------------------------------------------------------------
def classify_topics(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    text_col: str,
    time_col: str,
    text_kind: str,
):
    def prep(df: pd.DataFrame) -> pd.DataFrame:
        df = df[[text_col, time_col]].copy()
        df["clean"] = df[text_col].fillna("").astype(str).map(clean_text)
        df["time"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df[df["clean"].str.len() > 0]
        df = df[df["time"].notna()]
        return df.reset_index(drop=True)

    before = prep(df_before)
    after = prep(df_after)

    if before.empty or after.empty:
        print(f"[rq5] {text_kind}: one group empty after cleaning — skipping.")
        return None

    combined = pd.concat([before, after], ignore_index=True)
    texts = combined["clean"].tolist()

    # -------------------------------
    # BERT embeddings (sentence-split)
    # -------------------------------
    emb_matrix = embed_docs_with_segments(texts)  # (n_docs, d)

    # Normalize for cosine similarity
    doc_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    doc_norms = np.clip(doc_norms, 1e-9, None)
    emb_unit = emb_matrix / doc_norms

    anchor_norms = np.linalg.norm(ANCHOR_MATRIX, axis=1, keepdims=True)
    anchor_norms = np.clip(anchor_norms, 1e-9, None)
    anchor_unit = ANCHOR_MATRIX / anchor_norms

    # Cosine similarities: (n_docs, n_topics)
    sims = emb_unit @ anchor_unit.T

    # -------------------------------
    # Convert to soft weights per topic
    # -------------------------------
    weights = np.zeros_like(sims)
    for i in range(sims.shape[0]):
        weights[i] = softmax(sims[i], temp=0.4)

    # Hard assignment: topic with highest weight
    primary_idx = weights.argmax(axis=1)
    primary_labels = [CATEGORY_LIST[i] for i in primary_idx]
    combined["topic_label"] = primary_labels

    # Attach weight columns
    for j, label in enumerate(CATEGORY_LIST):
        col = WEIGHT_COLS[label]
        combined[col] = weights[:, j]

    # Split back into before / after
    before_out = combined.iloc[: len(before)].copy()
    before_out["group"] = "before"

    after_out = combined.iloc[len(before) :].copy()
    after_out["group"] = "after"

    # -------------------------------
    # Summary per topic
    # -------------------------------
    summary_rows = []
    for label in CATEGORY_LIST:
        w_col = WEIGHT_COLS[label]

        b_mask = before_out["topic_label"] == label
        a_mask = after_out["topic_label"] == label

        b_count = int(b_mask.sum())
        a_count = int(a_mask.sum())

        b_share = b_count / len(before_out)
        a_share = a_count / len(after_out)

        # Average soft weight across all docs — more stable metric
        b_mean_w = float(before_out[w_col].mean())
        a_mean_w = float(after_out[w_col].mean())

        summary_rows.append(
            {
                "topic_label": label,
                "count_before": b_count,
                "count_after": a_count,
                "share_before": b_share,
                "share_after": a_share,
                "delta_share": a_share - b_share,
                "mean_weight_before": b_mean_w,
                "mean_weight_after": a_mean_w,
                "delta_mean_weight": a_mean_w - b_mean_w,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return before_out, after_out, summary_df


# -------------------------------------------------------------
# Plotting (grouped bars: before vs after for hard shares)
# -------------------------------------------------------------
def plot_grouped(summary_df: pd.DataFrame, out_path: Path, title: str) -> None:
    summary_df = summary_df.set_index("topic_label").loc[CATEGORY_LIST]

    before = summary_df["share_before"].values
    after = summary_df["share_after"].values
    labels = summary_df.index.tolist()

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(x - width / 2, before, width, label="Before")
    plt.bar(x + width / 2, after, width, label="After")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Share of documents (primary topic)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# Optional: also plot mean weights (so you can compare with hard shares)
def plot_mean_weights(summary_df: pd.DataFrame, out_path: Path, title: str) -> None:
    summary_df = summary_df.set_index("topic_label").loc[CATEGORY_LIST]

    before = summary_df["mean_weight_before"].values
    after = summary_df["mean_weight_after"].values
    labels = summary_df.index.tolist()

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(x - width / 2, before, width, label="Before")
    plt.bar(x + width / 2, after, width, label="After")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Average topic weight")
    plt.title(title + " (mean soft weights)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def run() -> None:
    data = load_all()

    TARGETS = {
        "commit_messages": ("commit_messages_before", "commit_messages_after"),
        "pr_bodies": ("pr_bodies_before", "pr_bodies_after"),
        "issue_bodies": ("issue_bodies_before", "issue_bodies_after"),
        "review_comments": ("review_comments_before", "review_comments_after"),
    }

    for name, (bkey, akey) in TARGETS.items():
        print(f"[rq5] Processing {name} ...")

        before_df = data.get(bkey)
        after_df = data.get(akey)

        if before_df is None or after_df is None:
            print(f"[rq5] Missing data for {name} — skipping.")
            continue

        result = classify_topics(
            before_df,
            after_df,
            text_col="text",
            time_col="date",
            text_kind=name,
        )
        if result is None:
            continue

        before_out, after_out, summary_df = result

        # Save per-document assignments
        doc_path = OUT_TABLES / f"rq5_doc_topics_{name}.csv"
        pd.concat([before_out, after_out], ignore_index=True).to_csv(
            doc_path, index=False
        )
        print(f"[rq5] Saved doc-level topics → {doc_path}")

        # Save summary
        summary_path = OUT_TABLES / f"rq5_topic_summary_{name}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[rq5] Saved topic summary → {summary_path}")

        # Plot grouped bars (hard share)
        plot_grouped(
            summary_df,
            OUT_PLOTS / f"rq5_grouped_{name}.png",
            f"{name}: topic distribution before vs after",
        )

        # Also plot mean weights (soft)
        plot_mean_weights(
            summary_df,
            OUT_PLOTS / f"rq5_meanweights_{name}.png",
            f"{name}: topic weights before vs after",
        )

    print("[rq5] Done.")


if __name__ == "__main__":
    run()
