import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.cluster import KMeans

# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "outputs" / "rq5" / "tables"
PLOT_DIR = ROOT / "outputs" / "rq5" / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

TOPN_PATH = TABLE_DIR / "topic_topN_by_doc.csv"

df = pd.read_csv(TOPN_PATH)

# ============================================================
# Reconstruct group (before/after) from original RQ5 files
# ============================================================

DOC_FILES = {
    "commit_messages": TABLE_DIR / "rq5_doc_topics_commit_messages.csv",
    "pr_bodies": TABLE_DIR / "rq5_doc_topics_pr_bodies.csv",
    "issue_bodies": TABLE_DIR / "rq5_doc_topics_issue_bodies.csv",
    "review_comments": TABLE_DIR / "rq5_doc_topics_review_comments.csv",
}

group_map = {}

for text_type, path in DOC_FILES.items():
    if not path.exists():
        print(f"WARNING: Missing file {path}, skipping group mapping.")
        continue

    original = pd.read_csv(path)

    if "group" not in original.columns:
        raise ValueError(f"Original doc-topic file missing 'group' column: {path}")

    for i in range(len(original)):
        group_map[(text_type, i)] = original.loc[i, "group"]

df["group"] = df.apply(
    lambda r: group_map.get((r["text_type"], r["doc_id"]), None),
    axis=1
)

if df["group"].isna().any():
    print("WARNING: Some rows do not have group filled!")
else:
    print("✔ Successfully reconstructed group column")

# ============================================================
# Determine TOP_N and text types
# ============================================================

TOP_N = len([c for c in df.columns if c.startswith("top") and c.endswith("_weight")])
TEXT_TYPES = df["text_type"].unique()

print(f"Detected TOP_N={TOP_N}")
print(f"Found text types: {TEXT_TYPES}")

# ============================================================
# 1. Top-N Decay Plots (per text type)
# ============================================================

for t in TEXT_TYPES:
    sub = df[df.text_type == t]

    mean_weights = [
        sub[f"top{i+1}_weight"].mean()
        for i in range(TOP_N)
    ]

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, TOP_N + 1), mean_weights, marker="o")
    plt.title(f"Top-{TOP_N} Topic Weight Decay")
    plt.xlabel("Rank")
    plt.ylabel("Mean weight")
    plt.grid(True)

    out_path = PLOT_DIR / f"topN_decay_{t}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✔ Saved decay plot → {out_path}")

# ============================================================
# 1B. Top-N Decay for ALL COMBINED
# ============================================================

df_all = df

mean_weights_all = [
    df_all[f"top{i+1}_weight"].mean()
    for i in range(TOP_N)
]

plt.figure(figsize=(6, 4))
plt.plot(range(1, TOP_N + 1), mean_weights_all, marker="o")
plt.title(f"Top-{TOP_N} Topic Weight Decay")
plt.xlabel("Rank")
plt.ylabel("Mean weight")
plt.grid(True)

out_path = PLOT_DIR / f"topN_decay_ALL_COMBINED.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✔ Saved combined decay plot → {out_path}")

# ============================================================
# 1C. Top-N Decay for BEFORE vs AFTER (ALL COMBINED + per type)
# ============================================================

palette = ["#E74C3C", "#3498DB"]   # BEFORE = red, AFTER = blue

groups = ["before", "after"]

# ---- ALL COMBINED ----
mean_curves = {}
for g in groups:
    sub = df[df["group"] == g]
    mean_curves[g] = [sub[f"top{i+1}_weight"].mean() for i in range(TOP_N)]

plt.figure(figsize=(7, 5))
plt.plot(range(1, TOP_N + 1), mean_curves["before"],
         marker="o", color=palette[0], linewidth=2, label="Before")
plt.plot(range(1, TOP_N + 1), mean_curves["after"],
         marker="o", color=palette[1], linewidth=2, label="After")

plt.title(f"Top-{TOP_N} Topic Weight Decay")
plt.xlabel("Rank")
plt.ylabel("Mean topic weight")
plt.grid(True, alpha=0.3)
plt.legend()

out_path = PLOT_DIR / f"topN_decay_before_after_ALL_COMBINED.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✔ Saved Before/After Top-N decay → {out_path}")

# ---- Per text type ----
for t in TEXT_TYPES:
    sub = df[df.text_type == t]

    before = sub[sub["group"] == "before"]
    after = sub[sub["group"] == "after"]

    mean_before = [before[f"top{i+1}_weight"].mean() for i in range(TOP_N)]
    mean_after  = [after[f"top{i+1}_weight"].mean()  for i in range(TOP_N)]

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, TOP_N + 1), mean_before,
             marker="o", color=palette[0], linewidth=2, label="Before")
    plt.plot(range(1, TOP_N + 1), mean_after,
             marker="o", color=palette[1], linewidth=2, label="After")

    plt.title(f"Top-{TOP_N} Topic Weight Decay: Before vs After — {t}")
    plt.xlabel("Rank")
    plt.ylabel("Mean topic weight")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_path = PLOT_DIR / f"topN_decay_before_after_{t}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✔ Saved Before/After decay for {t} → {out_path}")

# ============================================================
# 2. Entropy per Document (per type)
# ============================================================

entropy_records = []

for t in TEXT_TYPES:
    sub = df[df.text_type == t]

    W = sub[[f"top{i+1}_weight" for i in range(TOP_N)]].values
    ent_vals = entropy(W.T)

    entropy_records.append({
        "text_type": t,
        "mean_entropy": float(ent_vals.mean()),
        "std_entropy": float(ent_vals.std()),
        "min_entropy": float(ent_vals.min()),
        "max_entropy": float(ent_vals.max()),
        "n_docs": len(ent_vals)
    })

    plt.figure(figsize=(6, 4))
    plt.hist(ent_vals, bins=40, alpha=0.75)
    plt.title(f"Entropy Distribution — {t}")
    plt.xlabel("Entropy")
    plt.ylabel("Freq")
    plt.grid(True)

    out_path = PLOT_DIR / f"entropy_hist_{t}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✔ Saved entropy histogram → {out_path}")

# ============================================================
# 2B. Entropy for ALL COMBINED
# ============================================================

W_all = df_all[[f"top{i+1}_weight" for i in range(TOP_N)]].values
entropy_all = entropy(W_all.T)

plt.figure(figsize=(6, 4))
plt.hist(entropy_all, bins=40, alpha=0.75)
plt.title("Entropy Distribution — ALL_COMBINED")
plt.xlabel("Entropy")
plt.ylabel("Freq")
plt.grid(True)

out_path = PLOT_DIR / f"entropy_hist_ALL_COMBINED.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✔ Saved combined entropy histogram → {out_path}")

entropy_records.append({
    "text_type": "ALL_COMBINED",
    "mean_entropy": float(entropy_all.mean()),
    "std_entropy": float(entropy_all.std()),
    "min_entropy": float(entropy_all.min()),
    "max_entropy": float(entropy_all.max()),
    "n_docs": len(entropy_all)
})

entropy_df = pd.DataFrame(entropy_records)
entropy_df.to_csv(TABLE_DIR / "topic_entropy_summary.csv", index=False)
print("✔ Saved entropy summary → topic_entropy_summary.csv")

# ============================================================
# 3. Ambiguous Document Detection
# ============================================================

AMBIG_THRESH = 0.02

ambig_rows = []

for t in TEXT_TYPES:
    sub = df[df.text_type == t]
    margin = sub["top1_weight"] - sub["top2_weight"]
    ambig = sub[margin < AMBIG_THRESH]

    for _, row in ambig.iterrows():
        ambig_rows.append({
            "text_type": t,
            "doc_id": row["doc_id"],
            "group": row["group"],
            "top1_topic": row["top1_topic"],
            "top1_weight": row["top1_weight"],
            "top2_topic": row["top2_topic"],
            "top2_weight": row["top2_weight"],
            "margin": float(row["top1_weight"] - row["top2_weight"])
        })

ambig_df = pd.DataFrame(ambig_rows)
ambig_df.to_csv(TABLE_DIR / "ambiguous_docs.csv", index=False)
print(f"✔ ambiguous docs saved → ambiguous_docs.csv")
print(f"Ambiguous rate = {len(ambig_df)} / {len(df)} = {len(ambig_df)/len(df):.3f}")

# ============================================================
# 4. Clustering documents by topic-weight profile
# ============================================================

cluster_records = []

for t in TEXT_TYPES:
    sub = df[df.text_type == t]
    W = sub[[f"top{i+1}_weight" for i in range(TOP_N)]].values

    k = 4
    km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(W)
    labels = km.labels_

    for doc_idx, cl in zip(sub.index, labels):
        cluster_records.append({
            "text_type": t,
            "doc_id": df.loc[doc_idx, "doc_id"],
            "cluster": int(cl)
        })

cluster_df = pd.DataFrame(cluster_records)
cluster_df.to_csv(TABLE_DIR / "topic_clusters.csv", index=False)
print("✔ Saved topic clusters → topic_clusters.csv")
