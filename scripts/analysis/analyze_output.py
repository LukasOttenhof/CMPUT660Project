import os
import pandas as pd

NUMERIC_PATH = "inputs/processed/numeric_raw.parquet"
TEXT_PATH = "inputs/processed/text_raw.parquet"

print("=======================================")
print("📦 CMPUT660Project Output Assessment")
print("=======================================\n")

# -------------------------------------------------------------
# 1. Check existence
# -------------------------------------------------------------
print("🔍 Checking Parquet file availability...")

if os.path.exists(NUMERIC_PATH):
    print(f"✔ Found {NUMERIC_PATH}")
else:
    print(f"❌ Missing {NUMERIC_PATH}")

if os.path.exists(TEXT_PATH):
    print(f"✔ Found {TEXT_PATH}")
else:
    print(f"❌ Missing {TEXT_PATH}")

print("\n=======================================")
print("📊 Loading data...")
print("=======================================\n")

numeric = pd.read_parquet(NUMERIC_PATH)
text    = pd.read_parquet(TEXT_PATH)

print("Loaded numeric:", numeric.shape, "rows, columns:", list(numeric.columns))
print("Loaded text:", text.shape, "rows, columns:", list(text.columns))

# -------------------------------------------------------------
# 2. Missing values
# -------------------------------------------------------------
print("\n=======================================")
print("📉 Missing Value Summary")
print("=======================================\n")

print("\n=== NUMERIC missing values ===")
print(numeric.isna().sum())

print("\n=== TEXT missing values ===")
print(text.isna().sum())

# -------------------------------------------------------------
# 3. Repo coverage
# -------------------------------------------------------------
print("\n=======================================")
print("📁 Repository Coverage Analysis")
print("=======================================\n")

numeric_repos = set(numeric["repo"].unique())
text_repos    = set(text["repo"].unique())

print(f"Numeric repos: {len(numeric_repos)}")
print(f"Text repos:    {len(text_repos)}\n")

missing_text = numeric_repos - text_repos
missing_numeric = text_repos - numeric_repos

print("Repos missing TEXT data:")
print(missing_text if missing_text else "✔ All repos have text data")

print("\nRepos missing NUMERIC data:")
print(missing_numeric if missing_numeric else "✔ All repos have numeric data")

# -------------------------------------------------------------
# 4. Activity type distribution
# -------------------------------------------------------------
print("\n=======================================")
print("📆 Activity Type Distribution (numeric)")
print("=======================================\n")

print(numeric["activity_type"].value_counts())

# -------------------------------------------------------------
# 5. Contribution counts per repo
# -------------------------------------------------------------
print("\n=======================================")
print("🧪 Per-Repo Contribution Counts")
print("=======================================\n")

counts = numeric.groupby("repo")["author"].count().sort_values(ascending=False)
print(counts.head(20))

# -------------------------------------------------------------
# 6. Weak or missing text output
# -------------------------------------------------------------
print("\n=======================================")
print("🔎 Weak Text Data Detection")
print("=======================================\n")

text_counts = text.groupby("repo")["text"].count().sort_values()

low_text_repos = text_counts[text_counts < 10]

print("Repos with fewer than 10 text items:")
print(low_text_repos if len(low_text_repos) > 0 else "✔ No repos have extremely low text data")

# -------------------------------------------------------------
# 7. Final summary
# -------------------------------------------------------------
print("\n=======================================")
print("✨ FINAL SUMMARY")
print("=======================================\n")

print(f"Total numeric rows: {len(numeric)}")
print(f"Total text rows:    {len(text)}")
print(f"Total repos processed: {len(numeric_repos)}")

if len(missing_text) > 0:
    print(f"\n⚠️ WARNING: {len(missing_text)} repos produced NO text data.")
if len(missing_numeric) > 0:
    print(f"⚠️ WARNING: {len(missing_numeric)} repos produced NO numeric data.")

print("\n✔ Analysis complete.\n")
