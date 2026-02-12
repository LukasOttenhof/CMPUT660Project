from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import List


REPO_LIST_TXT = "filtered_repos_3year100star.txt"
CLONE_DIR = "repos"


# ==========================================================
# Repo parsing (minimal)
# ==========================================================

@dataclass
class RepoSpec:
    full_name: str


def parse_repo_list(path: str) -> List[RepoSpec]:
    specs: List[RepoSpec] = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"Repo list not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # First field before "|" is the repo name
            parts = [p.strip() for p in line.split("|")]
            full_name = parts[0]

            if "/" not in full_name:
                print(f"Skipping malformed repo name: {full_name}")
                continue

            specs.append(RepoSpec(full_name=full_name))

    return specs


# ==========================================================
# Clone logic
# ==========================================================

def safe_repo_dirname(full_name: str) -> str:
    return full_name.replace("/", "_")


def ensure_clone(full_name: str) -> str:
    os.makedirs(CLONE_DIR, exist_ok=True)
    repo_dir = os.path.join(CLONE_DIR, safe_repo_dirname(full_name))

    # Reuse existing clone
    if os.path.isdir(repo_dir) and os.path.isdir(os.path.join(repo_dir, ".git")):
        print(f"✔ Using existing clone: {repo_dir}")
        return repo_dir

    print(f"⬇ Cloning {full_name} → {repo_dir}")
    url = f"https://github.com/{full_name}.git"

    subprocess.run(
        ["git", "clone", "--quiet", url, repo_dir],
        check=True
    )

    print(f"✔ Finished cloning {full_name}")
    return repo_dir


# ==========================================================
# Main
# ==========================================================

def main():
    specs = parse_repo_list(REPO_LIST_TXT)
    print(f"📦 Found {len(specs)} repositories to clone.\n")

    for i, spec in enumerate(specs, start=1):
        print("=" * 70)
        print(f"[{i}/{len(specs)}] Processing {spec.full_name}")
        print("=" * 70)

        try:
            ensure_clone(spec.full_name)
        except Exception as e:
            print(f"⚠ Failed to clone {spec.full_name}: {e}")

    print("\n✅ Clone process complete.")
    print(f"Repositories are in ./{CLONE_DIR}/")


if __name__ == "__main__":
    main()
