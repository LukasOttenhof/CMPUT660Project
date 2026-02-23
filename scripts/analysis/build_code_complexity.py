from __future__ import annotations

import os
import re
import json
import shutil
import tempfile
import subprocess
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

REPO_LIST_TXT = "/home/lottenho/660_pro/CMPUT660Project/filtered_repos_3year100star.txt"
CLONE_DIR = "/home/lottenho/660_pro/CMPUT660Project/scripts/build_dataset/repos"
OUTPUT_DIR = "/home/lottenho/660_pro/CMPUT660Project/inputs/processed"

OUT_PARQUET = os.path.join(OUTPUT_DIR, "repo_month_complexity_detailed.parquet")
CACHE_DIR = os.path.join(OUTPUT_DIR, "repo_month_complexity_cache")

# timeouts (seconds)
GIT_TIMEOUT = 120
FETCH_TIMEOUT = 300
LIZARD_TIMEOUT = 180  # per monthly snapshot

# logging
LOG_EVERY_MONTH = 1  

SKIP_DIR_NAMES = {
    ".git", "node_modules", "vendor", "dist", "build", "target",
    "bin", "obj", "__pycache__", ".venv", "venv", ".tox",
    ".mypy_cache", ".pytest_cache", ".idea", ".vscode",
}

SKIP_FILE_SUFFIXES = (".min.js", ".min.css", ".map")
MAX_TRACKED_FILE_BYTES = 2_000_000  # 2MB

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ==========================================================
# Helpers
# ==========================================================

def safe_repo_dirname(full_name: str) -> str:
    return full_name.replace("/", "_")

def run_cmd(args, *, timeout, cwd=None, text=True, env=None):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=text, timeout=timeout, env=env)

def parse_repo_list(path: str) -> List[Tuple[str, pd.Timestamp]]:
    specs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = [p.strip() for p in line.split("|")]
            full_name = parts[0]
            boundary = None
            for p in parts[1:]:
                if "first agent pr" in p.lower():
                    boundary_str = p.split(":")[-1].strip()
                    boundary = pd.to_datetime(boundary_str, utc=True, errors="coerce")
            if boundary is not None and not pd.isna(boundary):
                specs.append((full_name, boundary))
    return specs

def load_existing_output() -> pd.DataFrame:
    if os.path.exists(OUT_PARQUET):
        try:
            return pd.read_parquet(OUT_PARQUET)
        except Exception:
            print(f"[warn] Could not read existing parquet: {OUT_PARQUET}. Starting fresh.")
    return pd.DataFrame()

def save_output_incremental(rows: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df["year_month"] = df["year_month"].astype(str)
        df = df.sort_values(["repo", "date"])
    df.to_parquet(OUT_PARQUET, index=False)

def cache_path_for_repo(repo_name: str) -> str:
    return os.path.join(CACHE_DIR, f"{safe_repo_dirname(repo_name)}_cache.json")

def load_repo_cache(repo_name: str) -> Dict[str, Any]:
    path = cache_path_for_repo(repo_name)
    if not os.path.exists(path): return {"sha_to_metrics": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {"sha_to_metrics": {}}
    except Exception: return {"sha_to_metrics": {}}

def save_repo_cache(repo_name: str, cache: Dict[str, Any]) -> None:
    path = cache_path_for_repo(repo_name)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        os.replace(tmp, path)
    except Exception: pass

# ==========================================================
# GIT HISTORY & REPO UTILS
# ==========================================================

def ensure_full_history(repo_dir: str) -> None:
    shallow_file = os.path.join(repo_dir, ".git", "shallow")
    try:
        if os.path.exists(shallow_file):
            subprocess.run(["git", "-C", repo_dir, "fetch", "--unshallow"], timeout=FETCH_TIMEOUT, capture_output=True)
        else:
            subprocess.run(["git", "-C", repo_dir, "fetch", "--all", "--tags", "--prune"], timeout=FETCH_TIMEOUT, capture_output=True)
    except Exception: pass

def get_first_last_commit_dates(repo_dir: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    try:
        cp_last = run_cmd(["git", "-C", repo_dir, "log", "-1", "--pretty=format:%cI", "--all"], timeout=GIT_TIMEOUT)
        last = pd.to_datetime(cp_last.stdout.strip(), utc=True, errors="coerce")
        cp_first_sha = run_cmd(["git", "-C", repo_dir, "rev-list", "--max-parents=0", "--all"], timeout=GIT_TIMEOUT)
        first_sha = cp_first_sha.stdout.strip().splitlines()[0]
        cp_first_dt = run_cmd(["git", "-C", repo_dir, "show", "-s", "--pretty=format:%cI", first_sha], timeout=GIT_TIMEOUT)
        first = pd.to_datetime(cp_first_dt.stdout.strip(), utc=True, errors="coerce")
        return first, last
    except Exception: return None, None

def month_range_inclusive(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    start = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    end = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    months, cur = [], start
    while cur <= end:
        months.append(cur)
        cur = cur + pd.offsets.MonthBegin(1)
    return months

def month_end_timestamp(month_start: pd.Timestamp) -> pd.Timestamp:
    next_month = month_start + pd.offsets.MonthBegin(1)
    return (next_month - pd.Timedelta(seconds=1)).tz_convert("UTC")

def rev_list_before(repo_dir: str, before_dt: pd.Timestamp) -> Optional[str]:
    before_iso = before_dt.isoformat()
    try:
        cp = run_cmd(["git", "-C", repo_dir, "rev-list", "-1", "--all", f"--before={before_iso}"], timeout=GIT_TIMEOUT)
        sha = cp.stdout.strip()
        return sha or None
    except Exception: return None

def get_commit_datetime(repo_dir: str, sha: str) -> Optional[pd.Timestamp]:
    try:
        cp = run_cmd(["git", "-C", repo_dir, "show", "-s", "--pretty=format:%cI", sha], timeout=GIT_TIMEOUT)
        return pd.to_datetime(cp.stdout.strip(), utc=True, errors="coerce")
    except Exception: return None

# ==========================================================
# LIZARD PARSING (UPDATED)
# ==========================================================

# Regex for function level details
# Example: 4 2 22 0 5 test@2-6@path
_FUNCTION_PATTERN = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.+)")

# Regex for the summary lines (Total or per-file summary)
_SUMMARY_PATTERN = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(.+)$")

def parse_detailed_lizard(stdout_text: str) -> Dict[str, Any]:
    functions_info = []
    max_values = {
        "max_nloc": 0, "max_ccn": 0, "max_tokens": 0, 
        "max_params": 0, "max_length": 0
    }
    summary_data = None
    
    lines = stdout_text.splitlines()
    for line in lines:
        # 1. Try Function Match
        f_match = _FUNCTION_PATTERN.match(line)
        if f_match:
            nloc, ccn, tokens, params, length, loc = f_match.groups()
            nloc, ccn, tokens, params, length = int(nloc), int(ccn), int(tokens), int(params), int(length)
            
            functions_info.append({
                "nloc": nloc, "ccn": ccn, "tokens": tokens,
                "params": params, "length": length, "location": loc.strip()
            })
            
            # Update Maxes
            if nloc > max_values["max_nloc"]: max_values["max_nloc"] = nloc
            if ccn > max_values["max_ccn"]: max_values["max_ccn"] = ccn
            if tokens > max_values["max_tokens"]: max_values["max_tokens"] = tokens
            if params > max_values["max_params"]: max_values["max_params"] = params
            if length > max_values["max_length"]: max_values["max_length"] = length
            continue

        # 2. Try Summary/Total Match
        s_match = _SUMMARY_PATTERN.match(line)
        if s_match:
            # Skip if it looks like the header
            if not any(c.isdigit() for c in s_match.group(1)): continue
            
            nloc, avg_nloc, avg_ccn, avg_tokens, func_count, name = s_match.groups()
            
            # We prioritize the row that says "Total" or is the very last summary row
            if "total" in name.lower() or summary_data is None:
                summary_data = {
                    "total_nloc": int(nloc),
                    "avg_nloc": float(avg_nloc),
                    "avg_ccn": float(avg_ccn),
                    "avg_tokens": float(avg_tokens),
                    "function_count": int(func_count)
                }

    if not summary_data and not functions_info:
        return {"_error": "no_metrics_found"}

    result = summary_data if summary_data else {}
    result.update(max_values)
    result["functions_info"] = functions_info
    return result

def get_tracked_files_with_sizes(repo_dir: str, sha: str) -> List[Tuple[str, int]]:
    try:
        cp = run_cmd(["git", "-C", repo_dir, "ls-tree", "-r", "-l", sha], timeout=GIT_TIMEOUT)
        out = []
        for line in cp.stdout.splitlines():
            if "\t" not in line: continue
            left, path = line.split("\t", 1)
            parts = left.split()
            if len(parts) >= 4 and parts[1] == "blob":
                out.append((path.strip(), int(parts[3])))
        return out
    except Exception: return []

def materialize_snapshot_to_tempdir(repo_dir: str, sha: str) -> Tuple[Optional[str], int]:
    tracked = get_tracked_files_with_sizes(repo_dir, sha)
    if not tracked: return None, 0
    temp_dir = tempfile.mkdtemp(prefix="repo_snapshot_")
    written = 0
    for rel_path, size in tracked:
        lower = rel_path.lower().replace("\\", "/")
        if any(p in SKIP_DIR_NAMES for p in lower.split("/")) or lower.endswith(SKIP_FILE_SUFFIXES): continue
        if size <= 0 or size > MAX_TRACKED_FILE_BYTES: continue
        try:
            cp = run_cmd(["git", "-C", repo_dir, "show", f"{sha}:{rel_path}"], timeout=GIT_TIMEOUT, text=False)
            if cp.returncode == 0 and cp.stdout:
                out_path = os.path.join(temp_dir, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "wb") as f: f.write(cp.stdout)
                written += 1
        except Exception: continue
    if written == 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, 0
    return temp_dir, written

def compute_month_snapshot_metrics(repo_dir: str, sha: str) -> Dict[str, Any]:
    temp_dir = None
    try:
        temp_dir, nfiles = materialize_snapshot_to_tempdir(repo_dir, sha)
        if not temp_dir: return {"total_nloc": 0, "files_analyzed": 0, "functions_info": []}
        
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        cp = subprocess.run(["python", "-m", "lizard", temp_dir], capture_output=True, timeout=LIZARD_TIMEOUT, env=env, text=True)
        
        metrics = parse_detailed_lizard(cp.stdout)
        if "_error" not in metrics: metrics["files_analyzed"] = nfiles
        return metrics
    finally:
        if temp_dir: shutil.rmtree(temp_dir, ignore_errors=True)

# ==========================================================
# MAIN
# ==========================================================

def main():
    specs = parse_repo_list(REPO_LIST_TXT)
    existing = load_existing_output()
    existing_keys = set()
    rows = []

    if not existing.empty:
        for _, r in existing.iterrows():
            existing_keys.add((str(r["repo"]), str(r["year_month"])))
        rows = existing.to_dict(orient="records")

    for repo_idx, (repo_name, boundary) in enumerate(specs, start=1):
        repo_dir = os.path.join(CLONE_DIR, safe_repo_dirname(repo_name))
        if not os.path.isdir(repo_dir): continue

        ensure_full_history(repo_dir)
        print(f"\n>>> [{repo_idx}/{len(specs)}] {repo_name}")

        first_dt, last_dt = get_first_last_commit_dates(repo_dir)
        if not first_dt or not last_dt: continue

        months = month_range_inclusive(first_dt, last_dt)
        cache = load_repo_cache(repo_name)
        sha_cache = cache.get("sha_to_metrics", {})

        for month_start in months:
            year_month = f"{month_start.year:04d}-{month_start.month:02d}"
            if (repo_name, year_month) in existing_keys: continue

            sha = rev_list_before(repo_dir, month_end_timestamp(month_start))
            if not sha: continue

            dt = get_commit_datetime(repo_dir, sha) or month_start
            period = "before" if dt < boundary else "after"

            if sha in sha_cache and not sha_cache[sha].get("_error"):
                metrics = sha_cache[sha]
            else:
                metrics = compute_month_snapshot_metrics(repo_dir, sha)
                sha_cache[sha] = metrics

            if metrics.get("_error"): continue

            row = {
                "repo": repo_name,
                "year_month": year_month,
                "date": dt,
                "commit_sha": sha,
                "period": period,
                "total_nloc": metrics.get("total_nloc", 0),
                "avg_ccn": metrics.get("avg_ccn", 0.0),
                "max_nloc": metrics.get("max_nloc", 0),
                "max_ccn": metrics.get("max_ccn", 0),
                "max_tokens": metrics.get("max_tokens", 0),
                "max_params": metrics.get("max_params", 0),
                "max_length": metrics.get("max_length", 0),
                "function_count": metrics.get("function_count", 0),
                "files_analyzed": metrics.get("files_analyzed", 0),
                "functions_info": metrics.get("functions_info", []) # Stored as list of dicts
            }
            rows.append(row)
            existing_keys.add((repo_name, year_month))

        cache["sha_to_metrics"] = sha_cache
        save_repo_cache(repo_name, cache)
        save_output_incremental(rows)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()