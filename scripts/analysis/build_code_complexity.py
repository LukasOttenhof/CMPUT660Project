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

OUT_PARQUET = os.path.join(OUTPUT_DIR, "repo_month_complexity.parquet")
CACHE_DIR = os.path.join(OUTPUT_DIR, "repo_month_complexity_cache")

# timeouts (seconds)
GIT_TIMEOUT = 120
FETCH_TIMEOUT = 300
LIZARD_TIMEOUT = 180  # per monthly snapshot

# logging
LOG_EVERY_MONTH = 1  # print after each month

# prune common dependency/build dirs
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


def run_cmd(
    args: List[str],
    *,
    timeout: int,
    cwd: Optional[str] = None,
    text: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=text,
        timeout=timeout,
        env=env,
    )


def parse_repo_list(path: str) -> List[Tuple[str, pd.Timestamp]]:
    specs: List[Tuple[str, pd.Timestamp]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]
            full_name = parts[0]

            boundary = None
            for p in parts[1:]:
                if "first agent pr" in p.lower():
                    boundary_str = p.split(":")[-1].strip()
                    boundary = pd.to_datetime(boundary_str, utc=True, errors="coerce")

            if boundary is None or pd.isna(boundary):
                continue

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
    if not os.path.exists(path):
        return {"sha_to_metrics": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return {"sha_to_metrics": {}}
        if "sha_to_metrics" not in obj or not isinstance(obj["sha_to_metrics"], dict):
            obj["sha_to_metrics"] = {}
        return obj
    except Exception:
        return {"sha_to_metrics": {}}


def save_repo_cache(repo_name: str, cache: Dict[str, Any]) -> None:
    path = cache_path_for_repo(repo_name)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        os.replace(tmp, path)
    except Exception:
        pass


def ensure_full_history(repo_dir: str) -> None:
    shallow_file = os.path.join(repo_dir, ".git", "shallow")
    try:
        if os.path.exists(shallow_file):
            run_cmd(["git", "-C", repo_dir, "fetch", "--unshallow"], timeout=FETCH_TIMEOUT)
        else:
            run_cmd(["git", "-C", repo_dir, "fetch", "--all", "--tags", "--prune"], timeout=FETCH_TIMEOUT)
    except Exception:
        pass


def get_first_last_commit_dates(repo_dir: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    try:
        cp_last = run_cmd(["git", "-C", repo_dir, "log", "-1", "--pretty=format:%cI", "--all"], timeout=GIT_TIMEOUT)
        if cp_last.returncode != 0: return None, None
        last = pd.to_datetime(cp_last.stdout.strip(), utc=True, errors="coerce")

        cp_first_sha = run_cmd(["git", "-C", repo_dir, "rev-list", "--max-parents=0", "--all"], timeout=GIT_TIMEOUT)
        if cp_first_sha.returncode != 0 or not cp_first_sha.stdout.strip(): return None, None
        first_sha = cp_first_sha.stdout.strip().splitlines()[0]

        cp_first_dt = run_cmd(["git", "-C", repo_dir, "show", "-s", "--pretty=format:%cI", first_sha], timeout=GIT_TIMEOUT)
        first = pd.to_datetime(cp_first_dt.stdout.strip(), utc=True, errors="coerce")
        return first, last
    except Exception:
        return None, None


def month_range_inclusive(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    start = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    end = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    months: List[pd.Timestamp] = []
    cur = start
    while cur <= end:
        months.append(cur)
        cur = cur + pd.offsets.MonthBegin(1)
    return months


def month_end_timestamp(month_start: pd.Timestamp) -> pd.Timestamp:
    next_month = month_start + pd.offsets.MonthBegin(1)
    return (next_month - pd.Timedelta(seconds=1)).tz_convert("UTC")


def rev_list_before(repo_dir: str, before_dt: pd.Timestamp) -> Optional[str]:
    before_iso = before_dt.isoformat()
    cp = run_cmd(["git", "-C", repo_dir, "rev-list", "-1", "--all", f"--before={before_iso}"], timeout=GIT_TIMEOUT)
    sha = cp.stdout.strip()
    return sha or None


def get_commit_datetime(repo_dir: str, sha: str) -> Optional[pd.Timestamp]:
    cp = run_cmd(["git", "-C", repo_dir, "show", "-s", "--pretty=format:%cI", sha], timeout=GIT_TIMEOUT)
    dt = pd.to_datetime(cp.stdout.strip(), utc=True, errors="coerce")
    return dt if not pd.isna(dt) else None


def get_tracked_files_with_sizes(repo_dir: str, sha: str) -> List[Tuple[str, int]]:
    cp = run_cmd(["git", "-C", repo_dir, "ls-tree", "-r", "-l", sha], timeout=GIT_TIMEOUT)
    if cp.returncode != 0: return []
    out = []
    for line in cp.stdout.splitlines():
        if "\t" not in line: continue
        left, path = line.split("\t", 1)
        parts = left.split()
        if len(parts) >= 4 and parts[1] == "blob":
            try: out.append((path.strip(), int(parts[3])))
            except: continue
    return out


def should_skip_path(path: str) -> bool:
    lower = path.lower().replace("\\", "/")
    parts = lower.split("/")
    if any(p in SKIP_DIR_NAMES for p in parts): return True
    if lower.endswith(SKIP_FILE_SUFFIXES): return True
    return False


# ==========================================================
# Lizard runner + parsing including length
# ==========================================================

def _parse_totals_from_lizard_stdout(cli_output: str) -> Optional[Dict[str, Any]]:

    function_pattern = r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(.*@.*)"
    functions_info: List[Dict[str, Any]] = []

    for line in cli_output.splitlines():
        match = re.match(function_pattern, line)
        if match:
            nloc, ccn, tokens, params, length, location = match.groups()
            functions_info.append({
                "nloc": int(nloc),
                "ccn": float(ccn),
                "tokens": float(tokens),
                "params": int(params),
                "length": int(length)
            })

    if not functions_info: 
        return None

    funcs = len(functions_info)
    total_nloc = sum(f["nloc"] for f in functions_info)
    
    return {
        "total_nloc": total_nloc,
        "function_count": funcs,
        "avg_nloc": total_nloc / funcs,
        "avg_ccn": sum(f["ccn"] for f in functions_info) / funcs,
        "avg_tokens": sum(f["tokens"] for f in functions_info) / funcs,
        "total_params": sum(f["params"] for f in functions_info),
        "avg_params": sum(f["params"] for f in functions_info) / funcs,
        "total_length": sum(f["length"] for f in functions_info),
        "avg_length": sum(f["length"] for f in functions_info) / funcs
    }


def run_lizard_on_tempdir(temp_dir: str) -> Dict[str, Any]:
    try:
        cp = subprocess.run(["python3", "-m", "lizard", temp_dir], capture_output=True, text=True, timeout=LIZARD_TIMEOUT)
        metrics = _parse_totals_from_lizard_stdout(cp.stdout)
        return metrics if metrics else {"_error": "no_metrics"}
    except Exception as e:
        return {"_error": str(e)}


def materialize_snapshot_to_tempdir(repo_dir: str, sha: str) -> Tuple[Optional[str], int]:
    tracked = get_tracked_files_with_sizes(repo_dir, sha)
    if not tracked: return None, 0
    temp_dir = tempfile.mkdtemp(prefix="repo_snap_")
    files_written = 0
    for rel_path, size in tracked:
        if should_skip_path(rel_path) or size <= 0 or size > MAX_TRACKED_FILE_BYTES: continue
        cp = run_cmd(["git", "-C", repo_dir, "show", f"{sha}:{rel_path}"], timeout=GIT_TIMEOUT, text=False)
        if cp.returncode == 0:
            out_path = os.path.join(temp_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f: f.write(cp.stdout)
            files_written += 1
    if files_written == 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, 0
    return temp_dir, files_written


def compute_month_snapshot_metrics(repo_dir: str, sha: str) -> Dict[str, Any]:
    temp_dir = None
    try:
        temp_dir, nfiles = materialize_snapshot_to_tempdir(repo_dir, sha)
        if not temp_dir:
            return {"total_nloc": 0, "avg_nloc": 0.0, "avg_ccn": 0.0, "avg_tokens": 0.0, "function_count": 0, "files_analyzed": 0, "total_params": 0, "avg_params": 0.0, "total_length": 0, "avg_length": 0.0}
        metrics = run_lizard_on_tempdir(temp_dir)
        if not metrics.get("_error"): metrics["files_analyzed"] = nfiles
        return metrics
    finally:
        if temp_dir: shutil.rmtree(temp_dir, ignore_errors=True)


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    specs = parse_repo_list(REPO_LIST_TXT)
    total_repos = len(specs)

    existing = load_existing_output()
    existing_keys = set()
    rows: List[Dict[str, Any]] = []

    if not existing.empty:
        for _, r in existing.iterrows():
            existing_keys.add((str(r["repo"]), str(r["year_month"])))
        rows = existing.to_dict(orient="records")

    for repo_index, (repo_name, boundary) in enumerate(specs, start=1):
        repo_dir = os.path.join(CLONE_DIR, safe_repo_dirname(repo_name))
        if not os.path.isdir(repo_dir):
            print(f"[{repo_index}/{total_repos}] Skipping {repo_name} (not cloned)")
            continue

        print(f"\n[{repo_index}/{total_repos}] Repository: {repo_name}")
        ensure_full_history(repo_dir)

        first_dt, last_dt = get_first_last_commit_dates(repo_dir)
        if first_dt is None or last_dt is None: continue

        months = month_range_inclusive(first_dt, last_dt)
        cache = load_repo_cache(repo_name)
        sha_cache = cache.get("sha_to_metrics", {})

        for month_start in months:
            year_month = f"{month_start.year:04d}-{month_start.month:02d}"
            if (repo_name, year_month) in existing_keys: continue

            month_end = month_end_timestamp(month_start)
            sha = rev_list_before(repo_dir, month_end)
            if not sha: continue

            dt = get_commit_datetime(repo_dir, sha) or month_end
            period = "before" if dt < boundary else "after"

            metrics = sha_cache.get(sha)
            if not metrics or metrics.get("_error"):
                metrics = compute_month_snapshot_metrics(repo_dir, sha)
                sha_cache[sha] = metrics

            if metrics.get("_error"): continue

            row = {
                "repo": repo_name, "year_month": year_month, "date": dt, "commit_sha": sha, "period": period,
                "total_nloc": int(metrics.get("total_nloc", 0)),
                "avg_nloc": float(metrics.get("avg_nloc", 0.0)),
                "avg_ccn": float(metrics.get("avg_ccn", 0.0)),
                "avg_tokens": float(metrics.get("avg_tokens", 0.0)),
                "function_count": int(metrics.get("function_count", 0)),
                "files_analyzed": int(metrics.get("files_analyzed", 0)),
                "total_params": int(metrics.get("total_params", 0)),
                "avg_params": float(metrics.get("avg_params", 0.0)),
                "total_length": int(metrics.get("total_length", 0)),
                "avg_length": float(metrics.get("avg_length", 0.0)),
            }

            rows.append(row)
            existing_keys.add((repo_name, year_month))
            # PROGRESS PRINT
            print(f"   -> Finished {year_month} (Commit: {sha[:8]})")

        cache["sha_to_metrics"] = sha_cache
        save_repo_cache(repo_name, cache)
        save_output_incremental(rows)

    print("\n--- Done. Final Save. ---")
    save_output_incremental(rows)


if __name__ == "__main__":
    main()
