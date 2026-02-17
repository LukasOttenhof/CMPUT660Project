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

# prune common dependency/build dirs (path contains any of these segments)
SKIP_DIR_NAMES = {
    ".git", "node_modules", "vendor", "dist", "build", "target",
    "bin", "obj", "__pycache__", ".venv", "venv", ".tox",
    ".mypy_cache", ".pytest_cache", ".idea", ".vscode",
}

# optional: avoid minified/bundled files that can explode runtime
SKIP_FILE_SUFFIXES = (".min.js", ".min.css", ".map")

# skip very large tracked files (blob size in bytes)
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
    """
    Best-effort: fetch more history in-code (no manual steps required).
    """
    shallow_file = os.path.join(repo_dir, ".git", "shallow")
    try:
        if os.path.exists(shallow_file):
            print("[info] Repo is shallow. Fetching full history...")
            try:
                subprocess.run(
                    ["git", "-C", repo_dir, "fetch", "--unshallow"],
                    timeout=FETCH_TIMEOUT,
                    capture_output=True,
                )
            except Exception:
                subprocess.run(
                    ["git", "-C", repo_dir, "fetch", "--depth=1000000"],
                    timeout=FETCH_TIMEOUT,
                    capture_output=True,
                )
        else:
            subprocess.run(
                ["git", "-C", repo_dir, "fetch", "--all", "--tags", "--prune"],
                timeout=FETCH_TIMEOUT,
                capture_output=True,
            )
    except Exception:
        pass


def get_first_last_commit_dates(repo_dir: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Robust: first = oldest commit across --all, last = newest commit on HEAD.
    (We avoid relying on origin/HEAD or default branch naming.)
    """
    try:
        cp_last = run_cmd(
            ["git", "-C", repo_dir, "log", "-1", "--pretty=format:%cI", "--all"],
            timeout=GIT_TIMEOUT,
        )
        if cp_last.returncode != 0:
            return None, None
        last = pd.to_datetime(cp_last.stdout.strip(), utc=True, errors="coerce")
        if pd.isna(last):
            return None, None

        cp_first_sha = run_cmd(
            ["git", "-C", repo_dir, "rev-list", "--max-parents=0", "--all"],
            timeout=GIT_TIMEOUT,
        )
        if cp_first_sha.returncode != 0 or not cp_first_sha.stdout.strip():
            return None, None

        first_sha = cp_first_sha.stdout.strip().splitlines()[0]

        cp_first_dt = run_cmd(
            ["git", "-C", repo_dir, "show", "-s", "--pretty=format:%cI", first_sha],
            timeout=GIT_TIMEOUT,
        )
        if cp_first_dt.returncode != 0:
            return None, None

        first = pd.to_datetime(cp_first_dt.stdout.strip(), utc=True, errors="coerce")
        if pd.isna(first):
            return None, None

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
    """
    Latest commit at or before before_dt across *all refs*.
    This avoids default-branch ambiguity and avoids checkout entirely.
    """
    before_iso = before_dt.isoformat()
    try:
        cp = run_cmd(
            ["git", "-C", repo_dir, "rev-list", "-1", "--all", f"--before={before_iso}"],
            timeout=GIT_TIMEOUT,
        )
        if cp.returncode != 0:
            return None
        sha = cp.stdout.strip()
        return sha or None
    except Exception:
        return None


def get_commit_datetime(repo_dir: str, sha: str) -> Optional[pd.Timestamp]:
    try:
        cp = run_cmd(
            ["git", "-C", repo_dir, "show", "-s", "--pretty=format:%cI", sha],
            timeout=GIT_TIMEOUT,
        )
        if cp.returncode != 0:
            return None
        dt = pd.to_datetime(cp.stdout.strip(), utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt
    except Exception:
        return None


def should_skip_path(path: str) -> bool:
    lower = path.lower().replace("\\", "/")
    parts = lower.split("/")
    if any(p in SKIP_DIR_NAMES for p in parts):
        return True
    if lower.endswith(SKIP_FILE_SUFFIXES):
        return True
    return False


# ==========================================================
# Lizard runner for a snapshot WITHOUT checkout
#   - materialize filtered tracked files into a temp dir
#   - run lizard on that temp dir
# ==========================================================

_TOTAL_LINE_RE = re.compile(
    r"Total.*?nloc\s+(\d+).*?average[_ ]nloc\s+([\d.]+).*?average[_ ]ccn\s+([\d.]+).*?"
    r"average[_ ]token[s]?\s+([\d.]+).*?function[_ ]count\s+(\d+)",
    re.IGNORECASE,
)

_FILE_TABLE_RE = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(.+)$")


def _parse_totals_from_lizard_stdout(cli_output: str) -> Optional[Dict[str, Any]]:
    """
    Parse Lizard CLI output into totals using per-function metrics,
    including number of parameters (variables) per function.
    """
    function_pattern = r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.+)"
    functions_info: List[Dict[str, Any]] = []

    # Extract per-function info
    for line in cli_output.splitlines():
        match = re.match(function_pattern, line)
        if match:
            nloc, ccn, tokens, params, length, location = match.groups()
            functions_info.append({
                "nloc": int(nloc),
                "ccn": int(ccn),
                "tokens": int(tokens),
                "params": int(params),
                "length": int(length),
                "location": location.strip()
            })

    if not functions_info:
        return None

    # Aggregate totals across all functions
    total_nloc = sum(f["nloc"] for f in functions_info)
    function_count = len(functions_info)
    avg_nloc = total_nloc / function_count if function_count else 0
    avg_ccn = sum(f["ccn"] for f in functions_info) / function_count if function_count else 0
    avg_tokens = sum(f["tokens"] for f in functions_info) / function_count if function_count else 0
    total_params = sum(f["params"] for f in functions_info)
    avg_params = total_params / function_count if function_count else 0

    return {
        "total_nloc": total_nloc,
        "avg_nloc": avg_nloc,
        "avg_ccn": avg_ccn,
        "avg_tokens": avg_tokens,
        "function_count": function_count,
        "total_params": total_params,
        "avg_params": avg_params
    }


def get_tracked_files_with_sizes(repo_dir: str, sha: str) -> List[Tuple[str, int]]:
    """
    Uses: git ls-tree -r -l <sha>
    Returns list of (path, size_bytes). Size may be -1 for submodules; we skip those.
    """
    try:
        cp = run_cmd(
            ["git", "-C", repo_dir, "ls-tree", "-r", "-l", sha],
            timeout=GIT_TIMEOUT,
        )
        if cp.returncode != 0:
            return []
    except Exception:
        return []

    out: List[Tuple[str, int]] = []
    for line in cp.stdout.splitlines():
        # format: <mode> <type> <object> <size>\t<file>
        # example: 100644 blob abcd123 1234\tpath/to/file.py
        if "\t" not in line:
            continue
        left, path = line.split("\t", 1)
        parts = left.split()
        if len(parts) < 4:
            continue
        ftype = parts[1]
        if ftype != "blob":
            continue
        size_str = parts[3]
        try:
            size = int(size_str)
        except Exception:
            continue
        out.append((path.strip(), size))
    return out


def materialize_snapshot_to_tempdir(repo_dir: str, sha: str) -> Tuple[Optional[str], int]:
    """
    Writes filtered tracked files for commit sha into a temp directory.
    Returns (temp_dir, files_written).
    """
    tracked = get_tracked_files_with_sizes(repo_dir, sha)
    if not tracked:
        return None, 0

    temp_dir = tempfile.mkdtemp(prefix="repo_snapshot_")
    files_written = 0

    for rel_path, size in tracked:
        if should_skip_path(rel_path):
            continue
        if size <= 0:
            continue
        if size > MAX_TRACKED_FILE_BYTES:
            continue

        # pull bytes without checking out
        try:
            cp = run_cmd(
                ["git", "-C", repo_dir, "show", f"{sha}:{rel_path}"],
                timeout=GIT_TIMEOUT,
                text=False,
            )
            if cp.returncode != 0:
                continue
            data: bytes = cp.stdout or b""
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue

        if not data:
            continue

        out_path = os.path.join(temp_dir, rel_path)
        out_dir = os.path.dirname(out_path)
        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(data)
            files_written += 1
        except Exception:
            continue

    if files_written == 0:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        return None, 0

    return temp_dir, files_written


def run_lizard_on_tempdir(temp_dir: str) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        cp = subprocess.run(
            ["python", "-m", "lizard", temp_dir],
            capture_output=True,
            timeout=LIZARD_TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"_error": "timeout"}
    except Exception as e:
        return {"_error": f"exception:{type(e).__name__}"}

    try:
        out = cp.stdout.decode("utf-8", errors="ignore")
    except Exception:
        return {"_error": "decode_error"}

    totals = _parse_totals_from_lizard_stdout(out)
    if not totals:
        return {"_error": "no_summary"}

    return totals


def compute_month_snapshot_metrics(repo_dir: str, sha: str) -> Dict[str, Any]:
    """
    Full snapshot complexity for commit sha WITHOUT checkout.
    """
    temp_dir = None
    try:
        temp_dir, nfiles = materialize_snapshot_to_tempdir(repo_dir, sha)
        if not temp_dir:
            # no analyzable files for this snapshot
            return {
                "total_nloc": 0,
                "avg_nloc": 0.0,
                "avg_ccn": 0.0,
                "avg_tokens": 0.0,
                "function_count": 0,
                "files_analyzed": 0,
                "total_params": 0,
                "avg_params": 0.0,
            }


        metrics = run_lizard_on_tempdir(temp_dir)
        if metrics.get("_error"):
            return metrics

        metrics["files_analyzed"] = nfiles
        return metrics
    finally:
        if temp_dir:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


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

        ensure_full_history(repo_dir)

        print("\n" + "=" * 80)
        print(f"[{repo_index}/{total_repos}] Processing repo: {repo_name}")
        print("=" * 80)

        first_dt, last_dt = get_first_last_commit_dates(repo_dir)
        if first_dt is None or last_dt is None:
            print("Could not get first/last commit dates. Skipping.")
            continue

        months = month_range_inclusive(first_dt, last_dt)
        if not months:
            print("No months produced. Skipping.")
            continue

        cache = load_repo_cache(repo_name)
        sha_cache: Dict[str, Dict[str, Any]] = cache.get("sha_to_metrics", {})
        if not isinstance(sha_cache, dict):
            sha_cache = {}

        cache_hits_repo = 0
        cache_misses_repo = 0
        snapshots_written_repo = 0

        for idx, month_start in enumerate(months, start=1):
            year_month = f"{month_start.year:04d}-{month_start.month:02d}"
            if (repo_name, year_month) in existing_keys:
                continue

            month_end = month_end_timestamp(month_start)
            sha = rev_list_before(repo_dir, month_end)
            if not sha:
                if LOG_EVERY_MONTH:
                    print(f"[{repo_name}] {year_month} done | ERROR=no_commit_before_month_end")
                continue

            # timestamp for this snapshot
            dt = get_commit_datetime(repo_dir, sha) or month_end
            period = "before" if dt < boundary else "after"

            cached_metrics = sha_cache.get(sha)
            if isinstance(cached_metrics, dict) and not cached_metrics.get("_error"):
                metrics = cached_metrics
                cache_hits_repo += 1
            else:
                cache_misses_repo += 1
                metrics = compute_month_snapshot_metrics(repo_dir, sha)
                sha_cache[sha] = metrics  # cache even errors so we don't re-do pain

            if metrics.get("_error"):
                if LOG_EVERY_MONTH:
                    print(
                        f"[{repo_name}] {year_month} done | ERROR={metrics['_error']} "
                        f"| cache_hits={cache_hits_repo} cache_misses={cache_misses_repo}"
                    )
                continue

            row = {
                "repo": repo_name,
                "year_month": year_month,
                "date": dt,
                "commit_sha": sha,
                "period": period,
                "total_nloc": int(metrics.get("total_nloc", 0)),
                "avg_nloc": float(metrics.get("avg_nloc", 0.0)),
                "avg_ccn": float(metrics.get("avg_ccn", 0.0)),
                "avg_tokens": float(metrics.get("avg_tokens", 0.0)),
                "function_count": int(metrics.get("function_count", 0)),
                "files_analyzed": int(metrics.get("files_analyzed", 0)),
                "total_params": int(metrics.get("total_params", 0)),   # NEW
                "avg_params": float(metrics.get("avg_params", 0.0)),   # NEW
            }


            rows.append(row)
            existing_keys.add((repo_name, year_month))
            snapshots_written_repo += 1

            if LOG_EVERY_MONTH and (idx % LOG_EVERY_MONTH == 0 or idx == len(months)):
                print(
                    f"[{repo_name}] {year_month} done | "
                    f"total_nloc={row['total_nloc']} avg_ccn={row['avg_ccn']} files={row['files_analyzed']} | "
                    f"cache_hits={cache_hits_repo} cache_misses={cache_misses_repo}"
                )

            # periodic persistence so long runs are safe
            if snapshots_written_repo % 6 == 0:
                cache["sha_to_metrics"] = sha_cache
                save_repo_cache(repo_name, cache)
                save_output_incremental(rows)

        cache["sha_to_metrics"] = sha_cache
        save_repo_cache(repo_name, cache)
        save_output_incremental(rows)

        print(f"\nFinished repo: {repo_name}")
        print(f"   Months in range: {len(months)}")
        print(f"   Snapshots written: {snapshots_written_repo}")
        print(f"   Cache hits (repo): {cache_hits_repo}")
        print(f"   Cache misses (repo): {cache_misses_repo}")
        print("-" * 80)

    save_output_incremental(rows)
    print("\nAll repos complete.")
    print(f"Output parquet: {OUT_PARQUET}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
