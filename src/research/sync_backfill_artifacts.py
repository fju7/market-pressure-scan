from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple


def run(cmd: List[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    if capture:
        return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return subprocess.run(cmd, check=check)


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def latest_success_run_id(workflow_file: str, limit: int = 30) -> str:
    """
    Find latest successful run id for a workflow file, using gh run list.
    """
    # gh run list -w backfill_2weeks.yml -L 20 --json databaseId,conclusion,createdAt -q ...
    q = (
        f'.[] | select(.conclusion=="success") | .databaseId'
    )
    cp = run(
        [
            "gh", "run", "list",
            "-w", workflow_file,
            "-L", str(limit),
            "--json", "databaseId,conclusion,createdAt",
            "-q", q,
        ],
        capture=True,
    )
    rid = (cp.stdout or "").strip().splitlines()
    if not rid:
        raise SystemExit(f"No successful runs found for workflow: {workflow_file}")
    return rid[0].strip()


def download_run_artifacts(run_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["gh", "run", "download", run_id, "-D", str(out_dir)])


def rsync_derived_from_download(download_dir: Path, derived_dir: Path, dry_run: bool = False) -> List[Tuple[Path, Path]]:
    """
    Finds folders in download_dir that contain a 'scores_weekly/' or 'features_weekly/' directory
    at their root, and rsyncs them into derived_dir.
    Returns list of (src_root, derived_dir) sync operations performed.
    """
    candidates: List[Path] = []
    for p in download_dir.iterdir():
        if not p.is_dir():
            continue
        if (p / "scores_weekly").is_dir() or (p / "features_weekly").is_dir() or (p / "reports").is_dir():
            candidates.append(p)

    if not candidates:
        # Sometimes artifacts have an extra level (e.g., derived-backfill-.../scores_weekly/..)
        for p in download_dir.rglob("*"):
            if p.is_dir() and p.name in {"scores_weekly", "features_weekly", "reports"}:
                root = p.parent
                if root not in candidates:
                    candidates.append(root)

    if not candidates:
        raise SystemExit(
            f"Could not find any derived-like artifact folders under {download_dir}.\n"
            f"Tip: run: find {download_dir} -maxdepth 3 -type d | head"
        )

    ops: List[Tuple[Path, Path]] = []
    for root in sorted(set(candidates)):
        cmd = ["rsync", "-av"]
        if dry_run:
            cmd.append("--dry-run")
        cmd += [str(root) + "/", str(derived_dir) + "/"]
        run(cmd)
        ops.append((root, derived_dir))

    return ops


def verify_week(regime: str, schema: str, week: str, derived_dir: Path) -> bool:
    """
    Verify canonical artifacts exist for a week.
    """
    report_md = derived_dir / "reports" / f"week_ending={week}" / "weekly_report.md"
    report_meta = derived_dir / "reports" / f"week_ending={week}" / "report_meta.json"
    scores = (
        derived_dir
        / "scores_weekly"
        / f"regime={regime}"
        / f"schema={schema}"
        / f"week_ending={week}"
        / "scores_weekly.parquet"
    )
    feats = (
        derived_dir
        / "features_weekly"
        / f"regime={regime}"
        / f"schema={schema}"
        / f"week_ending={week}"
        / "features_weekly.parquet"
    )
    ok = report_md.exists() and report_meta.exists() and scores.exists() and feats.exists()
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Download latest successful backfill workflow artifacts and sync into data/derived/")
    ap.add_argument("--workflow", default="backfill_2weeks.yml", help="Workflow file name under .github/workflows/")
    ap.add_argument("--run-id", default="", help="Optional: specific GitHub Actions run id to download")
    ap.add_argument("--derived-dir", default="data/derived", help="Local derived directory to sync into")
    ap.add_argument("--regime", default="news-novelty-v1")
    ap.add_argument("--schema", default="news-novelty-v1b")
    ap.add_argument("--weeks", default="", help="Comma-separated weeks to verify after sync (e.g. 2026-01-02,2026-01-09)")
    ap.add_argument("--dry-run", action="store_true", help="Dry-run rsync only (no changes)")
    ap.add_argument("--keep-download", action="store_true", help="Do not delete the downloaded artifact directory")
    args = ap.parse_args()

    if not have("gh"):
        raise SystemExit("Missing dependency: gh (GitHub CLI)")
    if not have("rsync"):
        raise SystemExit("Missing dependency: rsync")

    derived_dir = Path(args.derived_dir)
    derived_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id.strip()
    if not run_id:
        run_id = latest_success_run_id(args.workflow)

    print(f"Using run_id={run_id} workflow={args.workflow}")

    download_dir = Path(tempfile.mkdtemp(prefix="mps_backfill_artifacts_"))
    print(f"Downloading to: {download_dir}")
    download_run_artifacts(run_id, download_dir)

    # Show a quick inventory
    print("\nDownloaded top-level folders:")
    for p in sorted(download_dir.iterdir()):
        if p.is_dir():
            print(" -", p.name)

    print("\nSyncing into:", derived_dir)
    ops = rsync_derived_from_download(download_dir, derived_dir, dry_run=args.dry_run)
    print(f"\nSync complete. Synced {len(ops)} folder(s).")

    weeks = [w.strip() for w in args.weeks.split(",") if w.strip()]
    if weeks:
        print("\nVerification:")
        for w in weeks:
            ok = verify_week(args.regime, args.schema, w, derived_dir)
            print(f" - {w}: {'OK' if ok else 'MISSING'}")

    if args.keep_download:
        print(f"\nKeeping download dir: {download_dir}")
    else:
        shutil.rmtree(download_dir, ignore_errors=True)
        print("\nCleaned up download dir.")

    print("\nDone.")


if __name__ == "__main__":
    main()
