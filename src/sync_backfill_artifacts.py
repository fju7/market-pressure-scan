# src/sync_backfill_artifacts.py
# OVERWRITE_MARKER: 2026-02-03_clean_no_toplevel_exec
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Optional


def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command with nice error messages."""
    try:
        return subprocess.run(cmd, check=check, text=True, capture_output=capture)
    except FileNotFoundError:
        raise SystemExit(f"Missing required executable: {cmd[0]!r}. Is it installed / on PATH?")
    except subprocess.CalledProcessError as e:
        msg = f"Command failed (exit={e.returncode}): {' '.join(cmd)}"
        if getattr(e, "stdout", None):
            msg += f"\n--- stdout ---\n{e.stdout}"
        if getattr(e, "stderr", None):
            msg += f"\n--- stderr ---\n{e.stderr}"
        raise SystemExit(msg)


def require_gh() -> None:
    run(["gh", "--version"], check=True, capture=True)


def ensure_gh_auth() -> None:
    p = run(["gh", "auth", "status"], check=False, capture=True)
    if p.returncode != 0:
        raise SystemExit("GitHub CLI is not authenticated. Run: gh auth login")


def pick_latest_success_run_id(workflow: str, branch: Optional[str], repo: Optional[str]) -> int:
    cmd = [
        "gh",
        "run",
        "list",
        "--workflow",
        workflow,
        "--limit",
        "30",
        "--json",
        "databaseId,conclusion,createdAt",
    ]
    if branch:
        cmd += ["--branch", branch]
    if repo:
        cmd += ["--repo", repo]

    out = run(cmd, check=True, capture=True).stdout
    runs = json.loads(out)

    success = [r for r in runs if r.get("conclusion") == "success"]
    pick = success[0] if success else (runs[0] if runs else None)
    if not pick:
        raise SystemExit(f"No runs found for workflow={workflow!r} (branch={branch!r}).")
    return int(pick["databaseId"])


def iter_zip_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.zip")


def find_all_derived_roots(search_root: Path) -> list[Path]:
    """
    Find directories that should be treated as a "derived root" inside the downloaded/unzipped artifacts.

    Preferred: any .../data/derived directory.
    Fallback: artifact layouts that already contain derived content at top-level (e.g. downloads/derived-YYYY-MM-DD/).
    """
    # 1) Preferred: .../data/derived
    hits: list[Path] = []
    for p in search_root.rglob("data/derived"):
        if p.is_dir():
            hits.append(p)

    def _dedup(paths: list[Path]) -> list[Path]:
        uniq: list[Path] = []
        seen: set[str] = set()
        for h in sorted(paths):
            s = str(h)
            if s not in seen:
                uniq.append(h)
                seen.add(s)
        return uniq

    hits = _dedup(hits)
    if hits:
        return hits

    # 2) Fallback: treat "derived-*" directories as derived roots if they look like derived content
    candidates: list[Path] = []
    for d in search_root.iterdir():
        if d.is_dir() and d.name.startswith("derived"):
            candidates.append(d)

    # Some artifacts might have an extra nested "derived" directory
    for d in list(candidates):
        nested = d / "derived"
        if nested.is_dir():
            candidates.append(nested)

    expected_markers = [
        "reports",
        "scores_weekly",
        "rep_enriched",
        "news_clusters",
        "company_news",
        "baskets",
        "backtest",
        "market_daily",
        "features_weekly",
        "features_scores",
        "scoreboards",
        "trader_sheets",
    ]

    roots: list[Path] = []
    for c in candidates:
        if any((c / m).exists() for m in expected_markers):
            roots.append(c)

    roots = _dedup(roots)
    return roots

def rsync_copy(src: Path, dest: Path, dry_run: bool) -> int:
    """
    Additive rsync: copy src -> dest without deleting anything in dest.
    Returns rsync return code (0 ok, 24 = vanished warnings).
    """
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-av"]
    if dry_run:
        cmd += ["--dry-run"]
    cmd += [str(src) + "/", str(dest) + "/"]
    p = run(cmd, check=False, capture=False)
    return p.returncode


def main() -> None:
    ap = argparse.ArgumentParser("Download + unzip + sync GitHub Actions backfill artifacts into data/derived")
    ap.add_argument("--workflow", default="backfill_2weeks.yml", help="Workflow filename under .github/workflows")
    ap.add_argument("--branch", default=None, help="Branch to filter runs (default: no filter)")
    ap.add_argument("--repo", default=None, help="Optional owner/repo, e.g. fred/market-pressure-scan (default: current)")
    ap.add_argument("--run_id", type=int, default=None, help="Explicit run databaseId to download (skips auto-pick)")
    ap.add_argument("--dest", default="data/derived", help="Destination derived dir (default: data/derived)")
    ap.add_argument("--out_dir", default=None, help="Where to stage downloads (default: temp dir)")
    ap.add_argument("--keep", action="store_true", help="Keep the staging directory (don’t delete temp)")
    ap.add_argument("--dry_run", action="store_true", help="Dry-run rsync (shows what would change)")
    args = ap.parse_args()

    require_gh()
    ensure_gh_auth()

    dest = Path(args.dest).resolve()
    if dest.name != "derived":
        print(f"NOTE: dest is {dest} (expected to end with .../data/derived). Proceeding anyway.")

    temp_ctx: Optional[tempfile.TemporaryDirectory] = None
    if args.out_dir:
        stage = Path(args.out_dir).resolve()
        stage.mkdir(parents=True, exist_ok=True)
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="mps_backfill_artifacts_")
        stage = Path(temp_ctx.name).resolve()

    downloads_root = stage / "downloads"
    unzipped = stage / "unzipped"
    downloads_root.mkdir(parents=True, exist_ok=True)
    unzipped.mkdir(parents=True, exist_ok=True)

    try:
        run_id = args.run_id or pick_latest_success_run_id(args.workflow, args.branch, args.repo)
        print(f"▶ Using run_id={run_id} workflow={args.workflow!r} branch={args.branch!r}")
        print(f"▶ Staging dir: {stage}")

        # IMPORTANT: download each run into its own unique directory to avoid
        # gh extraction collisions ("file exists") when re-running.
        downloads = downloads_root / f"run_id={run_id}"
        if downloads.exists():
            # If user re-runs with same run_id + --keep, start clean.
            # This is safe because downloads is staging-only.
            import shutil
            shutil.rmtree(downloads)
        downloads.mkdir(parents=True, exist_ok=True)

        cmd = ["gh", "run", "download", str(run_id), "-D", str(downloads)]
        if args.repo:
            cmd += ["--repo", args.repo]
        run(cmd, check=True, capture=False)

        # Some gh versions download artifacts already extracted into folders.
        # So: unzip zips if present, but if no zips exist, treat downloads as the extracted root.
        zips = list(iter_zip_files(downloads))
        if zips:
            for z in zips:
                target = unzipped / z.stem
                target.mkdir(parents=True, exist_ok=True)
                run(["unzip", "-o", str(z), "-d", str(target)], check=True, capture=False)
            search_root = unzipped
        else:
            search_root = downloads

        derived_roots = find_all_derived_roots(search_root)
        if not derived_roots:
            print(f"\nDEBUG: No 'data/derived' found under: {search_root}")
            print("DEBUG: Top-level contents:")
            try:
                for p in sorted(search_root.iterdir()):
                    print("  -", p)
            except Exception:
                pass
            raise SystemExit(
                "Downloaded artifacts but did not find a 'data/derived' directory inside them.\n"
                "Likely the workflow uploads paths that don't include the 'data/derived/' prefix."
            )

        print("\nFound these artifact derived roots:")
        for r in derived_roots:
            print(f"  - {r}")

        # Determine what kind of root we have.
        # If root contains "data/derived", that's the canonical subroot.
        # Otherwise assume the artifact root itself is a "derived-like" root.
        def as_effective_root(r: Path) -> Path:
            dd = r / "data" / "derived"
            return dd if dd.is_dir() else r

        SAFE_SUBDIRS = ["rep_enriched", "features_weekly", "scores_weekly", "reports", "backtest"]

        print(f"\n▶ Syncing into: {dest} (additive; no deletes)")
        overall_rc = 0
        for r in derived_roots:
            root = as_effective_root(r)
            print(f"\nSource root: {root}")

            for sub in SAFE_SUBDIRS:
                s = root / sub
                if not s.is_dir():
                    continue
                d = dest / sub
                rc = rsync_copy(s, d, dry_run=bool(args.dry_run))
                if rc not in (0, 24):
                    raise SystemExit(f"rsync failed (exit={rc}) for {s} -> {d}")
                overall_rc = max(overall_rc, rc)

        print("\n✅ Done.")
        if overall_rc == 24:
            print("NOTE: rsync reported 'vanished files' warnings (code 24). This is usually harmless for staging trees.")

        print("\n✅ Done.")
        if args.dry_run:
            print("NOTE: This was a dry-run. Re-run without --dry_run to apply changes.")
        if args.keep:
            print(f"NOTE: Keeping staging dir: {stage}")

    finally:
        if temp_ctx is not None and (not args.keep):
            temp_ctx.cleanup()


if __name__ == "__main__":
    main()
