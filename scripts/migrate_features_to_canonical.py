from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import re
import sys


def week_from_path(p: Path) -> str | None:
    m = re.search(r"week_ending=(\d{4}-\d{2}-\d{2})", str(p))
    return m.group(1) if m else None


def discover_weeks_from_scores(derived_root: Path, regime: str, schema: str) -> list[str]:
    scores_root = derived_root / "scores_weekly" / f"regime={regime}" / f"schema={schema}"
    if not scores_root.exists():
        raise FileNotFoundError(f"Canonical scores root not found: {scores_root}")

    weeks: list[str] = []
    for f in scores_root.rglob("scores_weekly.parquet"):
        w = week_from_path(f)
        if w:
            weeks.append(w)
    return sorted(set(weeks))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Copy legacy features_weekly.parquet into canonical regime+schema layout (idempotent)."
    )
    ap.add_argument("--regime", required=True, help="Regime ID, e.g. news-novelty-v1")
    ap.add_argument("--schema", required=True, help="Schema ID, e.g. news-novelty-v1b")
    ap.add_argument(
        "--derived_root",
        default="data/derived",
        help="Derived root directory (default: data/derived)",
    )
    ap.add_argument(
        "--weeks",
        default="",
        help="Optional comma-separated week_end list (YYYY-MM-DD). If omitted, derives weeks from canonical scores.",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print actions without copying files.",
    )

    args = ap.parse_args()
    regime: str = args.regime
    schema: str = args.schema
    derived_root = Path(args.derived_root)

    if args.weeks.strip():
        weeks = [w.strip() for w in args.weeks.split(",") if w.strip()]
    else:
        weeks = discover_weeks_from_scores(derived_root, regime, schema)

    if not weeks:
        print("No weeks found. Nothing to do.")
        return 0

    print(f"Derived root: {derived_root.resolve()}")
    print(f"Regime: {regime}")
    print(f"Schema: {schema}")
    print(f"Weeks ({len(weeks)}): {', '.join(weeks)}")
    print(f"Dry-run: {args.dry_run}")

    features_root = derived_root / "features_weekly"

    copied = 0
    skipped = 0
    missing = 0

    for w in weeks:
        dest = features_root / f"regime={regime}" / f"schema={schema}" / f"week_ending={w}" / "features_weekly.parquet"
        if dest.exists():
            print(f"OK   {w} canonical exists -> {dest}")
            skipped += 1
            continue

        # Preferred legacy source: regime-only
        src_regime_only = features_root / f"regime={regime}" / f"week_ending={w}" / "features_weekly.parquet"
        # Fallback legacy source: flat
        src_flat = features_root / f"week_ending={w}" / "features_weekly.parquet"

        src = None
        if src_regime_only.exists():
            src = src_regime_only
        elif src_flat.exists():
            src = src_flat

        if src is None:
            print(f"MISS {w} no legacy features found (checked {src_regime_only} and {src_flat})")
            missing += 1
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            print(f"COPY {w} {src} -> {dest}")
        else:
            shutil.copy2(src, dest)
            print(f"COPY {w} {src} -> {dest}")
        copied += 1

    print("----")
    print(f"Copied:  {copied}")
    print(f"Skipped: {skipped}")
    print(f"Missing: {missing}")

    # Return non-zero if anything was missing (useful in CI checks)
    return 0 if missing == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
