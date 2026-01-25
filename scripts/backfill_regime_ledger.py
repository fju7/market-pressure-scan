#!/usr/bin/env python
"""
Backfill v1b regime and maintain a ledger for comparison.

This script:
1. Runs offline rescoring for v1b on multiple weeks
2. Maintains a ledger comparing v1 vs v1b decisions
3. Answers: "How often did each regime trade?", "What did it pick?", "When did it trade different kinds of weeks?"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def get_week_list(start_week: str, end_week: str) -> List[str]:
    """Generate list of Friday dates between start and end (inclusive)"""
    start = datetime.fromisoformat(start_week)
    end = datetime.fromisoformat(end_week)
    
    # Ensure start is a Friday
    if start.weekday() != 4:
        raise ValueError(f"Start week {start_week} is not a Friday")
    
    weeks = []
    current = start
    while current <= end:
        weeks.append(current.date().isoformat())
        current += timedelta(days=7)
    
    return weeks


def get_git_sha(repo_path: Path) -> str:
    """Get current git SHA"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            text=True
        ).strip()
    except Exception:
        return "unknown"


def rescore_week(week_end: str, regime_id: str, repo_path: Path) -> bool:
    """Run offline rescoring for a single week"""
    print(f"\n📊 Rescoring week {week_end} with regime {regime_id}...")
    
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "src.rescore_week",
                "--week_end", week_end,
                "--schema", regime_id,
                "--offline"
            ],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"❌ Rescoring failed:\n{result.stderr}")
            return False
        
        print(f"✅ Rescoring complete")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"❌ Rescoring timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Rescoring error: {e}")
        return False


def extract_week_metadata(week_end: str, regime_id: str, base_dir: Path) -> Dict[str, Any]:
    """Extract metadata from a scored week"""
    score_dir = base_dir / "scores_weekly" / f"regime={regime_id}" / f"week_ending={week_end}"
    
    meta_path = score_dir / "report_meta.json"
    score_path = score_dir / "scores_weekly.parquet"
    
    metadata = {
        "week_end": week_end,
        "regime_id": regime_id,
        "regime_hash": None,
        "git_sha": None,
        "is_skip": None,
        "skip_reasons": [],
        "n_candidates": 0,
        "top_symbols": [],
        "exists": False,
    }
    
    # Read report_meta.json
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            metadata["regime_hash"] = meta.get("schema_hash")
            metadata["git_sha"] = meta.get("git_sha")
            metadata["exists"] = True
        except Exception as e:
            print(f"  ⚠️  Failed to read {meta_path}: {e}")
    
    # Read scores to determine skip/trade and top symbols
    if score_path.exists():
        try:
            scores = pd.read_parquet(score_path)
            
            if len(scores) == 0:
                metadata["is_skip"] = True
                metadata["skip_reasons"] = ["NO_CANDIDATES"]
            else:
                # Assume trade if we have scores
                metadata["is_skip"] = False
                metadata["n_candidates"] = len(scores)
                
                # Top 10 symbols by UPS_adj
                if "UPS_adj" in scores.columns and "symbol" in scores.columns:
                    top10 = scores.nlargest(10, "UPS_adj")["symbol"].tolist()
                    metadata["top_symbols"] = top10
                    
        except Exception as e:
            print(f"  ⚠️  Failed to read {score_path}: {e}")
    
    return metadata


def build_ledger(weeks: List[str], regimes: List[str], base_dir: Path) -> pd.DataFrame:
    """Build comparison ledger for multiple regimes"""
    print("\n📋 Building comparison ledger...")
    
    records = []
    
    for week in weeks:
        print(f"\n  Week {week}:")
        week_record = {"week_end": week}
        
        for regime in regimes:
            meta = extract_week_metadata(week, regime, base_dir)
            
            if not meta["exists"]:
                print(f"    {regime}: ⚠️  Not found")
                week_record[f"{regime}_exists"] = False
                continue
            
            print(f"    {regime}: {'SKIP' if meta['is_skip'] else 'TRADE'} ({meta['n_candidates']} candidates)")
            
            week_record[f"{regime}_exists"] = True
            week_record[f"{regime}_is_skip"] = meta["is_skip"]
            week_record[f"{regime}_n_candidates"] = meta["n_candidates"]
            week_record[f"{regime}_top5"] = ",".join(meta["top_symbols"][:5])
            week_record[f"{regime}_hash"] = meta["regime_hash"]
            
        records.append(week_record)
    
    return pd.DataFrame(records)


def analyze_ledger(ledger: pd.DataFrame, regimes: List[str]):
    """Analyze and print ledger insights"""
    print("\n" + "=" * 60)
    print("📊 LEDGER ANALYSIS")
    print("=" * 60)
    
    n_weeks = len(ledger)
    print(f"\nTotal weeks analyzed: {n_weeks}\n")
    
    for regime in regimes:
        skip_col = f"{regime}_is_skip"
        exists_col = f"{regime}_exists"
        
        if skip_col not in ledger.columns:
            print(f"{regime}: No data")
            continue
        
        exists = ledger[exists_col].fillna(False)
        skips = ledger[exists][skip_col].fillna(True)
        
        n_exists = exists.sum()
        n_skip = skips.sum()
        n_trade = n_exists - n_skip
        
        skip_rate = (n_skip / n_exists * 100) if n_exists > 0 else 0
        trade_rate = (n_trade / n_exists * 100) if n_exists > 0 else 0
        
        print(f"{regime}:")
        print(f"  Weeks scored: {n_exists}/{n_weeks} ({n_exists/n_weeks*100:.1f}%)")
        print(f"  Trade: {n_trade} ({trade_rate:.1f}%)")
        print(f"  Skip:  {n_skip} ({skip_rate:.1f}%)")
    
    # Compare regimes (if we have v1 and v1b)
    if "news-novelty-v1" in regimes and "news-novelty-v1b" in regimes:
        print("\n" + "-" * 60)
        print("Regime Comparison (v1 vs v1b):")
        print("-" * 60)
        
        both_exist = (
            ledger["news-novelty-v1_exists"].fillna(False) &
            ledger["news-novelty-v1b_exists"].fillna(False)
        )
        
        comparable = ledger[both_exist].copy()
        n_comparable = len(comparable)
        
        if n_comparable == 0:
            print("  No weeks with both regimes scored")
            return
        
        v1_skip = comparable["news-novelty-v1_is_skip"].fillna(True)
        v1b_skip = comparable["news-novelty-v1b_is_skip"].fillna(True)
        
        # Agreement categories
        both_trade = (~v1_skip) & (~v1b_skip)
        both_skip = v1_skip & v1b_skip
        v1_only = (~v1_skip) & v1b_skip
        v1b_only = v1_skip & (~v1b_skip)
        
        print(f"\nWeeks with both regimes: {n_comparable}")
        print(f"  Both trade: {both_trade.sum()} ({both_trade.sum()/n_comparable*100:.1f}%)")
        print(f"  Both skip:  {both_skip.sum()} ({both_skip.sum()/n_comparable*100:.1f}%)")
        print(f"  v1 only:    {v1_only.sum()} ({v1_only.sum()/n_comparable*100:.1f}%)")
        print(f"  v1b only:   {v1b_only.sum()} ({v1b_only.sum()/n_comparable*100:.1f}%)")
        
        # Show v1b-only weeks (signal absence vs model blindness)
        if v1b_only.any():
            print(f"\nWeeks where v1b traded but v1 skipped:")
            for _, row in comparable[v1b_only].iterrows():
                print(f"  {row['week_end']}: v1b picked {row['news-novelty-v1b_n_candidates']} candidates")


def main():
    parser = argparse.ArgumentParser(description="Backfill regime and maintain ledger")
    parser.add_argument("--start_week", required=True, help="Start week (YYYY-MM-DD, Friday)")
    parser.add_argument("--end_week", required=True, help="End week (YYYY-MM-DD, Friday)")
    parser.add_argument("--regimes", nargs="+", default=["news-novelty-v1", "news-novelty-v1b"],
                        help="Regime IDs to backfill")
    parser.add_argument("--skip_rescore", action="store_true",
                        help="Skip rescoring, just analyze existing data")
    parser.add_argument("--output", type=Path, help="Output ledger CSV path")
    
    args = parser.parse_args()
    
    repo_path = Path(__file__).resolve().parents[1]
    base_dir = repo_path / "data" / "derived"
    
    # Get week list
    weeks = get_week_list(args.start_week, args.end_week)
    print(f"📅 Backfilling {len(weeks)} weeks from {args.start_week} to {args.end_week}")
    print(f"   Regimes: {', '.join(args.regimes)}")
    
    # Rescore if not skipped
    if not args.skip_rescore:
        print("\n" + "=" * 60)
        print("🔄 RESCORING")
        print("=" * 60)
        
        for regime in args.regimes:
            print(f"\nRegime: {regime}")
            success_count = 0
            
            for week in weeks:
                if rescore_week(week, regime, repo_path):
                    success_count += 1
            
            print(f"\n✅ Rescored {success_count}/{len(weeks)} weeks for {regime}")
    
    # Build ledger
    ledger = build_ledger(weeks, args.regimes, base_dir)
    
    # Analyze
    analyze_ledger(ledger, args.regimes)
    
    # Save ledger
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(args.output, index=False)
        print(f"\n💾 Saved ledger to {args.output}")
    else:
        default_path = base_dir / "regime_backfill_ledger.csv"
        ledger.to_csv(default_path, index=False)
        print(f"\n💾 Saved ledger to {default_path}")
    
    print("\n✅ Backfill complete")


if __name__ == "__main__":
    main()
