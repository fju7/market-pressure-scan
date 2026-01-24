#!/usr/bin/env python3
"""
Test script to validate feature panel contract independently.
Run this to diagnose panel construction issues without running full pipeline.

Usage:
    python test_panel_contract.py --week_end 2026-01-23
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def test_panel_contract(week_end: str):
    """Test feature panel contract for a specific week."""
    print(f"Testing feature panel contract for week_end={week_end}")
    print("="*70)
    
    # Import after arguments parsed to catch import errors
    try:
        import src.features_scores as fs
    except ImportError as e:
        print(f"❌ Failed to import src.features_scores: {e}")
        return False
    
    # Setup paths
    paths = fs.default_paths()
    print(f"\n📁 Paths:")
    print(f"  Root: {paths.root}")
    print(f"  Derived: {paths.derived}")
    
    # Load universe
    universe_path = Path("sp500_universe.csv")
    if not universe_path.exists():
        print(f"❌ Universe file not found: {universe_path}")
        return False
    
    universe = fs.load_universe(universe_path)
    print(f"  Universe: {len(universe)} symbols")
    
    # Parse week_end
    try:
        week_end_et = fs.parse_week_end(week_end)
    except Exception as e:
        print(f"❌ Failed to parse week_end '{week_end}': {e}")
        return False
    
    print(f"\n🔨 Building feature panel...")
    
    # Build panel
    try:
        panel = fs.build_news_feature_panel(
            paths=paths,
            universe=universe,
            week_end_et=week_end_et,
            week_end=week_end,
            lookback_weeks=12,
        )
    except Exception as e:
        print(f"❌ build_news_feature_panel failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"✓ Panel built successfully")
    
    # Contract validation
    print(f"\n🔍 CONTRACT VALIDATION")
    print("="*70)
    
    passed = True
    
    # 1. symbol is a column
    if "symbol" not in panel.columns:
        print(f"❌ FAIL: 'symbol' is not a column")
        print(f"   Columns: {list(panel.columns)}")
        print(f"   Index names: {panel.index.names}")
        passed = False
    else:
        print(f"✓ 'symbol' is a column")
    
    # 2. week_ending_date is present
    if "week_ending_date" not in panel.columns:
        print(f"❌ FAIL: 'week_ending_date' missing")
        print(f"   Columns: {list(panel.columns)}")
        passed = False
    else:
        print(f"✓ 'week_ending_date' is present")
    
    # 3. No 'index' column
    if "index" in panel.columns:
        print(f"❌ FAIL: Unexpected 'index' column found")
        print(f"   This indicates unnamed index materialization")
        passed = False
    else:
        print(f"✓ No unexpected 'index' column")
    
    # 4. Required features
    required = ["NV_raw", "NA_raw", "NS_raw", "SS_raw", "EI_raw", "EC_raw"]
    missing = [f for f in required if f not in panel.columns]
    if missing:
        print(f"❌ FAIL: Missing required features: {missing}")
        passed = False
    else:
        print(f"✓ All {len(required)} required features present")
    
    # 5. Uniqueness check (if symbol and week_ending_date exist)
    if "symbol" in panel.columns and "week_ending_date" in panel.columns:
        dup_count = panel.duplicated(["symbol", "week_ending_date"]).sum()
        if dup_count > 0:
            print(f"❌ FAIL: {dup_count} duplicate (symbol, week_ending_date) pairs")
            dups = panel[panel.duplicated(["symbol", "week_ending_date"], keep=False)]
            print(f"   Sample duplicates:")
            print(dups[["symbol", "week_ending_date"]].head(10))
            passed = False
        else:
            print(f"✓ No duplicate (symbol, week_ending_date) pairs")
    
    # Diagnostics
    print(f"\n📊 PANEL DIAGNOSTICS")
    print("="*70)
    print(f"Shape: {panel.shape}")
    print(f"Columns ({len(panel.columns)}): {list(panel.columns)}")
    print(f"Index: {panel.index.names}")
    print(f"Memory usage: {panel.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    if "symbol" in panel.columns:
        print(f"\nSymbols: {panel['symbol'].nunique()} unique")
        print(f"Sample symbols: {sorted(panel['symbol'].unique())[:10]}")
    
    if "week_ending_date" in panel.columns:
        print(f"\nWeeks: {panel['week_ending_date'].nunique()} unique")
        print(f"Week values: {sorted(panel['week_ending_date'].unique())}")
    
    # Show sample rows
    print(f"\nSample rows (first 3):")
    print(panel.head(3))
    
    # Check for NaNs in key features
    if all(f in panel.columns for f in required):
        print(f"\nNaN counts in required features:")
        for feat in required:
            nan_count = panel[feat].isna().sum()
            nan_pct = nan_count / len(panel) * 100
            print(f"  {feat:12s}: {nan_count:4d} ({nan_pct:5.1f}%)")
    
    print("\n" + "="*70)
    if passed:
        print("✅ ALL CONTRACT CHECKS PASSED")
        return True
    else:
        print("❌ CONTRACT VALIDATION FAILED")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test feature panel contract")
    parser.add_argument(
        "--week_end",
        required=True,
        help="Week ending date (YYYY-MM-DD)"
    )
    args = parser.parse_args()
    
    success = test_panel_contract(args.week_end)
    sys.exit(0 if success else 1)
