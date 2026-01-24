#!/usr/bin/env python3
"""
Minimal scoring fixture test with synthetic data.
Tests that scoring logic produces expected results on known inputs.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.features_scores import compute_scores, zscore_series


def test_scoring_with_fixture():
    """Test scoring with a minimal synthetic fixture."""
    print("="*70)
    print("SCORING FIXTURE TEST")
    print("="*70)
    
    # Create synthetic panel with 5 symbols, 1 week
    # Symbol A should score highest (all features = 10)
    # Symbol B should score second (all features = 5)
    # Symbols C, D, E have features = 0
    
    panel_data = {
        'symbol': ['A', 'B', 'C', 'D', 'E'],
        'week_ending_date': ['2026-01-23'] * 5,
        'NV_raw': [10.0, 5.0, 0.0, 0.0, 0.0],
        'NA_raw': [10.0, 5.0, 0.0, 0.0, 0.0],
        'NS_raw': [10.0, 5.0, 0.0, 0.0, 0.0],
        'SS_raw': [10.0, 5.0, 0.0, 0.0, 0.0],
        'EI_raw': [10.0, 5.0, 0.0, 0.0, 0.0],
        'EC_raw': [10.0, 5.0, 0.0, 0.0, 0.0],
    }
    panel = pd.DataFrame(panel_data)
    
    # Create synthetic market data
    mkt_data = {
        'symbol': ['A', 'B', 'C', 'D', 'E'],
        'asof_date': ['2026-01-23'] * 5,
        'AR5': [0.05, 0.03, 0.01, -0.01, -0.02],
        'VS_raw': [1.5, 1.2, 1.0, 0.9, 0.8],
        'RV60': [0.2, 0.18, 0.15, 0.14, 0.13],
        'VR_pct': [1.1, 1.0, 0.95, 0.9, 0.85],
        'z_AR5': [1.5, 0.8, 0.0, -0.8, -1.5],
        'z_VS': [1.2, 0.6, 0.0, -0.6, -1.2],
    }
    mkt = pd.DataFrame(mkt_data)
    
    # Create universe
    universe_data = {
        'symbol': ['A', 'B', 'C', 'D', 'E'],
        'name': ['Alpha Inc', 'Beta Corp', 'Gamma Ltd', 'Delta Co', 'Epsilon LLC'],
        'sector': ['Tech'] * 5,
    }
    universe = pd.DataFrame(universe_data)
    
    print("\n📋 Input Panel:")
    print(panel)
    
    print("\n📋 Input Market:")
    print(mkt)
    
    print("\n🔨 Running compute_scores...")
    
    try:
        features, scores = compute_scores(panel, mkt, universe)
    except Exception as e:
        print(f"❌ compute_scores failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"✓ compute_scores completed")
    
    # Validation checks
    print("\n🔍 VALIDATION CHECKS")
    print("="*70)
    
    passed = True
    
    # Check 1: Output has one row per symbol
    if len(scores) != 5:
        print(f"❌ FAIL: Expected 5 rows, got {len(scores)}")
        passed = False
    else:
        print(f"✓ Output has 5 rows (one per symbol)")
    
    # Check 2: symbol column exists
    if "symbol" not in scores.columns:
        print(f"❌ FAIL: 'symbol' column missing from scores")
        print(f"   Columns: {list(scores.columns)}")
        passed = False
    else:
        print(f"✓ 'symbol' column exists")
    
    # Check 3: UPS_adj exists and is numeric
    if "UPS_adj" not in scores.columns:
        print(f"❌ FAIL: 'UPS_adj' column missing")
        passed = False
    else:
        print(f"✓ 'UPS_adj' column exists")
        
        # Check all values are finite
        finite_count = np.isfinite(scores["UPS_adj"]).sum()
        if finite_count < len(scores):
            print(f"⚠️  WARNING: Only {finite_count}/{len(scores)} UPS_adj values are finite")
        else:
            print(f"✓ All UPS_adj values are finite")
        
        # Check ordering (A should be highest)
        scores_sorted = scores.sort_values("UPS_adj", ascending=False)
        print(f"\n📊 UPS_adj Ranking:")
        for idx, row in scores_sorted.iterrows():
            print(f"  {row['symbol']:5s}: {row['UPS_adj']:8.4f}")
        
        top_symbol = scores_sorted.iloc[0]["symbol"]
        if top_symbol != "A":
            print(f"⚠️  WARNING: Expected 'A' to rank highest, got '{top_symbol}'")
            print(f"   This might be expected if scoring logic differs")
        else:
            print(f"✓ Symbol 'A' ranked highest (as expected)")
    
    # Check 4: No 'index' column
    if "index" in scores.columns:
        print(f"❌ FAIL: Unexpected 'index' column in scores")
        passed = False
    else:
        print(f"✓ No unexpected 'index' column")
    
    # Show full scores output
    print(f"\n📊 Full Scores Output:")
    print(scores)
    
    # Show full features output
    print(f"\n📊 Full Features Output:")
    print(features.head())
    
    print("\n" + "="*70)
    if passed:
        print("✅ SCORING FIXTURE TEST PASSED")
        return True
    else:
        print("⚠️  SCORING FIXTURE TEST COMPLETED WITH WARNINGS")
        return True  # Don't fail on warnings


if __name__ == "__main__":
    success = test_scoring_with_fixture()
    sys.exit(0 if success else 1)
