#!/usr/bin/env python3
"""
Quick test to verify add_roll preserves symbol correctly.
"""

import pandas as pd
import sys

def test_add_roll_symbol_preservation():
    """Test that add_roll never loses symbol column."""
    
    # Mock weekly_counts data similar to what features_scores creates
    data = {
        'week_ending_date': ['2026-01-09', '2026-01-16', '2026-01-09', '2026-01-16', '2026-01-09', '2026-01-16'],
        'symbol': ['AAPL', 'AAPL', 'MSFT', 'MSFT', 'GOOGL', 'GOOGL'],
        'total_clusters': [5, 7, 3, 4, 6, 8],
        'total_raw_echo': [10.0, 15.0, 7.0, 9.0, 12.0, 16.0],
        'EC_raw': [2.0, 2.1, 2.3, 2.25, 2.0, 2.0],
        'unique_sources_mean': [1.5, 1.8, 1.2, 1.4, 1.6, 1.7]
    }
    weekly_counts = pd.DataFrame(data)
    weekly_counts['week_dt'] = pd.to_datetime(weekly_counts['week_ending_date'])
    weekly_counts = weekly_counts.sort_values(['symbol', 'week_dt'])
    
    print("BEFORE add_roll:")
    print(f"  Columns: {list(weekly_counts.columns)}")
    print(f"  Index names: {weekly_counts.index.names}")
    print(f"  Has 'symbol': {'symbol' in weekly_counts.columns}")
    print(f"  Has 'index': {'index' in weekly_counts.columns}")
    
    # Define add_roll exactly as in features_scores.py (updated version)
    def add_roll(g: pd.DataFrame) -> pd.DataFrame:
        sym = getattr(g, "name", None)
        
        if isinstance(sym, tuple):
            raise ValueError(f"add_roll: unexpected tuple group key g.name={sym} (groupby changed?)")
        
        if sym is None:
            raise ValueError("add_roll: missing group key (g.name is None)")
        
        # Debug print for AAPL
        if sym == "AAPL":
            print(f"\nDEBUG add_roll: g.name={sym}, type={type(sym)}")
            print(f"DEBUG add_roll: columns={list(g.columns)}, index.name={g.index.name}, index.names={getattr(g.index, 'names', None)}")
        
        g = g.copy()
        
        # Force symbol column to exist
        if "symbol" in g.columns:
            bad = g["symbol"].notna() & (g["symbol"] != sym)
            if bad.any():
                raise ValueError(f"add_roll: symbol column inconsistent with group key {sym}")
        g["symbol"] = sym
        
        # Perform rolling computations
        g = g.sort_values("week_dt")
        g["count_5d_dedup"] = g["total_clusters"]
        g["count_20d_dedup"] = g["total_clusters"].rolling(window=4, min_periods=1).sum()
        g["count_60d_dedup"] = g["total_clusters"].rolling(window=12, min_periods=1).sum()
        
        # Final invariant
        if "symbol" not in g.columns:
            raise ValueError("add_roll: BUG - symbol dropped inside add_roll")
        
        return g
    
    # Apply the function
    result = (
        weekly_counts
        .groupby("symbol", group_keys=False, sort=False)
        .apply(add_roll)
        .reset_index(drop=True)
    )
    
    print("\nAFTER add_roll:")
    print(f"  Columns: {list(result.columns)}")
    print(f"  Index names: {result.index.names}")
    print(f"  Has 'symbol': {'symbol' in result.columns}")
    print(f"  Has 'index': {'index' in result.columns}")
    
    # Validation checks
    if "index" in result.columns:
        print("\n❌ FAIL: 'index' column found!")
        print(result.head())
        return False
    
    if "symbol" not in result.columns:
        print("\n❌ FAIL: 'symbol' column missing!")
        print(result.head())
        return False
    
    # Verify all expected symbols are present
    expected_symbols = {'AAPL', 'MSFT', 'GOOGL'}
    actual_symbols = set(result['symbol'].unique())
    if expected_symbols != actual_symbols:
        print(f"\n❌ FAIL: Expected symbols {expected_symbols}, got {actual_symbols}")
        return False
    
    print("\n✅ PASS: symbol column preserved correctly!")
    print("\nSample output:")
    print(result[['symbol', 'week_ending_date', 'count_5d_dedup', 'count_20d_dedup', 'count_60d_dedup']].head(10))
    
    return True

if __name__ == "__main__":
    success = test_add_roll_symbol_preservation()
    sys.exit(0 if success else 1)
