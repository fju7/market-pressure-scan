#!/usr/bin/env python
"""
Test script to validate regime system components.
Run this to verify all parts are working correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def test_regime_loading():
    """Test regime config loading"""
    print("=" * 60)
    print("TEST 1: Regime Config Loading")
    print("=" * 60)
    
    from src.regime_config import load_regime
    
    try:
        v1 = load_regime("news-novelty-v1")
        print(f"✓ Loaded {v1.name}")
        print(f"  Skip rules: min_novelty={v1.skip_rules.min_total_novelty}")
        print(f"  Weights: nov={v1.scoring_weights.novelty}, div={v1.scoring_weights.divergence}")
        
        v1b = load_regime("news-novelty-v1b")
        print(f"✓ Loaded {v1b.name}")
        print(f"  Skip rules: min_novelty={v1b.skip_rules.min_total_novelty}")
        print(f"  Weights: nov={v1b.scoring_weights.novelty}, div={v1b.scoring_weights.divergence}")
        
        assert v1.scoring_weights.divergence == 0.0, "v1 should have no divergence"
        assert v1b.scoring_weights.divergence == 0.45, "v1b should have 45% divergence"
        
        print("\n✅ Regime loading: PASS\n")
        return True
    except Exception as e:
        print(f"\n❌ Regime loading: FAIL - {e}\n")
        return False


def test_derived_paths():
    """Test path helper"""
    print("=" * 60)
    print("TEST 2: Derived Paths Helper")
    print("=" * 60)
    
    from src.derived_paths import DerivedPaths
    
    try:
        dp = DerivedPaths()
        
        # Test without regime
        path1 = dp.week_dir("features_weekly", "2026-01-09", regime=None)
        assert "regime=" not in str(path1), "Path without regime should not contain regime="
        print(f"✓ Path without regime: {path1}")
        
        # Test with regime
        path2 = dp.week_dir("features_weekly", "2026-01-09", regime="news-novelty-v1b")
        assert "regime=news-novelty-v1b" in str(path2), "Path should contain regime"
        print(f"✓ Path with regime: {path2}")
        
        # Test file creation (creates dirs)
        file_path = dp.file("test_artifact", "test.txt", "2026-01-09", regime="test-regime")
        assert file_path.parent.exists(), "Directories should be created"
        print(f"✓ File path with auto-mkdir: {file_path}")
        
        # Cleanup
        file_path.parent.rmdir()
        file_path.parent.parent.rmdir()
        
        print("\n✅ Derived paths: PASS\n")
        return True
    except Exception as e:
        print(f"\n❌ Derived paths: FAIL - {e}\n")
        return False


def test_atomic_writes():
    """Test atomic Parquet writer"""
    print("=" * 60)
    print("TEST 3: Atomic Parquet Writes")
    print("=" * 60)
    
    from src.io_atomic import write_parquet_atomic
    
    try:
        # Create test data
        df = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "score": [1.5, 2.3, 1.8]
        })
        
        test_path = Path("data/derived/test_atomic.parquet")
        test_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write atomically
        write_parquet_atomic(df, test_path)
        print(f"✓ Wrote test parquet to {test_path}")
        
        # Verify readable
        df_read = pd.read_parquet(test_path)
        assert len(df_read) == 3, "Should read 3 rows"
        assert list(df_read.columns) == ["symbol", "score"], "Columns should match"
        print(f"✓ Successfully read back {len(df_read)} rows")
        
        # Cleanup
        test_path.unlink()
        
        print("\n✅ Atomic writes: PASS\n")
        return True
    except Exception as e:
        print(f"\n❌ Atomic writes: FAIL - {e}\n")
        return False


def test_divergence_computation():
    """Test divergence feature computation"""
    print("=" * 60)
    print("TEST 4: Divergence Computation")
    print("=" * 60)
    
    from src.divergence import compute_5d_returns, add_divergence_feature
    
    try:
        # Check if candles exist
        candles_path = Path("data/derived/market_daily/candles_daily.parquet")
        if not candles_path.exists():
            print(f"⚠ Skipping: {candles_path} not found")
            print("  Run: python -m src.ingest_market_candles --universe sp500_universe.csv --week_end 2026-01-09")
            return None  # Skip, not a failure
        
        candles = pd.read_parquet(candles_path)
        print(f"✓ Loaded {len(candles)} candle records")
        
        # Test 5d returns
        rets = compute_5d_returns(candles, "2026-01-09")
        assert len(rets) > 0, "Should compute returns for multiple symbols"
        assert "ret_5d" in rets.columns, "Should have ret_5d column"
        print(f"✓ Computed 5d returns for {len(rets)} symbols")
        
        # Test divergence feature
        features = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "NV_raw": [0.8, 0.3, 0.5]
        })
        
        features_div = add_divergence_feature(features, candles, "2026-01-09")
        assert "divergence" in features_div.columns, "Should add divergence column"
        assert "excess_ret_5d" in features_div.columns, "Should add excess_ret_5d"
        print(f"✓ Added divergence features")
        print(f"  Sample divergence values: {features_div['divergence'].describe()['mean']:.6f} (mean)")
        
        print("\n✅ Divergence computation: PASS\n")
        return True
    except Exception as e:
        print(f"\n❌ Divergence computation: FAIL - {e}\n")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("REGIME SYSTEM VALIDATION TESTS")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("Regime Loading", test_regime_loading()))
    results.append(("Derived Paths", test_derived_paths()))
    results.append(("Atomic Writes", test_atomic_writes()))
    results.append(("Divergence", test_divergence_computation()))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    
    for name, result in results:
        status = "✅ PASS" if result is True else ("❌ FAIL" if result is False else "⚠ SKIP")
        print(f"{status:10s} {name}")
    
    print()
    print(f"Passed:  {passed}/{len(results)}")
    print(f"Failed:  {failed}/{len(results)}")
    print(f"Skipped: {skipped}/{len(results)}")
    print("=" * 60)
    
    if failed > 0:
        print("\n❌ Some tests failed. Fix issues before proceeding.")
        sys.exit(1)
    else:
        print("\n✅ All critical tests passed!")
        if skipped > 0:
            print(f"⚠ {skipped} test(s) skipped (missing data, likely candles)")
        sys.exit(0)


if __name__ == "__main__":
    main()
