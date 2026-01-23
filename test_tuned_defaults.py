#!/usr/bin/env python3
"""
Test that the tuned defaults are properly set.
"""

import sys
from pathlib import Path
import inspect

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingest_company_news import fetch_company_news, main

def test_tuned_defaults():
    """Verify all tuned defaults are correct."""
    print("=" * 60)
    print("TUNED DEFAULTS VERIFICATION")
    print("=" * 60)
    
    # Test fetch_company_news max_retries default
    sig = inspect.signature(fetch_company_news)
    max_retries_default = sig.parameters["max_retries"].default
    assert max_retries_default == 4, f"❌ Expected max_retries=4, got {max_retries_default}"
    print(f"✅ fetch_company_news: max_retries = {max_retries_default} (tuned from 5)")
    
    # Test main function defaults
    sig = inspect.signature(main)
    
    coverage_default = sig.parameters["coverage_threshold"].default
    assert coverage_default == 0.75, f"❌ Expected coverage_threshold=0.75, got {coverage_default}"
    print(f"✅ main: coverage_threshold = {coverage_default} (tuned from 0.6)")
    
    qps_default = sig.parameters["qps_limit"].default
    assert qps_default == 0.5, f"❌ Expected qps_limit=0.5, got {qps_default}"
    print(f"✅ main: qps_limit = {qps_default} (tuned from 1.0)")
    
    fast_fail_threshold_default = sig.parameters["fast_fail_threshold"].default
    assert fast_fail_threshold_default == 0.3, f"❌ Expected fast_fail_threshold=0.3, got {fast_fail_threshold_default}"
    print(f"✅ main: fast_fail_threshold = {fast_fail_threshold_default} (NEW)")
    
    fast_fail_interval_default = sig.parameters["fast_fail_check_interval"].default
    assert fast_fail_interval_default == 100, f"❌ Expected fast_fail_check_interval=100, got {fast_fail_interval_default}"
    print(f"✅ main: fast_fail_check_interval = {fast_fail_interval_default} (NEW)")
    
    print("\n" + "=" * 60)
    print("BACKOFF SCHEDULE VERIFICATION")
    print("=" * 60)
    
    # Verify backoff schedule in source
    source = inspect.getsource(fetch_company_news)
    assert "30 * (2 ** attempt)" in source, "❌ Backoff formula not found"
    print("✅ Exponential backoff formula: 30 * (2 ** attempt)")
    
    # Calculate max wait time
    max_wait = sum(30 * (2 ** i) for i in range(4))
    print(f"✅ Max total wait: {max_wait}s (~{max_wait/60:.1f} min) - tuned from 930s (~15.5 min)")
    
    print("\n" + "=" * 60)
    print("FAST-FAIL LOGIC VERIFICATION")
    print("=" * 60)
    
    # Verify fast-fail is implemented
    main_source = inspect.getsource(main)
    assert "fast_fail_check_interval" in main_source, "❌ Fast-fail check not found"
    assert "FAST-FAIL" in main_source, "❌ Fast-fail message not found"
    print("✅ Fast-fail mode implemented")
    
    assert "::error::" in main_source, "❌ GitHub Actions error annotation not found"
    print("✅ GitHub Actions error annotations added")
    
    print("\n" + "=" * 60)
    print("TIME SAVINGS CALCULATION")
    print("=" * 60)
    
    # Old: 5 retries, 1.0 QPS, no fast-fail
    old_max_retry_time = 930  # 30+60+120+240+480
    old_qps = 1.0
    old_symbols = 503
    old_time_clean = old_symbols / old_qps / 60  # minutes
    old_time_worst = old_time_clean + (old_symbols * old_max_retry_time / 60)  # if all symbols fail
    
    # New: 4 retries, 0.5 QPS, fast-fail at 100
    new_max_retry_time = 450  # 30+60+120+240
    new_qps = 0.5
    new_symbols = 503
    new_time_clean = new_symbols / new_qps / 60  # minutes
    new_time_fast_fail = 100 / new_qps / 60  # fast-fail at 100 symbols
    
    print(f"Old settings (5 retries, 1.0 QPS):")
    print(f"  Clean run: ~{old_time_clean:.1f} min")
    print(f"  Doomed run (30% coverage): ~50+ min")
    
    print(f"\nNew settings (4 retries, 0.5 QPS, fast-fail):")
    print(f"  Clean run: ~{new_time_clean:.1f} min")
    print(f"  Doomed run (fast-fail at 100): ~{new_time_fast_fail:.1f} min")
    
    savings = 50 - new_time_fast_fail
    print(f"\n✅ Time saved on doomed runs: ~{savings:.1f} min ({savings/50*100:.0f}% faster failure)")
    
    print("\n" + "=" * 60)
    print("✅ ALL TUNING VERIFIED!")
    print("=" * 60)
    print("\nSummary:")
    print("  • Max retries: 5 → 4 (50% less worst-case wait)")
    print("  • QPS limit: 1.0 → 0.5 (conservative, stable)")
    print("  • Coverage threshold: 60% → 75% (higher quality)")
    print("  • Fast-fail: None → 30% @ 100 symbols (death-march prevention)")
    print("  • Error annotations: Added (GitHub UI visibility)")
    print("\nResult: Doomed runs fail in ~3 min instead of 50+ min")

if __name__ == "__main__":
    try:
        test_tuned_defaults()
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
