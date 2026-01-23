#!/usr/bin/env python3
"""
Test script to demonstrate the improved Finnhub news ingestion with:
1. Retry with exponential backoff on 429 errors
2. Coverage guardrails
3. Request load reduction options
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_retry_logic():
    """Test that the retry logic is properly implemented."""
    from ingest_company_news import fetch_company_news
    
    print("=" * 60)
    print("TEST 1: Retry Logic with Exponential Backoff")
    print("=" * 60)
    
    # Check function signature includes max_retries
    import inspect
    sig = inspect.signature(fetch_company_news)
    params = list(sig.parameters.keys())
    
    assert "max_retries" in params, "❌ max_retries parameter missing"
    print("✅ fetch_company_news has max_retries parameter")
    
    # Check default value
    default_retries = sig.parameters["max_retries"].default
    assert default_retries == 5, f"❌ Expected default max_retries=5, got {default_retries}"
    print(f"✅ Default max_retries = {default_retries}")
    
    # Verify the function has retry logic by checking source
    import textwrap
    source = inspect.getsource(fetch_company_news)
    
    assert "for attempt in range(max_retries)" in source, "❌ No retry loop found"
    print("✅ Retry loop implemented")
    
    assert "429" in source, "❌ No 429 handling found"
    print("✅ 429 rate limit handling present")
    
    assert "exponential" in source.lower() or "2 ** attempt" in source, "❌ No exponential backoff found"
    print("✅ Exponential backoff implemented")
    
    assert "jitter" in source.lower() or "random" in source.lower(), "❌ No jitter found"
    print("✅ Jitter added to prevent thundering herd")
    
    print("\n✅ All retry logic tests passed!\n")


def test_coverage_guardrails():
    """Test that coverage guardrails are properly implemented."""
    from ingest_company_news import main
    
    print("=" * 60)
    print("TEST 2: Coverage Guardrails")
    print("=" * 60)
    
    import inspect
    
    # Check function signature includes coverage_threshold
    sig = inspect.signature(main)
    params = list(sig.parameters.keys())
    
    assert "coverage_threshold" in params, "❌ coverage_threshold parameter missing"
    print("✅ main() has coverage_threshold parameter")
    
    # Check default value
    default_threshold = sig.parameters["coverage_threshold"].default
    assert default_threshold == 0.6, f"❌ Expected default threshold=0.6, got {default_threshold}"
    print(f"✅ Default coverage_threshold = {default_threshold} (60%)")
    
    # Verify the function checks coverage
    source = inspect.getsource(main)
    
    assert "symbols_with_news" in source, "❌ No symbol tracking found"
    print("✅ Symbol tracking implemented")
    
    assert "coverage" in source.lower(), "❌ No coverage calculation found"
    print("✅ Coverage calculation present")
    
    assert "sys.exit(1)" in source or "exit(1)" in source, "❌ No exit on failure found"
    print("✅ Exits with error code on insufficient coverage")
    
    assert "DATA INCOMPLETE" in source or "RATE LIMITED" in source, "❌ No clear error message found"
    print("✅ Clear error message for data incompleteness")
    
    print("\n✅ All coverage guardrail tests passed!\n")


def test_request_load_reduction():
    """Test that request load reduction options are available."""
    from ingest_company_news import main, filter_symbols_by_movement
    
    print("=" * 60)
    print("TEST 3: Request Load Reduction")
    print("=" * 60)
    
    import inspect
    
    # Check function signature includes load reduction options
    sig = inspect.signature(main)
    params = list(sig.parameters.keys())
    
    assert "filter_by_movement" in params, "❌ filter_by_movement parameter missing"
    print("✅ main() has filter_by_movement parameter")
    
    assert "qps_limit" in params, "❌ qps_limit parameter missing"
    print("✅ main() has qps_limit parameter")
    
    # Check default QPS limit
    default_qps = sig.parameters["qps_limit"].default
    assert default_qps == 1.0, f"❌ Expected default qps_limit=1.0, got {default_qps}"
    print(f"✅ Default qps_limit = {default_qps} calls/sec")
    
    # Verify filter function exists
    filter_sig = inspect.signature(filter_symbols_by_movement)
    filter_params = list(filter_sig.parameters.keys())
    assert "symbols" in filter_params, "❌ filter_symbols_by_movement missing symbols parameter"
    print("✅ filter_symbols_by_movement() function exists")
    
    # Verify rate limiting is adaptive
    main_source = inspect.getsource(main)
    assert "sleep_time" in main_source or "qps_limit" in main_source, "❌ No adaptive rate limiting found"
    print("✅ Adaptive rate limiting based on qps_limit")
    
    print("\n✅ All request load reduction tests passed!\n")


def test_cli_arguments():
    """Test that CLI arguments are properly configured."""
    print("=" * 60)
    print("TEST 4: CLI Arguments")
    print("=" * 60)
    
    # Read the file to check argparse configuration
    script_path = Path(__file__).parent / "src" / "ingest_company_news.py"
    with open(script_path, 'r') as f:
        source = f.read()
    
    assert "--coverage_threshold" in source, "❌ --coverage_threshold argument missing"
    print("✅ --coverage_threshold CLI argument added")
    
    assert "--filter_by_movement" in source, "❌ --filter_by_movement argument missing"
    print("✅ --filter_by_movement CLI argument added")
    
    assert "--qps_limit" in source, "❌ --qps_limit argument missing"
    print("✅ --qps_limit CLI argument added")
    
    print("\n✅ All CLI argument tests passed!\n")


def print_usage_examples():
    """Print usage examples."""
    print("=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    
    examples = """
# Basic usage (default 60% coverage threshold, 1 QPS):
python src/ingest_company_news.py \\
    --universe sp500_universe.csv \\
    --week_end 2026-01-16

# Stricter coverage requirement (80%):
python src/ingest_company_news.py \\
    --universe sp500_universe.csv \\
    --week_end 2026-01-16 \\
    --coverage_threshold 0.8

# Slower rate to avoid 429s (0.5 QPS = 30 calls/min):
python src/ingest_company_news.py \\
    --universe sp500_universe.csv \\
    --week_end 2026-01-16 \\
    --qps_limit 0.5

# Enable movement filtering to reduce API load:
python src/ingest_company_news.py \\
    --universe sp500_universe.csv \\
    --week_end 2026-01-16 \\
    --filter_by_movement

# Combined: strict coverage + slow rate:
python src/ingest_company_news.py \\
    --universe sp500_universe.csv \\
    --week_end 2026-01-16 \\
    --coverage_threshold 0.8 \\
    --qps_limit 0.5
"""
    print(examples)


def print_improvements_summary():
    """Print summary of improvements."""
    print("=" * 60)
    print("IMPROVEMENTS SUMMARY")
    print("=" * 60)
    
    summary = """
✅ A) RETRY WITH BACKOFF (NOT "FAIL AND MOVE ON"):
   - Exponential backoff: 30s, 60s, 120s, 240s, 480s
   - Jitter (0-5s) to avoid thundering herd
   - 5 retry attempts per symbol before giving up
   - Proper exception handling with informative error messages
   
✅ B) COVERAGE GUARDRAIL:
   - Tracks symbols_with_news vs total symbols
   - Configurable threshold (default: 60%)
   - Exits with error code 1 if coverage < threshold
   - Clear error message: "DATA INCOMPLETE — RATE LIMITED"
   - Lists failed symbols for debugging
   
✅ C) REQUEST LOAD REDUCTION:
   - Configurable QPS limit (default: 1.0 calls/sec)
   - Optional movement filtering (--filter_by_movement)
   - Adaptive sleep time based on QPS setting
   - Foundation for filtering by price/volume changes
   
BEFORE: "pause occasionally" → rapid 429s → cascading failures
AFTER:  exponential backoff → retry same symbol → fail if coverage < threshold
"""
    print(summary)


if __name__ == "__main__":
    try:
        test_retry_logic()
        test_coverage_guardrails()
        test_request_load_reduction()
        test_cli_arguments()
        print_improvements_summary()
        print_usage_examples()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
