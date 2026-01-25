#!/usr/bin/env python
"""
Test candle store, scoring schemas, and diagnostics.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

def test_candle_store():
    """Test incremental candle storage"""
    print("=" * 60)
    print("TEST: Candle Store (Incremental + Atomic)")
    print("=" * 60)
    
    from src.market_candles_store import (
        CandleStoreConfig,
        upsert_candles,
        load_existing,
        audit_candles,
    )
    
    try:
        # Setup test store
        test_path = Path("data/derived/test_candle_store.parquet")
        cfg = CandleStoreConfig(path=test_path)
        
        # Initial data
        df1 = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "date": pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-02"]).date,
            "open": [150.0, 300.0, 2800.0],
            "high": [152.0, 305.0, 2850.0],
            "low": [149.0, 298.0, 2780.0],
            "close": [151.0, 303.0, 2820.0],
            "volume": [1000000, 500000, 200000],
        })
        
        n_existing, n_added, n_final = upsert_candles(cfg, df1)
        print(f"✓ Initial insert: {n_existing} existing + {n_added} added = {n_final} final")
        assert n_final == 3, "Should have 3 rows"
        
        # Add more data (incremental)
        df2 = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "date": pd.to_datetime(["2026-01-03", "2026-01-03"]).date,
            "open": [151.0, 303.0],
            "high": [153.0, 308.0],
            "low": [150.0, 301.0],
            "close": [152.0, 306.0],
            "volume": [1100000, 550000],
        })
        
        n_existing, n_added, n_final = upsert_candles(cfg, df2)
        print(f"✓ Incremental append: {n_existing} existing + {n_added} added = {n_final} final")
        assert n_final == 5, "Should have 5 rows total"
        
        # Test deduplication (update existing row)
        df3 = pd.DataFrame({
            "symbol": ["AAPL"],
            "date": pd.to_datetime(["2026-01-02"]).date,
            "open": [150.5],  # Different value
            "high": [152.5],
            "low": [149.5],
            "close": [151.5],
            "volume": [1050000],
        })
        
        n_existing, n_added, n_final = upsert_candles(cfg, df3)
        print(f"✓ Deduplication (last wins): {n_final} final rows")
        assert n_final == 5, "Should still have 5 rows (deduped)"
        
        # Verify updated value
        loaded = load_existing(cfg)
        aapl_jan2 = loaded[(loaded["symbol"] == "AAPL") & (loaded["date"] == pd.to_datetime("2026-01-02").date())]
        assert float(aapl_jan2["close"].iloc[0]) == 151.5, "Should have updated close price"
        print(f"✓ Verified deduplication updated value correctly")
        
        # Test audit
        audit = audit_candles(loaded, expected_date_range=("2026-01-02", "2026-01-03"))
        print(f"✓ Audit: {audit['total_rows']} rows, {audit['unique_symbols']} symbols, ok={audit['ok']}")
        
        # Cleanup
        test_path.unlink()
        
        print("\n✅ Candle store: PASS\n")
        return True
    except Exception as e:
        print(f"\n❌ Candle store: FAIL - {e}\n")
        return False

def test_scoring_schemas():
    """Test schema loading and hashing"""
    print("=" * 60)
    print("TEST: Scoring Schemas (Versioned + Hashed)")
    print("=" * 60)
    
    from src.scoring_schema import load_schema, write_schema_provenance
    
    try:
        # Load v1
        v1 = load_schema("news-novelty-v1")
        print(f"✓ Loaded {v1.schema_id} (hash: {v1.content_hash})")
        assert v1.get_weights()["divergence"] == 0.0, "v1 should have no divergence"
        
        # Load v1b
        v1b = load_schema("news-novelty-v1b")
        print(f"✓ Loaded {v1b.schema_id} (hash: {v1b.content_hash})")
        assert v1b.get_weights()["divergence"] == 0.45, "v1b should have 45% divergence"
        
        # Test provenance write
        test_dir = Path("data/derived/test_schema_provenance")
        prov_path = write_schema_provenance(v1b, test_dir)
        print(f"✓ Wrote provenance to {prov_path}")
        assert prov_path.exists(), "Provenance file should exist"
        
        # Verify content
        content = prov_path.read_text()
        assert v1b.content_hash in content, "Hash should be in provenance file"
        assert "news-novelty-v1b" in content, "Schema ID should be in provenance"
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
        print("\n✅ Scoring schemas: PASS\n")
        return True
    except Exception as e:
        print(f"\n❌ Scoring schemas: FAIL - {e}\n")
        return False

def test_diagnostics():
    """Test diagnostic functions"""
    print("=" * 60)
    print("TEST: Diagnostics (Coverage + Skip Reasons)")
    print("=" * 60)
    
    from src.diagnostics import (
        compute_coverage_diagnostics,
        compute_skip_reasons,
        compute_counterfactual_scores,
    )
    
    try:
        # Mock rep_enriched data
        rep_enriched = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "cluster_id": ["c1", "c1", "c2"],
            "event_json": [
                '{"event_type_primary": "PRICE_ACTION_RECAP", "event_confidence": 0.8}',
                '{"event_type_primary": "PRICE_ACTION_RECAP", "event_confidence": 0.7}',
                '{"event_type_primary": "EARNINGS", "event_confidence": 0.9}',
            ],
        })
        
        # Test coverage diagnostics
        coverage = compute_coverage_diagnostics(rep_enriched)
        print(f"✓ Coverage: {coverage.n_symbols_with_news} symbols, {coverage.n_clusters} clusters")
        assert coverage.n_symbols_with_news == 3, "Should have 3 symbols"
        assert coverage.price_action_recap_share > 0.6, "Should have high PA share"
        
        # Test skip reasons
        week_summary = {
            "event_intensity": 0.3,
            "price_action_recap_share": 0.92,
            "high_severity_clusters": 1,
        }
        skip_thresholds = {
            "min_event_intensity": 0.75,
            "max_price_action_recap_share": 0.80,
            "min_high_severity_clusters": 2,
        }
        
        skip_info = compute_skip_reasons(week_summary, skip_thresholds)
        print(f"✓ Skip reasons: is_skip={skip_info['is_skip']}, {len(skip_info['reasons'])} reasons")
        assert skip_info["is_skip"], "Should recommend skip"
        assert len(skip_info["reasons"]) == 3, "Should have 3 skip reasons"
        
        # Test counterfactual
        features = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "event_type_primary": ["PRICE_ACTION_RECAP", "EARNINGS", "EARNINGS"],
            "score": [1.5, 2.3, 1.8],
            "novelty": [0.2, 0.7, 0.6],
        })
        
        cf = compute_counterfactual_scores(features, {"exclude_event_types": ["PRICE_ACTION_RECAP"]})
        print(f"✓ Counterfactual: baseline={cf['baseline_candidates']}, no_filter={cf['counterfactual_no_filter_candidates']}")
        
        print("\n✅ Diagnostics: PASS\n")
        return True
    except Exception as e:
        print(f"\n❌ Diagnostics: FAIL - {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("CANDLE STORE + SCHEMA + DIAGNOSTICS TESTS")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("Candle Store", test_candle_store()))
    results.append(("Scoring Schemas", test_scoring_schemas()))
    results.append(("Diagnostics", test_diagnostics()))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} {name}")
    
    print()
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print("=" * 60)
    
    if failed > 0:
        print("\n❌ Some tests failed")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
