#!/usr/bin/env python
"""
Example: Run weekly pipeline with incremental candles and schema v1b
"""
from __future__ import annotations

from pathlib import Path
from datetime import date
import pandas as pd

def example_weekly_run_with_incremental():
    """
    Shows how to integrate:
    1. Incremental candle fetching
    2. Schema-based scoring
    3. Diagnostics
    """
    print("\n" + "=" * 60)
    print("EXAMPLE: Weekly Run with Incremental System")
    print("=" * 60 + "\n")
    
    # Config
    week_end = "2026-01-16"
    schema_id = "news-novelty-v1b"
    universe_path = Path("sp500_universe.csv")
    
    # Step 1: Ensure candles (incremental)
    print("STEP 1: Ensure candles (incremental fetch)")
    print("-" * 60)
    
    from src.ensure_candles import ensure_candles, CandleRequest
    
    # Read universe
    universe = pd.read_csv(universe_path)
    symbols = universe["symbol"].tolist()
    
    # Define what we need
    request = CandleRequest(
        symbols=symbols,
        date_min="2025-10-13",  # 62 trading days lookback
        date_max=week_end,
    )
    
    # Mock fetch function (replace with actual Finnhub)
    def fetch_candles_batch(symbol_list, start_date, end_date):
        """Mock: would call Finnhub API here"""
        print(f"  [MOCK] Fetching {len(symbol_list)} symbols from {start_date} to {end_date}")
        return pd.DataFrame({
            "symbol": symbol_list * 10,  # Mock data
            "date": [date.fromisoformat(week_end)] * (len(symbol_list) * 10),
            "open": [100.0] * (len(symbol_list) * 10),
            "high": [102.0] * (len(symbol_list) * 10),
            "low": [99.0] * (len(symbol_list) * 10),
            "close": [101.0] * (len(symbol_list) * 10),
            "volume": [1000000] * (len(symbol_list) * 10),
        })
    
    candle_stats = ensure_candles(
        store_path=Path("data/derived/market_candles/candles.parquet"),
        req=request,
        fetch_fn=fetch_candles_batch,
        full_refresh=False,  # Incremental
    )
    
    print(f"✓ Candles: {candle_stats['n_existing']} existing + {candle_stats['n_added']} added = {candle_stats['n_final']} total\n")
    
    # Step 2: Load scoring schema
    print("STEP 2: Load scoring schema")
    print("-" * 60)
    
    from src.scoring_schema import load_schema
    
    schema = load_schema(schema_id)
    print(f"✓ Schema: {schema.schema_id} (hash: {schema.content_hash})")
    print(f"  Weights: {schema.get_weights()}")
    print(f"  Skip rules: {schema.get_skip_rules()}\n")
    
    # Step 3: Compute scores (showing key parts)
    print("STEP 3: Score computation (conceptual)")
    print("-" * 60)
    print("  # In actual features_scores.py:")
    print("  weights = schema.get_weights()")
    print("  score = (")
    print("      weights['novelty'] * novelty_z +")
    print("      weights['divergence'] * divergence_z +")
    print("      weights['sentiment'] * sentiment_z +")
    print("      weights['event_intensity'] * event_intensity_z")
    print("  )\n")
    
    # Step 4: Diagnostics
    print("STEP 4: Diagnostics")
    print("-" * 60)
    
    from src.diagnostics import compute_coverage_diagnostics, compute_skip_reasons
    
    # Mock rep_enriched
    rep_enriched = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "cluster_id": ["c1", "c1", "c2"],
        "event_json": [
            '{"event_type_primary": "EARNINGS", "event_confidence": 0.9}',
            '{"event_type_primary": "EARNINGS", "event_confidence": 0.85}',
            '{"event_type_primary": "PRICE_ACTION_RECAP", "event_confidence": 0.7}',
        ],
    })
    
    coverage = compute_coverage_diagnostics(rep_enriched)
    print(f"✓ Coverage: {coverage.n_symbols_with_news} symbols with news")
    print(f"  Price action share: {coverage.price_action_recap_share:.1%}")
    
    week_summary = {
        "event_intensity": 0.85,  # Mock value
        "price_action_recap_share": coverage.price_action_recap_share,
        "high_severity_clusters": coverage.n_clusters,
    }
    
    skip = compute_skip_reasons(week_summary, schema.get_skip_rules())
    if skip["is_skip"]:
        print(f"⚠️  Week should be skipped:")
        for reason in skip["reasons"]:
            print(f"    {reason['code']}: {reason['value']:.2f} vs {reason['threshold']:.2f}")
    else:
        print(f"✓ Week passes skip rules\n")
    
    # Step 5: Write outputs (namespace by schema)
    print("STEP 5: Write outputs")
    print("-" * 60)
    
    from src.scoring_schema import write_schema_provenance
    
    output_dir = Path(f"data/derived/scores_weekly/schema={schema_id}/week_ending={week_end}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write schema provenance
    prov_path = write_schema_provenance(schema, output_dir)
    print(f"✓ Wrote schema provenance to {prov_path}")
    
    # Write metadata
    import json
    meta = {
        "week_end": week_end,
        "schema_id": schema_id,
        "schema_hash": schema.content_hash,
        "candles_ingestion": {
            "n_existing": candle_stats["n_existing"],
            "n_added": candle_stats["n_added"],
            "n_final": candle_stats["n_final"],
        },
        "diagnostics": {
            "coverage": {
                "n_symbols_with_news": coverage.n_symbols_with_news,
                "price_action_recap_share": coverage.price_action_recap_share,
            },
            "skip_decision": {
                "is_skip": skip["is_skip"],
                "n_reasons": len(skip["reasons"]),
            },
        },
    }
    
    meta_path = output_dir / "score_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"✓ Wrote metadata to {meta_path}\n")
    
    print("=" * 60)
    print("✅ Weekly run complete!")
    print("=" * 60)
    print(f"\nOutputs in: {output_dir}")
    print("\nNext steps:")
    print("  1. Review score_meta.json for diagnostics")
    print("  2. Compare with baseline schema (news-novelty-v1)")
    print("  3. Export basket if passed skip rules")
    print()

def example_compare_schemas():
    """Compare v1 vs v1b results"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Compare Schema Results")
    print("=" * 60 + "\n")
    
    week_end = "2026-01-09"
    
    # Check if both exist
    v1_path = Path(f"data/derived/scores_weekly/schema=news-novelty-v1/week_ending={week_end}/scores_weekly.parquet")
    v1b_path = Path(f"data/derived/scores_weekly/schema=news-novelty-v1b/week_ending={week_end}/scores_weekly.parquet")
    
    if not v1_path.exists():
        print(f"⚠️  v1 scores not found at {v1_path}")
        print("   Run: python -m src.rescore_week --week_end 2026-01-09 --schema news-novelty-v1 --offline\n")
        return
    
    if not v1b_path.exists():
        print(f"⚠️  v1b scores not found at {v1b_path}")
        print("   Run: python -m src.rescore_week --week_end 2026-01-09 --schema news-novelty-v1b --offline\n")
        return
    
    v1 = pd.read_parquet(v1_path)
    v1b = pd.read_parquet(v1b_path)
    
    print("v1 (Pure News - 45% novelty):")
    print(v1.nlargest(10, "score")[["symbol", "score", "novelty"]])
    
    print("\nv1b (Divergence-Heavy - 20% novelty, 45% divergence):")
    print(v1b.nlargest(10, "score")[["symbol", "score", "novelty", "divergence"]])
    
    # Compare overlap
    v1_top10 = set(v1.nlargest(10, "score")["symbol"])
    v1b_top10 = set(v1b.nlargest(10, "score")["symbol"])
    overlap = v1_top10 & v1b_top10
    
    print(f"\nTop 10 overlap: {len(overlap)}/10 symbols")
    print(f"  v1 only: {v1_top10 - v1b_top10}")
    print(f"  v1b only: {v1b_top10 - v1_top10}")

def example_offline_rescore():
    """Example: Rescore past weeks with new schema"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Offline Rescore (Batch)")
    print("=" * 60 + "\n")
    
    weeks = ["2025-12-26", "2026-01-02", "2026-01-09", "2026-01-16"]
    schema_id = "news-novelty-v1b"
    
    print(f"Rescoring {len(weeks)} weeks with schema: {schema_id}\n")
    
    for week_end in weeks:
        print(f"Week {week_end}:")
        
        # Check if source data exists
        rep_path = Path(f"data/derived/rep_enriched/week_ending={week_end}/rep_enriched.parquet")
        if not rep_path.exists():
            print(f"  ⚠️  Skipping (no rep_enriched)")
            continue
        
        print(f"  ✓ Found rep_enriched.parquet")
        print(f"  → python -m src.rescore_week --week_end {week_end} --schema {schema_id} --offline")
    
    print("\nTo run batch:")
    print("  for week in 2025-12-26 2026-01-02 2026-01-09 2026-01-16; do")
    print(f"    python -m src.rescore_week --week_end $week --schema {schema_id} --offline")
    print("  done\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "weekly":
            example_weekly_run_with_incremental()
        elif cmd == "compare":
            example_compare_schemas()
        elif cmd == "rescore":
            example_offline_rescore()
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python scripts/example_incremental_integration.py [weekly|compare|rescore]")
    else:
        # Run all examples
        example_weekly_run_with_incremental()
        # example_compare_schemas()
        # example_offline_rescore()
