# Incremental System Implementation Guide

## Overview

Complete implementation of:
1. **Incremental Candle Storage** - Atomic writes, deduplication, append-only
2. **Versioned Scoring Schemas** - Lockable configs with content hashing
3. **Offline Rescoring** - Rebuild scores without network fetch
4. **Diagnostics** - Signal absence vs model blindness detection

## Files Created

### Core Storage (3 files)
- ✅ `src/market_candles_store.py` - Incremental candle store with atomic writes
- ✅ `src/ensure_candles.py` - Weekly pipeline integration
- ✅ `src/diagnostics.py` - Coverage, counterfactuals, skip reasons

### Schema System (3 files)
- ✅ `src/scoring_schema.py` - Schema loader with content hashing
- ✅ `configs/scoring_schemas/news-novelty-v1.yaml` - Baseline schema
- ✅ `configs/scoring_schemas/news-novelty-v1b.yaml` - Divergence-heavy schema

### Offline Rescoring (1 file)
- ✅ `src/rescore_week.py` - Rescore without network fetch

### Testing (1 file)
- ✅ `scripts/test_incremental_system.py` - Comprehensive test suite

## Key Features

### 1. Incremental Candle Appending

**Problem Solved:**
- No more re-fetching entire history every week
- Atomic writes prevent Parquet corruption
- Automatic deduplication on (symbol, date)

**Usage:**
```python
from src.ensure_candles import ensure_candles, CandleRequest

# Define what you need
request = CandleRequest(
    symbols=["AAPL", "MSFT", "GOOGL"],
    date_min="2025-12-01",
    date_max="2026-01-09"
)

# Fetch only missing candles
stats = ensure_candles(
    store_path=Path("data/derived/market_candles/candles.parquet"),
    req=request,
    fetch_fn=your_finnhub_fetch_function,
    full_refresh=False  # Only fetch missing
)

# Stats for report_meta.json
print(f"Added {stats['n_added']} new candles")
print(f"Total store: {stats['n_final']} candles")
```

**Full Refresh (Rare):**
```bash
python -m src.ingest_market_candles \
  --universe sp500_universe.csv \
  --week_end 2026-01-09 \
  --full_refresh
```

### 2. Versioned Scoring Schemas

**Problem Solved:**
- Lock baseline (v1) while experimenting with variants (v1b)
- Content hashing ensures you know exactly what schema was used
- Provenance tracking (schema_used.yaml) prevents confusion

**Schema Structure:**
```yaml
schema_id: news-novelty-v1

skip_rules:
  min_event_intensity: 0.75
  max_price_action_recap_share: 0.80

weights:
  novelty: 0.40
  divergence: 0.00  # v1 = pure news
```

**Usage:**
```python
from src.scoring_schema import load_schema, write_schema_provenance

schema = load_schema("news-novelty-v1b")
print(f"Schema: {schema.schema_id} (hash: {schema.content_hash})")

# Write provenance (exact YAML used)
write_schema_provenance(schema, output_dir)
```

### 3. Offline Rescoring

**Problem Solved:**
- Backtest schema changes without re-ingesting news
- Deterministic replay: same inputs → same outputs
- Full provenance trail (input hashes, output hashes)

**Usage:**
```bash
# Rescore single week
python -m src.rescore_week \
  --week_end 2026-01-09 \
  --schema news-novelty-v1b \
  --offline

# Batch rescore (create wrapper script)
for week in 2026-01-02 2026-01-09 2026-01-16; do
  python -m src.rescore_week \
    --week_end $week \
    --schema news-novelty-v1b \
    --offline
done
```

**Outputs (Schema-Namespaced):**
```
data/derived/scores_weekly/
├── schema=news-novelty-v1/week_ending=2026-01-09/
│   ├── scores_weekly.parquet
│   ├── schema_used.yaml
│   └── rescore_meta.json
└── schema=news-novelty-v1b/week_ending=2026-01-09/
    ├── scores_weekly.parquet
    ├── schema_used.yaml
    └── rescore_meta.json
```

### 4. Diagnostics

**Problem Solved:**
- Distinguish "quiet market" from "over-filtering"
- Structured skip reasons (explorable decisions)
- Counterfactual analysis

**Coverage Diagnostics:**
```python
from src.diagnostics import compute_coverage_diagnostics

coverage = compute_coverage_diagnostics(rep_enriched, clusters)
print(f"Symbols with news: {coverage.n_symbols_with_news}")
print(f"Price action share: {coverage.price_action_recap_share:.1%}")
```

**Skip Reasons:**
```python
from src.diagnostics import compute_skip_reasons

skip = compute_skip_reasons(week_summary, skip_thresholds)
if skip["is_skip"]:
    for reason in skip["reasons"]:
        print(f"{reason['code']}: {reason['value']} vs threshold {reason['threshold']}")
```

**Counterfactual Scoring:**
```python
from src.diagnostics import compute_counterfactual_scores

cf = compute_counterfactual_scores(features, baseline_filters)
print(f"Baseline: {cf['baseline_candidates']} candidates")
print(f"Without filter: {cf['counterfactual_no_filter_candidates']} candidates")

# If counterfactual >> baseline → model blindness
# If both low → signal absence
```

## Integration Checklist

### Phase 1: Candle Store (Priority: High)

- [ ] Update `src/ingest_market_candles.py`:
  ```python
  from src.ensure_candles import ensure_candles, CandleRequest
  
  # Instead of fetch-all:
  stats = ensure_candles(
      store_path=Path("data/derived/market_candles/candles.parquet"),
      req=CandleRequest(symbols, date_min, date_max),
      fetch_fn=fetch_candles_batch,
      full_refresh=args.full_refresh
  )
  ```

- [ ] Add to `report_meta.json`:
  ```json
  "candles_ingestion": {
    "n_existing": 31171,
    "n_added": 2500,
    "n_final": 33671,
    "full_refresh": false,
    "store_path": "data/derived/market_candles/candles.parquet"
  }
  ```

### Phase 2: Schema System (Priority: High)

- [ ] Update `src/features_scores.py`:
  ```python
  from src.scoring_schema import load_schema, write_schema_provenance
  
  # Add --schema argument
  schema = load_schema(args.schema)
  
  # Use schema weights
  weights = schema.get_weights()
  score = (weights["novelty"] * novelty_z +
           weights["divergence"] * divergence_z + ...)
  
  # Write provenance
  write_schema_provenance(schema, output_dir)
  ```

- [ ] Namespace outputs:
  ```
  scores_weekly/schema={schema_id}/week_ending={week_end}/
  ```

### Phase 3: Diagnostics (Priority: Medium)

- [ ] Add to scoring pipeline:
  ```python
  from src.diagnostics import (
      compute_coverage_diagnostics,
      compute_skip_reasons,
      write_diagnostics,
  )
  
  coverage = compute_coverage_diagnostics(rep_enriched, clusters)
  skip = compute_skip_reasons(week_summary, schema.get_skip_rules())
  
  write_diagnostics(week_end, coverage, counterfactual, skip, sensitivity, diag_dir)
  ```

### Phase 4: Offline Rescoring (Priority: Low)

- [ ] Use for backtesting schema changes
- [ ] Create wrapper script for batch rescoring
- [ ] Compare v1 vs v1b performance

## Testing

```bash
# Test all components
PYTHONPATH=$PWD python scripts/test_incremental_system.py
```

Expected output:
```
✅ PASS     Candle Store
✅ PASS     Scoring Schemas
✅ PASS     Diagnostics

Passed: 3/3
```

## Example Workflow

### Weekly Run (Incremental)

```bash
# 1. Ensure candles (only fetches missing)
python -m src.ingest_market_candles \
  --universe sp500_universe.csv \
  --week_end 2026-01-16 \
  --incremental  # New flag

# 2. Score with both schemas
python -m src.features_scores \
  --universe sp500_universe.csv \
  --week_end 2026-01-16 \
  --schema news-novelty-v1

python -m src.features_scores \
  --universe sp500_universe.csv \
  --week_end 2026-01-16 \
  --schema news-novelty-v1b

# 3. Compare results
python - <<'PY'
import pandas as pd

v1 = pd.read_parquet("data/derived/scores_weekly/schema=news-novelty-v1/week_ending=2026-01-16/scores_weekly.parquet")
v1b = pd.read_parquet("data/derived/scores_weekly/schema=news-novelty-v1b/week_ending=2026-01-16/scores_weekly.parquet")

print("v1 top 10:")
print(v1.nlargest(10, "score")[["symbol", "score"]])

print("\nv1b top 10:")
print(v1b.nlargest(10, "score")[["symbol", "score"]])
PY
```

### Backtest Schema (Offline)

```bash
# Rescore past 12 weeks with v1b
for week in $(seq 2025-10-31 7 2026-01-23 | xargs -I{} date -d {} +%Y-%m-%d); do
  python -m src.rescore_week \
    --week_end $week \
    --schema news-novelty-v1b \
    --offline
done

# Compare performance
python scripts/compare_schema_performance.py \
  --schema_a news-novelty-v1 \
  --schema_b news-novelty-v1b \
  --weeks_back 12
```

## Benefits

1. **Efficiency**: Only fetch missing candles (~10x faster)
2. **Safety**: Atomic writes prevent corruption
3. **Flexibility**: Test schema variants without breaking baseline
4. **Provenance**: Content hashing ensures reproducibility
5. **Debuggability**: Diagnostics distinguish signal absence from model blindness

## Next Steps

1. Run test suite to validate
2. Integrate candle store into weekly pipeline
3. Migrate to schema-based scoring
4. Add diagnostics to reports
5. Backtest v1b on historical weeks
