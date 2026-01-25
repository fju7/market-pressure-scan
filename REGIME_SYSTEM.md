# News-Novelty Regime System

## Overview

This system allows multiple scoring regimes to coexist and be evaluated in parallel without re-ingesting or re-enriching news data.

## Architecture

### Data Flow

```
Raw Data (regime-agnostic):
  ├── news_raw/week_ending=YYYY-MM-DD/company_news.parquet
  ├── rep_enriched/week_ending=YYYY-MM-DD/rep_enriched.parquet
  ├── news_clusters/week_ending=YYYY-MM-DD/clusters.parquet
  └── market_daily/candles_daily.parquet

Regime-Specific Outputs:
  ├── features_weekly/regime=<name>/week_ending=YYYY-MM-DD/features_weekly.parquet
  ├── scores_weekly/regime=<name>/week_ending=YYYY-MM-DD/scores_weekly.parquet
  ├── baskets/regime=<name>/week_ending=YYYY-MM-DD/basket.csv
  ├── reports/regime=<name>/week_ending=YYYY-MM-DD/weekly_report.md
  └── trader_sheets/regime=<name>/week_ending=YYYY-MM-DD/trader_sheet.pdf
```

### Key Components

1. **`src/derived_paths.py`** - Path helper with regime support
2. **`src/regime_config.py`** - Config loader for regime YAML files
3. **`src/io_atomic.py`** - Atomic Parquet writes (prevents corruption)
4. **`src/divergence.py`** - Divergence feature computation
5. **`config/regimes/*.yaml`** - Regime definitions

## Regimes

### news-novelty-v1 (Baseline)

**Philosophy**: Conservative, news-driven signal
- High novelty threshold (0.60)
- Requires multiple high-severity clusters (2+)
- Rejects weeks dominated by price action (>70%)
- No divergence component (pure news signal)

**Scoring Weights**:
- Novelty: 45%
- Severity: 35%
- Sentiment: 20%
- Divergence: 0%

### news-novelty-v1b (Divergence-Heavy)

**Philosophy**: Capture market dislocations and price/news divergence
- Looser novelty threshold (0.35)
- Single high-severity cluster sufficient
- More tolerant of price action (>85% to skip)
- **Heavy divergence weighting** (45%)

**Scoring Weights**:
- Novelty: 20%
- Severity: 25%
- Sentiment: 10%
- **Divergence: 45%** ⭐

**Divergence Calculation**:
```python
divergence = |excess_ret_5d| * (1 - normalized_novelty)

where:
  excess_ret_5d = symbol_return_5d - spy_return_5d
  normalized_novelty = novelty scaled to [0, 1]
```

This formula captures:
- Stocks moving significantly vs market (high |excess_ret_5d|)
- With low news novelty (market dislocation, not news-driven)

## Usage

### Run with specific regime

```bash
python -m src.features_scores \
  --universe sp500_universe.csv \
  --week_end 2026-01-09 \
  --regime news-novelty-v1b
```

### Backfill past weeks

```bash
python scripts/rescore_regime.py \
  --regime news-novelty-v1b \
  --weeks 2026-01-02 2026-01-09 2026-01-16
```

### Compare regimes

```bash
# Run both regimes for same week
python -m src.features_scores --universe sp500_universe.csv --week_end 2026-01-09 --regime news-novelty-v1
python -m src.features_scores --universe sp500_universe.csv --week_end 2026-01-09 --regime news-novelty-v1b

# Compare baskets
diff \
  data/derived/baskets/regime=news-novelty-v1/week_ending=2026-01-09/basket.csv \
  data/derived/baskets/regime=news-novelty-v1b/week_ending=2026-01-09/basket.csv
```

## Safety Features

### Atomic Writes

All Parquet writes use `write_parquet_atomic()` which:
1. Writes to temporary file
2. Atomically replaces target (prevents corruption on interruption)
3. Guarantees valid Parquet footer

### Shadow-Only Evaluation

All regimes default to `evaluation.mode: shadow` in config. This is a hard guardrail to prevent accidental live trading with experimental regimes.

To enable a regime check:
```python
from src.regime_config import load_regime

cfg = load_regime("news-novelty-v1b")
assert cfg.evaluation_mode == "shadow", "Only shadow evaluation allowed"
```

## Adding New Regimes

1. Create config file: `config/regimes/my-regime.yaml`
2. Define skip_rules and scoring weights
3. Set `evaluation.mode: shadow`
4. Run backfill to score historical weeks
5. Compare results vs baseline

Example config:
```yaml
name: my-regime
skip_rules:
  enabled: true
  min_total_novelty: 0.40
  min_high_severity_clusters: 1
  max_price_action_share: 0.80

scoring:
  weights:
    novelty: 0.30
    severity: 0.30
    sentiment: 0.20
    divergence: 0.20

evaluation:
  mode: shadow
```

## Migration Notes

### Existing Data

All existing data in `data/derived/**/week_ending=*` remains unchanged. This is your v1 baseline.

To lock it as news-novelty-v1:
```bash
# Symlink or copy existing data
for week in 2026-01-02 2026-01-09 2026-01-16; do
  # Features
  mkdir -p data/derived/features_weekly/regime=news-novelty-v1/week_ending=$week
  ln -sf ../../../../week_ending=$week/features_weekly.parquet \
    data/derived/features_weekly/regime=news-novelty-v1/week_ending=$week/
  
  # Repeat for scores, baskets, reports, trader_sheets
done
```

### Workflow Integration

To run both regimes in CI:
```yaml
- name: Run v1 (baseline)
  run: python -m src.features_scores --week_end $WEEK_END --regime news-novelty-v1
  
- name: Run v1b (divergence-heavy)
  run: python -m src.features_scores --week_end $WEEK_END --regime news-novelty-v1b
```

## Troubleshooting

### Parquet Corruption

If you see "Parquet magic bytes not found":
1. Delete corrupted file
2. Re-run with atomic writer (already patched in v1b)

### Missing Divergence Column

Divergence requires market_daily/candles_daily.parquet. Ensure:
```bash
python -m src.ingest_market_candles --universe sp500_universe.csv --week_end YYYY-MM-DD
```

### Regime Not Found

Check `config/regimes/<name>.yaml` exists and is valid YAML.

## Performance

- **Raw ingestion**: ~10 min (unchanged, only once per week)
- **Enrichment**: ~5 min (unchanged, only once per week)
- **Regime scoring**: ~30 sec per regime (can run in parallel)

Running 2 regimes adds <1 min to total workflow.
