# Canonical Entrypoint: `run_weekly_pipeline.py`

## Overview

`src/run_weekly_pipeline.py` is the **canonical entrypoint** for all weekly market pressure scans, both in GitHub Actions and local execution.

## Usage

```bash
# Production usage (GitHub Actions)
python -m src.run_weekly_pipeline \
  --week_end YYYY-MM-DD \
  --regime news-novelty-v1 \
  --max_clusters_per_symbol 1 \
  --skip_backtest

# Development usage
python -m src.run_weekly_pipeline \
  --week_end 2026-01-16 \
  --regime news-novelty-v1b
```

## Parameters

- `--week_end` (required): Week ending date (YYYY-MM-DD format, must be Friday in ET)
- `--regime` (optional): Regime ID for multi-schema scoring (default: `news-novelty-v1`)
- `--universe` (optional): Path to universe CSV (default: `sp500_universe.csv`)
- `--max_clusters_per_symbol` (optional): Cluster limit per symbol (default: 3)
- `--skip_backtest` (optional): Skip backtest step (faster execution)

## Pipeline Steps

The canonical entrypoint orchestrates these 10 steps:

1. **Ingest Market Candles** → `data/derived/market_daily/`
2. **Ingest Company News** → `data/derived/news_raw/week_ending=YYYY-MM-DD/`
3. **Cluster News** → `data/derived/news_clusters/week_ending=YYYY-MM-DD/`
4. **Enrich Reps (OpenAI)** → `data/derived/rep_enriched/week_ending=YYYY-MM-DD/`
5. **Features & Scores** → **Regime-namespaced**:
   - `data/derived/features_weekly/regime={regime}/week_ending=YYYY-MM-DD/`
   - `data/derived/scores_weekly/regime={regime}/week_ending=YYYY-MM-DD/`
6. **Weekly Report** → **Compatibility anchor**:
   - `data/derived/reports/week_ending=YYYY-MM-DD/weekly_report.md`
   - `data/derived/reports/week_ending=YYYY-MM-DD/report_meta.json` ⭐
7. **Export Basket** → `data/derived/baskets/week_ending=YYYY-MM-DD/`
8. **Generate Trader Sheet** → `data/derived/trader_sheets/week_ending=YYYY-MM-DD/`
9. **Update Weekly PnL** → `data/live/weekly_pnl.csv`
10. **Log Week Decision** → `data/live/weeks_log.csv`, `data/live/scoreboard.csv`

## Dual Metadata System

The pipeline produces **two** `report_meta.json` files:

### 1. Compatibility Anchor (Old Format)
**Path**: `data/derived/reports/week_ending=YYYY-MM-DD/report_meta.json`

**Purpose**: Consumed by downstream tools (`trader_sheet.py`, `export_basket.py`, `log_week_decision.py`) that expect non-namespaced paths.

**Contents**:
```json
{
  "week_end": "2026-01-16",
  "top_n": 20,
  "bottom_n": 20,
  "generated_at_utc": "2026-01-16T12:34:56Z",
  "machine": "ubuntu",
  "python_version": "3.11.5"
}
```

**Written by**: `src/report_weekly.py` (step 6)

### 2. Scoring Provenance (New Format)
**Path**: `data/derived/scores_weekly/regime={regime}/week_ending=YYYY-MM-DD/report_meta.json`

**Purpose**: Full provenance tracking for regime-namespaced scoring runs. Enables reproducibility and divergence detection.

**Contents**:
```json
{
  "regime_id": "news-novelty-v1",
  "schema_id": "news-novelty-v1",
  "schema_hash": "b2090d1d93f218c3",
  "git_sha": "facf01ed7c8a9b1234567890abcdef",
  "timestamp_utc": "2026-01-16T12:34:56.789012",
  "machine": "ubuntu",
  "python_version": "3.11.5",
  "week_end": "2026-01-16"
}
```

**Written by**: `src/features_scores.py` (step 5)

## GitHub Actions Integration

### Workflow File
`.github/workflows/weekly.yml`

### Trigger
- **Scheduled**: Fridays at 4:05pm ET (21:05 UTC during EST)
- **Manual**: workflow_dispatch with optional `week_end` and `regime` inputs

### Invocation
```yaml
- name: Run weekly pipeline
  env:
    FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    WEEK_END="${{ steps.week_end.outputs.week_end }}"
    REGIME="${{ github.event.inputs.regime || 'news-novelty-v1' }}"
    python -m src.run_weekly_pipeline \
      --week_end "$WEEK_END" \
      --regime "$REGIME" \
      --max_clusters_per_symbol 1 \
      --skip_backtest
```

### Artifact Packaging
Actions packages **all week artifacts** from multiple paths:

```yaml
- name: Upload week artifacts
  uses: actions/upload-artifact@v4
  with:
    name: week_${{ steps.week_end.outputs.week_end }}_run_${{ github.run_id }}
    path: |
      week_end.txt
      data/live/
      data/derived/baskets/week_ending=${{ steps.week_end.outputs.week_end }}/
      data/derived/reports/week_ending=${{ steps.week_end.outputs.week_end }}/
      data/derived/trader_sheets/week_ending=${{ steps.week_end.outputs.week_end }}/
      data/derived/scores_weekly/week_ending=${{ steps.week_end.outputs.week_end }}/
      data/derived/scores_weekly/regime=*/week_ending=${{ steps.week_end.outputs.week_end }}/
      data/derived/scores_weekly/schema=*/week_ending=${{ steps.week_end.outputs.week_end }}/
      data/derived/features_weekly/week_ending=${{ steps.week_end.outputs.week_end }}/
      data/derived/features_weekly/regime=*/week_ending=${{ steps.week_end.outputs.week_end }}/
```

This captures:
- Legacy non-namespaced paths (`week_ending=*`)
- Regime-namespaced paths (`regime=*/week_ending=*`)
- Schema-namespaced paths (`schema=*/week_ending=*`)
- All live tracking data (`data/live/`)

## Regime System

### What is a Regime?

A **regime** is a versioned scoring configuration with:
- Unique regime ID (e.g., `news-novelty-v1`, `news-novelty-v1b`)
- Schema definition (feature formulas, scoring logic)
- Full provenance tracking (git SHA, schema hash)

### Why Regimes?

1. **Parallel experimentation**: Run multiple scoring schemas side-by-side
2. **Reproducibility**: Every score can be traced to exact code + config
3. **Divergence detection**: Compare schema changes via content hashing
4. **Safe iteration**: Test new formulas without breaking production

### Regime Namespacing

```
data/derived/
├── scores_weekly/
│   ├── regime=news-novelty-v1/
│   │   └── week_ending=2026-01-16/
│   │       ├── scores_weekly.parquet
│   │       └── report_meta.json
│   └── regime=news-novelty-v1b/
│       └── week_ending=2026-01-16/
│           ├── scores_weekly.parquet
│           └── report_meta.json
└── features_weekly/
    ├── regime=news-novelty-v1/
    │   └── week_ending=2026-01-16/
    │       └── features_weekly.parquet
    └── regime=news-novelty-v1b/
        └── week_ending=2026-01-16/
            └── features_weekly.parquet
```

### Shared vs. Namespaced Artifacts

**Shared** (regime-agnostic):
- `news_raw/week_ending=*` (ingested news is the same for all regimes)
- `news_clusters/week_ending=*` (clustering is schema-independent)
- `rep_enriched/week_ending=*` (OpenAI enrichment is schema-independent)

**Namespaced** (regime-specific):
- `features_weekly/regime=*/week_ending=*` (features depend on schema)
- `scores_weekly/regime=*/week_ending=*` (scores depend on schema)

**Compatibility anchors** (non-namespaced for downstream tools):
- `reports/week_ending=*` (trader sheet, basket export, week logging)
- `baskets/week_ending=*`
- `trader_sheets/week_ending=*`

## Backward Compatibility

The canonical entrypoint maintains **full backward compatibility** with existing tools:

1. **`trader_sheet.py`** reads from `reports/week_ending=*/report_meta.json`
2. **`export_basket.py`** reads from `reports/week_ending=*/report_meta.json`
3. **`log_week_decision.py`** reads from `reports/week_ending=*/report_meta.json`

These tools **do not need to be regime-aware**. The compatibility anchor ensures they continue working with the default regime output.

## Local Packaging Script

For local development, use the packaging helper:

```bash
# Package all artifacts for a given week
./scripts/package_week_artifacts.sh 2026-01-16

# Custom output name
./scripts/package_week_artifacts.sh 2026-01-16 my_custom_archive.zip
```

This script finds **all** directories matching `week_ending=YYYY-MM-DD` across the entire `data/` tree and packages them together.

## Migration Path

### Old Way (Deprecated)
```bash
# Manual step-by-step execution
python -m src.ingest_market_candles --lookback_days 30
python -m src.ingest_company_news --week_end 2026-01-16
python -m src.cluster_news --week_end 2026-01-16
python -m src.enrich_reps_openai --week_end 2026-01-16
python -m src.features_scores --week_end 2026-01-16  # No regime!
python -m src.report_weekly --week_end 2026-01-16
# ... more steps
```

### New Way (Canonical)
```bash
# Single command with regime awareness
python -m src.run_weekly_pipeline \
  --week_end 2026-01-16 \
  --regime news-novelty-v1
```

## Benefits

1. **Single source of truth**: One command for all weekly scans
2. **Regime awareness**: Built-in support for multi-schema scoring
3. **Full provenance**: Every run tracked with git SHA + schema hash
4. **GitHub Actions ready**: Direct integration with CI/CD
5. **Backward compatible**: Existing tools work without changes
6. **Artifact completeness**: Actions packages all week data automatically

## See Also

- [Regime System Documentation](REGIME_SYSTEM.md)
- [Provenance Tracking](PROVENANCE_TRACKING.md)
- [Divergence Detection](DIVERGENCE_DETECTION.md)
- [GitHub Actions Workflow](.github/workflows/weekly.yml)
