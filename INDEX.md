# News-Novelty Regime System - File Index

## Quick Reference

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| **QUICK_START.md** | Start here - implementation summary | 180 | ✅ |
| **MIGRATION_GUIDE.md** | Step-by-step integration patches | 350 | ✅ |
| **REGIME_SYSTEM.md** | Architecture & full documentation | 280 | ✅ |
| **REGIME_IMPLEMENTATION.md** | Implementation details & examples | 250 | ✅ |

## Source Files

### Core Infrastructure (`src/`)

| File | Purpose | Exports | Status |
|------|---------|---------|--------|
| **src/derived_paths.py** | Regime-aware path helper | `DerivedPaths` | ✅ Tested |
| **src/io_atomic.py** | Atomic Parquet writer | `write_parquet_atomic()` | ✅ Tested |
| **src/regime_config.py** | Config loader | `RegimeConfig`, `load_regime()` | ✅ Tested |
| **src/divergence.py** | Divergence computation | `add_divergence_feature()` | ✅ Tested |

### Configuration (`config/regimes/`)

| File | Regime | Novelty Weight | Divergence Weight | Status |
|------|--------|----------------|-------------------|--------|
| **news-novelty-v1.yaml** | Baseline (locked) | 45% | 0% | ✅ |
| **news-novelty-v1b.yaml** | Divergence-heavy | 20% | 45% | ✅ |

### Scripts

| File | Purpose | Usage | Status |
|------|---------|-------|--------|
| **scripts/rescore_regime.py** | Backfill tool | `--regime v1b --weeks 2026-01-09 ...` | ✅ |
| **scripts/test_regime_system.py** | Validation tests | Run to verify setup | ✅ All pass |

### Patches Applied

| File | Change | Impact | Status |
|------|--------|--------|--------|
| **src/ingest_market_candles.py** | Atomic writes | Prevents Parquet corruption | ✅ |

## Directory Structure

```
market-pressure-scan/
├── QUICK_START.md                    # ⭐ Start here
├── MIGRATION_GUIDE.md                # Integration steps
├── REGIME_SYSTEM.md                  # Full documentation
├── REGIME_IMPLEMENTATION.md          # Implementation details
│
├── config/
│   └── regimes/
│       ├── news-novelty-v1.yaml      # Baseline (locked)
│       └── news-novelty-v1b.yaml     # Divergence-heavy
│
├── src/
│   ├── derived_paths.py              # Path helper
│   ├── io_atomic.py                  # Atomic writes
│   ├── regime_config.py              # Config loader
│   ├── divergence.py                 # Divergence features
│   └── ingest_market_candles.py      # ✏️ Patched for atomic writes
│
├── scripts/
│   ├── rescore_regime.py             # Backfill tool
│   └── test_regime_system.py         # Validation suite
│
└── data/derived/                     # After integration
    ├── news_raw/                     # Shared across regimes
    ├── rep_enriched/                 # Shared across regimes
    ├── market_daily/                 # Shared across regimes
    │
    ├── features_weekly/
    │   ├── regime=news-novelty-v1/   # v1 features
    │   └── regime=news-novelty-v1b/  # v1b features
    │
    ├── scores_weekly/
    │   ├── regime=news-novelty-v1/   # v1 scores
    │   └── regime=news-novelty-v1b/  # v1b scores
    │
    └── baskets/
        ├── regime=news-novelty-v1/   # v1 baskets
        └── regime=news-novelty-v1b/  # v1b baskets
```

## Validation Status

All components tested and validated:

```bash
$ PYTHONPATH=$PWD python scripts/test_regime_system.py

✅ PASS - Regime Config Loading
✅ PASS - Derived Paths Helper
✅ PASS - Atomic Parquet Writes
✅ PASS - Divergence Computation

Passed:  4/4
Failed:  0/4
```

## Integration Checklist

- [x] Infrastructure components created
- [x] Configuration files created
- [x] Validation tests pass
- [x] Documentation complete
- [x] Atomic writes patched
- [ ] **TODO: Integrate into features_scores.py** (see MIGRATION_GUIDE.md)
- [ ] **TODO: Update basket/report/sheet scripts** (use regime paths)
- [ ] **TODO: Update CI workflow** (run both regimes)

## Key Concepts

### Regime Isolation

Raw data (news, enrichment, candles) is shared. Only scoring differs:

```
Raw (shared):
  ├── news_raw/week_ending=2026-01-09/
  ├── rep_enriched/week_ending=2026-01-09/
  └── market_daily/candles_daily.parquet

Scored (regime-specific):
  ├── scores_weekly/regime=v1/week_ending=2026-01-09/
  └── scores_weekly/regime=v1b/week_ending=2026-01-09/
```

### Divergence Formula

```python
divergence = |excess_return_5d| × (1 - normalized_novelty)
```

High when:
- Stock moves significantly vs SPY (`|excess_return_5d|` large)
- News novelty is low (captures market dislocations)

### Safety Guardrails

1. **Atomic writes** - Prevents Parquet corruption
2. **Shadow-only mode** - All regimes default to `evaluation.mode: shadow`
3. **Config-driven** - All parameters in YAML (no magic numbers)

## Usage Examples

### Run with regime
```bash
python -m src.features_scores \
  --universe sp500_universe.csv \
  --week_end 2026-01-09 \
  --regime news-novelty-v1b
```

### Backfill multiple weeks
```bash
python scripts/rescore_regime.py \
  --regime news-novelty-v1b \
  --weeks 2026-01-02 2026-01-09 2026-01-16
```

### Compare regimes
```python
import pandas as pd

v1 = pd.read_parquet("data/derived/scores_weekly/regime=news-novelty-v1/week_ending=2026-01-09/scores_weekly.parquet")
v1b = pd.read_parquet("data/derived/scores_weekly/regime=news-novelty-v1b/week_ending=2026-01-09/scores_weekly.parquet")

print("v1 top 10:")
print(v1.nlargest(10, "UPS_adj")[["symbol", "UPS_adj"]])

print("\nv1b top 10:")
print(v1b.nlargest(10, "UPS_adj")[["symbol", "UPS_adj"]])
```

## Support

- **Architecture**: See [REGIME_SYSTEM.md](REGIME_SYSTEM.md)
- **Integration**: See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **Quick Start**: See [QUICK_START.md](QUICK_START.md)
- **Validation**: Run `scripts/test_regime_system.py`

---

**Status**: ✅ Implementation complete, ready for integration

**Next**: Review [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) to integrate into `src/features_scores.py`
