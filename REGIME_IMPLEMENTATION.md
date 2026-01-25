# Regime System Implementation Summary

## ✅ Completed Components

### 1. Core Infrastructure
- ✅ **`src/derived_paths.py`** - Regime-aware path helper
- ✅ **`src/io_atomic.py`** - Atomic Parquet writes (prevents corruption)
- ✅ **`src/regime_config.py`** - YAML config loader with dataclasses
- ✅ **`src/divergence.py`** - Divergence feature computation

### 2. Configuration Files
- ✅ **`config/regimes/news-novelty-v1.yaml`** - Baseline regime (locked)
- ✅ **`config/regimes/news-novelty-v1b.yaml`** - Divergence-heavy regime

### 3. Patches Applied
- ✅ **`src/ingest_market_candles.py`** - Now uses atomic writes

### 4. Scripts
- ✅ **`scripts/rescore_regime.py`** - Backfill tool for rescoring past weeks

### 5. Documentation
- ✅ **`REGIME_SYSTEM.md`** - Complete system documentation

## 🔧 Next Steps (Integration)

To complete the integration, you need to patch `src/features_scores.py`:

### Required Changes

1. **Add imports**:
```python
from src.regime_config import load_regime, RegimeConfig
from src.derived_paths import DerivedPaths
from src.io_atomic import write_parquet_atomic
from src.divergence import add_divergence_feature
```

2. **Add `--regime` argument** to argparse:
```python
p.add_argument("--regime", default="news-novelty-v1", help="Regime name (config/regimes/<name>.yaml)")
```

3. **Load regime config** in main:
```python
cfg = load_regime(args.regime)
dp = DerivedPaths()
```

4. **Add divergence** to features (after panel is built):
```python
# After computing panel features, before scoring
if cfg.scoring_weights.divergence > 0:
    panel = add_divergence_feature(panel, mkt_candles, args.week_end, novelty_col="NV_raw")
```

5. **Make scoring regime-aware**:
```python
def compute_scores_regime(
    panel: pd.DataFrame,
    mkt: pd.DataFrame,
    universe: pd.DataFrame,
    cfg: RegimeConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # ... existing logic ...
    
    # Add regime-configurable final score
    w = cfg.scoring_weights
    df["UPS_regime"] = (
        w.novelty * df["NV"].fillna(0.0) +
        w.severity * df["EVS"].fillna(0.0) +
        w.sentiment * df["SS"].fillna(0.0) +
        w.divergence * df.get("divergence", pd.Series(0.0, index=df.index)).fillna(0.0)
    )
    
    # Keep existing UPS_adj as baseline; add UPS_regime as experimental
    # Or replace UPS_adj = UPS_regime to fully switch
```

6. **Write to regime paths**:
```python
# Instead of:
# out_feat_dir = paths.out_features_dir / f"week_ending={week_end}"

# Use:
feat_path = dp.file("features_weekly", "features_weekly.parquet", args.week_end, regime=cfg.name)
score_path = dp.file("scores_weekly", "scores_weekly.parquet", args.week_end, regime=cfg.name)

write_parquet_atomic(features, feat_path)
write_parquet_atomic(scores, score_path)
```

## 🧪 Testing the System

### 1. Test Regime Loading
```bash
python - <<'PY'
from src.regime_config import load_regime
v1 = load_regime("news-novelty-v1")
v1b = load_regime("news-novelty-v1b")
print(f"v1 divergence weight: {v1.scoring_weights.divergence}")
print(f"v1b divergence weight: {v1b.scoring_weights.divergence}")
PY
```

### 2. Test Divergence Computation
```bash
python - <<'PY'
import pandas as pd
from src.divergence import compute_5d_returns

candles = pd.read_parquet("data/derived/market_daily/candles_daily.parquet")
rets = compute_5d_returns(candles, "2026-01-09")
print(rets.head())
PY
```

### 3. Test Atomic Writes
```bash
python -m src.ingest_market_candles --universe sp500_universe.csv --week_end 2026-01-09
# Verify no corruption
python -c "import pandas as pd; df = pd.read_parquet('data/derived/market_daily/candles_daily.parquet'); print(f'✓ {len(df)} records')"
```

## 📊 Regime Comparison Workflow

Once integrated, compare regimes:

```bash
# Run both regimes for same week
python -m src.features_scores \
  --universe sp500_universe.csv \
  --week_end 2026-01-09 \
  --regime news-novelty-v1

python -m src.features_scores \
  --universe sp500_universe.csv \
  --week_end 2026-01-09 \
  --regime news-novelty-v1b

# Compare scores
python - <<'PY'
import pandas as pd

v1 = pd.read_parquet("data/derived/scores_weekly/regime=news-novelty-v1/week_ending=2026-01-09/scores_weekly.parquet")
v1b = pd.read_parquet("data/derived/scores_weekly/regime=news-novelty-v1b/week_ending=2026-01-09/scores_weekly.parquet")

# Merge and compare
comp = v1[["symbol", "UPS_adj"]].merge(
    v1b[["symbol", "UPS_adj"]], 
    on="symbol", 
    suffixes=("_v1", "_v1b")
)
comp["diff"] = comp["UPS_adj_v1b"] - comp["UPS_adj_v1"]
print("Top divergences:")
print(comp.nlargest(10, "diff")[["symbol", "UPS_adj_v1", "UPS_adj_v1b", "diff"]])
PY
```

## 🔒 Safety Guardrails

All regimes have `evaluation.mode: shadow` by default. To enforce:

```python
from src.regime_config import load_regime

cfg = load_regime(args.regime)
if cfg.evaluation_mode != "shadow":
    raise RuntimeError(f"Only shadow evaluation allowed for regime={cfg.name}")
```

## 🎯 Key Features

1. **No re-ingestion needed** - Raw/enriched data shared across regimes
2. **Atomic writes** - Parquet corruption fixed
3. **Parallel evaluation** - Run v1 and v1b simultaneously
4. **Backfill ready** - Rescore historical weeks with new regimes
5. **Shadow-only** - Hard guardrail against accidental live trading

## 📁 Directory Structure After Implementation

```
data/derived/
├── news_raw/week_ending=2026-01-09/          # Shared
├── rep_enriched/week_ending=2026-01-09/      # Shared
├── news_clusters/week_ending=2026-01-09/     # Shared
├── market_daily/candles_daily.parquet        # Shared
│
├── features_weekly/
│   ├── regime=news-novelty-v1/week_ending=2026-01-09/
│   └── regime=news-novelty-v1b/week_ending=2026-01-09/
│
├── scores_weekly/
│   ├── regime=news-novelty-v1/week_ending=2026-01-09/
│   └── regime=news-novelty-v1b/week_ending=2026-01-09/
│
└── baskets/
    ├── regime=news-novelty-v1/week_ending=2026-01-09/
    └── regime=news-novelty-v1b/week_ending=2026-01-09/
```
