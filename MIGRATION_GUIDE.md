# Migration Guide: Adding Regime Support to features_scores.py

This guide shows the exact changes needed to integrate regime support into your existing `src/features_scores.py`.

## Overview of Changes

1. Add imports (4 new imports)
2. Add `--regime` argument to CLI
3. Load regime config
4. Add divergence computation
5. Make scoring configurable
6. Write to regime-specific paths

## Step-by-Step Patches

### 1. Add Imports (at top of file)

**Find:** (around line 4-17)
```python
import argparse
import hashlib
import inspect
import json
import math
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil import tz
```

**Add after existing imports:**
```python
from src.regime_config import load_regime, RegimeConfig
from src.derived_paths import DerivedPaths
from src.io_atomic import write_parquet_atomic
from src.divergence import add_divergence_feature
```

### 2. Update Paths dataclass (optional but cleaner)

**Find:** (around line 53-67)
```python
@dataclass(frozen=True)
class Paths:
    root: Path
    derived: Path
    news_clusters_dir: Path
    rep_enriched_dir: Path
    market_daily_path: Path
    out_features_dir: Path
    out_scores_dir: Path
```

**Option A:** Keep as-is, add DerivedPaths separately
**Option B:** Simplify to just use DerivedPaths everywhere

For minimal changes, keep existing Paths and use DerivedPaths only for output.

### 3. Update compute_scores signature

**Find:** (around line 812)
```python
def compute_scores(
    panel: pd.DataFrame,
    mkt: pd.DataFrame,
    universe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
```

**Add cfg parameter:**
```python
def compute_scores(
    panel: pd.DataFrame,
    mkt: pd.DataFrame,
    universe: pd.DataFrame,
    cfg: RegimeConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
```

### 4. Add regime-configurable scoring inside compute_scores

**Find:** (around line 854-855)
```python
    # Volatility penalty (v1): if VR_pct > 0.90, shrink to 85%
    df["UPS_adj"] = np.where(df["VR_pct"].fillna(0.0) > 0.90, df["UPS_raw"] * 0.85, df["UPS_raw"])
    df["DPS_adj"] = np.where(df["VR_pct"].fillna(0.0) > 0.90, df["DPS_raw"] * 0.85, df["DPS_raw"])
```

**Add after these lines:**
```python
    # Regime-configurable scoring (v1b adds divergence)
    w = cfg.scoring_weights
    
    # Ensure divergence column exists (will be 0 for v1, non-zero for v1b)
    if "divergence" not in df.columns:
        df["divergence"] = 0.0
    
    # Normalize components to roughly same scale before weighting
    df["NV_norm"] = zscore_series(df["NV_raw"].fillna(0.0))
    df["EVS_norm"] = zscore_series(df["EI_raw"].fillna(0.0))
    df["SS_norm"] = zscore_series(df["SS_raw"].fillna(0.0))
    df["div_norm"] = zscore_series(df["divergence"].fillna(0.0))
    
    # Regime score (replaces or augments UPS_adj)
    df["UPS_regime"] = (
        w.novelty * df["NV_norm"] +
        w.severity * df["EVS_norm"] +
        w.sentiment * df["SS_norm"] +
        w.divergence * df["div_norm"]
    )
    
    # For v1 (div=0), UPS_regime ≈ UPS_adj
    # For v1b (div=0.45), UPS_regime emphasizes divergence
    
    # Use UPS_regime as the primary score (or keep UPS_adj for compatibility)
    df["UPS_final"] = df["UPS_regime"]  # Use this for ranking
```

### 5. Update run() function signature

**Find:** (around line 900)
```python
def run(universe_path: Path, week_end: str, lookback_weeks: int = 12):
```

**Add regime parameter:**
```python
def run(universe_path: Path, week_end: str, lookback_weeks: int = 12, regime: str = "news-novelty-v1"):
```

### 6. Load regime config in run()

**Add at start of run() function:**
```python
def run(universe_path: Path, week_end: str, lookback_weeks: int = 12, regime: str = "news-novelty-v1"):
    # Load regime configuration
    cfg = load_regime(regime)
    dp = DerivedPaths()
    
    print(f"Running with regime: {cfg.name}")
    print(f"  Scoring weights: nov={cfg.scoring_weights.novelty}, div={cfg.scoring_weights.divergence}")
    
    # ... existing code ...
```

### 7. Add divergence computation before scoring

**Find where you call compute_scores** (around line 930-940):
```python
    panel = ...  # your existing panel construction
    
    features, scores = compute_scores(panel, mkt, universe)
```

**Add divergence before scoring:**
```python
    panel = ...  # your existing panel construction
    
    # Add divergence feature if regime uses it
    if cfg.scoring_weights.divergence > 0:
        print("Computing divergence features...")
        mkt_candles = pd.read_parquet(paths.market_daily_path)
        panel = add_divergence_feature(panel, mkt_candles, week_end, novelty_col="NV_raw")
    
    features, scores = compute_scores(panel, mkt, universe, cfg)
```

### 8. Write to regime-specific paths

**Find:** (around line 980-990)
```python
    # Write outputs
    out_feat_dir = paths.out_features_dir / f"week_ending={week_end}"
    out_score_dir = paths.out_scores_dir / f"week_ending={week_end}"
    out_feat_dir.mkdir(parents=True, exist_ok=True)
    out_score_dir.mkdir(parents=True, exist_ok=True)

    feat_path = out_feat_dir / "features_weekly.parquet"
    score_path = out_score_dir / "scores_weekly.parquet"
    features.to_parquet(feat_path, index=False)
    scores.to_parquet(score_path, index=False)
```

**Replace with regime-aware paths and atomic writes:**
```python
    # Write outputs to regime-specific paths (atomic)
    feat_path = dp.file("features_weekly", "features_weekly.parquet", week_end, regime=cfg.name)
    score_path = dp.file("scores_weekly", "scores_weekly.parquet", week_end, regime=cfg.name)
    
    write_parquet_atomic(features, feat_path)
    write_parquet_atomic(scores, score_path)
```

### 9. Update CLI argument parsing

**Find:** (at bottom of file)
```python
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--universe", required=True, help="Path to sp500_universe.csv")
    p.add_argument("--week_end", required=True, help="Week ending Friday (ET) YYYY-MM-DD")
    p.add_argument("--lookback_weeks", default=12, type=int, help="History window in weeks for NV/NA and baselines")
    args = p.parse_args()

    run(Path(args.universe), args.week_end, lookback_weeks=args.lookback_weeks)
```

**Add --regime argument:**
```python
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--universe", required=True, help="Path to sp500_universe.csv")
    p.add_argument("--week_end", required=True, help="Week ending Friday (ET) YYYY-MM-DD")
    p.add_argument("--lookback_weeks", default=12, type=int, help="History window in weeks for NV/NA and baselines")
    p.add_argument("--regime", default="news-novelty-v1", help="Regime name (config/regimes/<name>.yaml)")
    args = p.parse_args()

    run(Path(args.universe), args.week_end, lookback_weeks=args.lookback_weeks, regime=args.regime)
```

## Testing After Changes

### 1. Verify syntax
```bash
python -m py_compile src/features_scores.py
```

### 2. Test v1 (baseline - should match existing behavior)
```bash
python -m src.features_scores \
  --universe sp500_universe.csv \
  --week_end 2026-01-09 \
  --regime news-novelty-v1
```

### 3. Test v1b (divergence-heavy)
```bash
python -m src.features_scores \
  --universe sp500_universe.csv \
  --week_end 2026-01-09 \
  --regime news-novelty-v1b
```

### 4. Verify outputs exist
```bash
# v1 outputs
ls -lh data/derived/scores_weekly/regime=news-novelty-v1/week_ending=2026-01-09/

# v1b outputs
ls -lh data/derived/scores_weekly/regime=news-novelty-v1b/week_ending=2026-01-09/
```

## Rollback Plan

If anything breaks:

1. **Keep backup**: `cp src/features_scores.py src/features_scores.py.backup`
2. **Test incrementally**: Add patches one at a time
3. **Validate**: Run test suite after each change

## Summary of Changes

| Section | Lines Changed | Complexity |
|---------|--------------|------------|
| Imports | +4 lines | Easy |
| compute_scores signature | +1 param | Easy |
| Regime scoring logic | +20 lines | Medium |
| run() signature | +1 param | Easy |
| Divergence computation | +5 lines | Easy |
| Output paths | ~10 lines | Easy |
| CLI args | +1 line | Easy |

**Total**: ~40 lines of new code, ~10 lines modified.

The changes are minimal and backward-compatible (default regime is v1).
