# Panel Contract & Scoring Diagnostics

## Quick Reference for Testing & Debugging

### 1. Test Panel Contract (Independent of Full Pipeline)

**Purpose**: Validate that `build_news_feature_panel` returns data matching the expected schema.

**Command**:
```bash
python test_panel_contract.py --week_end 2026-01-23
```

**What it checks**:
- ✓ `symbol` is a column (not index)
- ✓ `week_ending_date` is present
- ✓ No unexpected `index` column (unnamed index materialization)
- ✓ All required features exist: `NV_raw`, `NA_raw`, `NS_raw`, `SS_raw`, `EI_raw`, `EC_raw`
- ✓ No duplicate `(symbol, week_ending_date)` pairs
- 📊 Diagnostics: shape, memory, sample data, NaN counts

**Expected output if passing**:
```
✅ ALL CONTRACT CHECKS PASSED
```

**If failing**: The panel construction has a bug. Fix that before debugging scoring.

---

### 2. Test Scoring Logic (Synthetic Fixture)

**Purpose**: Verify scoring logic works correctly on known synthetic inputs.

**Command**:
```bash
python test_scoring_fixture.py
```

**What it checks**:
- ✓ Output has one row per symbol
- ✓ `symbol` column exists
- ✓ `UPS_adj` is finite and numeric
- ✓ Ranking order matches expectations (high input → high score)
- ✓ No unexpected `index` column

**Expected output if passing**:
```
✅ SCORING FIXTURE TEST PASSED
```

---

### 3. Run Full Pipeline with Diagnostics

**Purpose**: Full end-to-end test with real data, including new contract checks and scoring diagnostics.

**Command**:
```bash
python -m src.features_scores --universe sp500_universe.csv --week_end 2026-01-23
```

**New diagnostics printed**:
1. **Panel Contract Validation** (after `build_news_feature_panel`):
   ```
   ✓ build_news_feature_panel output contract validated:
     - Shape: (500, 15)
     - Has 'symbol': True (column)
     - No 'index' column: True
     - Required features present: 6/6
     - Unique (symbol, week) pairs: 500 (no duplicates)
   ```

2. **Scoring Self-Check** (after `compute_scores`):
   ```
   📊 SCORING SELF-CHECK
   ============================================================
   
   NaN% per feature (top 10 worst):
     NV              :   5.2%
     NA              :   5.2%
     ...
   
   Scores DataFrame:
     Rows: 500
     Symbols eligible for scoring: 500
   
   UPS_adj distribution:
     Valid (finite, non-NaN): 495 / 500 (99.0%)
     Min:    -2.1234
     Median:  0.0012
     Max:     2.4567
   
   Top 5 symbols by UPS_adj:
     NVDA  :  2.4567
     AMD   :  2.1234
     ...
   ```

---

### 4. Test with Small Universe (Fast Iteration)

**Purpose**: Test full pipeline with reduced data for faster debugging.

**Setup**:
```bash
# Create small universe
head -11 sp500_universe.csv > tmp_universe.csv  # 10 symbols + header
```

**Command**:
```bash
python -m src.features_scores --universe tmp_universe.csv --week_end 2026-01-23
```

**Benefit**: Runs in seconds instead of minutes, easier to spot issues.

---

### 5. Programmatic Panel Inspection (REPL/Script)

**Purpose**: Manually inspect panel structure without running full pipeline.

**Code**:
```python
import pandas as pd
from src.features_scores import build_news_feature_panel, default_paths, load_universe, parse_week_end

# Setup
paths = default_paths()
universe = load_universe("sp500_universe.csv")
week_end = "2026-01-23"
week_end_et = parse_week_end(week_end)

# Build panel
panel = build_news_feature_panel(
    paths=paths,
    universe=universe,
    week_end_et=week_end_et,
    week_end=week_end,
    lookback_weeks=12,
)

# Inspect
print('Columns:', list(panel.columns))
print('Index names:', panel.index.names)
print('Has symbol:', 'symbol' in panel.columns)
print('Has index col:', 'index' in panel.columns)
print('Shape:', panel.shape)
print('Duplicates:', panel.duplicated(['symbol','week_ending_date']).sum())
print('\nSample:')
print(panel.head())
```

---

## Interpretation Guide

### If Panel Contract Fails

**Symptom**: `test_panel_contract.py` fails or CI shows panel contract error

**Likely causes**:
1. `symbol` dropped during groupby/apply (check `add_roll`)
2. Unnamed index materialized (missing `drop=True` in `reset_index()`)
3. DataFrame reconstruction without including `symbol`
4. Feature computation dropped required columns

**Fix location**: [src/features_scores.py](src/features_scores.py) in `build_news_feature_panel`

---

### If Scoring Fixture Fails

**Symptom**: `test_scoring_fixture.py` fails

**Likely causes**:
1. Merge dropping `symbol` column
2. Division by zero in normalization
3. Missing feature handling creates NaN cascade
4. Incorrect z-score computation

**Fix location**: [src/features_scores.py](src/features_scores.py) in `compute_scores`

---

### If CI Fails But Local Passes

**Symptom**: Local tests pass, CI fails

**Check**:
1. Data differences (missing parquet files in CI)
2. Pandas version differences
3. Cached artifacts in CI
4. Different branch/commit being tested

**Debug**:
```bash
# In CI, add this step before running pipeline:
- name: Debug environment
  run: |
    python -c "import pandas; print('pandas:', pandas.__version__)"
    ls -la data/derived/news_raw/
    git log -1 --oneline
```

---

## Files Modified

- [src/features_scores.py](src/features_scores.py) - Added `validate_feature_panel_contract()` and scoring diagnostics
- [test_panel_contract.py](test_panel_contract.py) - Independent panel validation test
- [test_scoring_fixture.py](test_scoring_fixture.py) - Synthetic data scoring test

---

## Next Steps After Validation Passes

Once all contract checks pass:

1. ✅ Panel structure is correct
2. ✅ Scoring logic works on synthetic data
3. → Now safe to debug business logic (feature weights, thresholds, etc.)
4. → Can trust that schema bugs won't interfere with scoring
