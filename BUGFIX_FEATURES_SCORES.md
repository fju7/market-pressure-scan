# Fix: features_scores.py KeyError and GitHub Actions JS Error

## Issues Fixed

### 1. KeyError: 'symbol' in features_scores.py

**Problem**: 
- `cur_counts.merge(ns[...], on="symbol", how="left")` failed with `KeyError: 'symbol'`
- Root cause: After `groupby("symbol").apply(add_roll)`, the `symbol` was set as the index instead of being a column
- This prevented the merge operation from finding the 'symbol' column

**Fix**: Added `.reset_index(drop=True)` after the `groupby().apply()` call

**Location**: [`src/features_scores.py`](src/features_scores.py#L394)

**Before**:
```python
weekly_counts = weekly_counts.groupby("symbol", group_keys=False).apply(add_roll)

# Extract current week rows
cur_counts = weekly_counts[weekly_counts["week_ending_date"] == week_end].copy()
```

**After**:
```python
weekly_counts = weekly_counts.groupby("symbol", group_keys=False).apply(add_roll).reset_index(drop=True)

# Extract current week rows
cur_counts = weekly_counts[weekly_counts["week_ending_date"] == week_end].copy()

# Ensure symbol is a column (not index) for merging
if "symbol" not in cur_counts.columns and "symbol" in cur_counts.index.names:
    cur_counts = cur_counts.reset_index()
```

**After**:
```python
weekly_counts = weekly_counts.groupby("symbol", group_keys=False).apply(add_roll).reset_index(drop=True)

# Extract current week rows
cur_counts = weekly_counts[weekly_counts["week_ending_date"] == week_end].copy()

# Ensure symbol is a column (not index) for merging - precise fallback
if "symbol" not in cur_counts.columns:
    if "symbol" in cur_counts.index.names:
        cur_counts = cur_counts.reset_index()
    else:
        raise ValueError(f"'symbol' not found in columns or index. Columns: {cur_counts.columns.tolist()}, Index names: {cur_counts.index.names}")
```

**Additional safeguard**: Added detailed diagnostic assertions before the merge:
```python
# Enhanced assertions with full diagnostics for GitHub Actions logs
if "symbol" not in cur_counts.columns:
    raise ValueError(
        f"cur_counts missing 'symbol' column.\n"
        f"  Columns: {cur_counts.columns.tolist()}\n"
        f"  Index names: {cur_counts.index.names}\n"
        f"  Shape: {cur_counts.shape}\n"
        f"  Sample index: {cur_counts.index[:3].tolist() if len(cur_counts) > 0 else 'empty'}"
    )

# Sample join keys for debugging
print(f"  [DEBUG] Merging on 'symbol': cur_counts has {len(cur_counts)} rows, ns has {len(ns)} rows")
if len(cur_counts) > 0:
    print(f"  [DEBUG] cur_counts sample symbols: {cur_counts['symbol'].head(3).tolist()}")
if len(ns) > 0:
    print(f"  [DEBUG] ns sample symbols: {ns['symbol'].head(3).tolist()}")
```

This makes failures immediately actionable from GitHub Actions logs.

### 2. All-NaN Novelty Score Warning

**Problem**:
- `RuntimeWarning: All-NaN slice encountered` when computing `np.nanmedian(ns["NS_raw"])`
- This indicates no valid novelty scores were computed (missing embeddings or insufficient historical data)
- Silent failure - script continues with invalid data

**Fix**: Added comprehensive warning and smart handling

**Location**: [`src/features_scores.py`](src/features_scores.py#L468-L495)

**Before**:
```python
median_ns = float(np.nanmedian(ns["NS_raw"].to_numpy()))
```

**After**:
```python
ns_array = ns["NS_raw"].to_numpy()
total_symbols = len(ns)
nan_count = np.sum(np.isnan(ns_array))
valid_count = total_symbols - nan_count
nan_pct = (nan_count / total_symbols * 100) if total_symbols > 0 else 0.0

# Log novelty score coverage
print(f"  [Novelty] Total symbols: {total_symbols} | Valid NS_raw: {valid_count} ({100-nan_pct:.1f}%) | NaN: {nan_count} ({nan_pct:.1f}%)")

# Detect complete failure vs partial degradation
if np.all(np.isnan(ns_array)):
    # All NaN - likely embeddings pipeline broke or insufficient history
    avg_history = ns["nov_hist_count"].mean() if len(ns) > 0 else 0
    
    # Decide: hard fail if embeddings broke, soft degrade if insufficient history
    if avg_history < 1.0:
        # Insufficient historical data - expected for early weeks
        print("   Reason: Insufficient historical data (expected for early weeks)")
        print("   → Setting median_ns = 0.0 and continuing with degraded novelty scores")
        median_ns = 0.0
    else:
        # Have history but no embeddings - embeddings pipeline likely broke
        print("   Reason: Historical data exists but no embeddings found")
        print("   → This suggests the embeddings pipeline is broken!")
        raise RuntimeError(
            f"Embeddings pipeline failure: {total_symbols} symbols have history (avg={avg_history:.1f}) "
            f"but all NS_raw are NaN. Check embedding generation in enrichment step."
        )
elif nan_pct > 50.0:
    # Partial degradation - warn but continue
    print(f"  ⚠️  WARNING: High NaN rate ({nan_pct:.1f}%) in NS_raw - novelty scores may be degraded")
    median_ns = float(np.nanmedian(ns_array))
else:
    median_ns = float(np.nanmedian(ns_array))
```

This provides:
- **Coverage statistics**: Total, valid count, NaN count and percentage
- **Partial degradation detection**: Warns if >50% NaN but continues
- **Smart failure logic**:
  - **Soft degrade** (median_ns=0.0): All NaN with avg_history < 1.0 (early weeks - expected)
  - **Hard fail** (RuntimeError): All NaN with avg_history >= 1.0 (embeddings broke - critical)
- **Rationale**: Don't trade on broken embeddings, but allow graceful degradation for early weeks

### 3. GitHub Actions JavaScript SyntaxError

**Problem**:
- `SyntaxError: Identifier 'fs' has already been declared`
- In the "Create notification issue" step, `const fs = require('fs')` was declared twice (lines 119 and 124)

**Fix**: Removed the duplicate declaration

**Location**: [`.github/workflows/weekly.yml`](../.github/workflows/weekly.yml#L119-L124)

**Before**:
```yaml
script: |
  const fs = require('fs');
  const week_end = fs.readFileSync('week_end.txt', 'utf8').trim();
  const status = '${{ job.status }}';
  const runUrl = `${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}`;
  
  const fs = require('fs');  // ❌ Duplicate!
  
  // Read scoreboard
  ...
```

**After**:
```yaml
script: |
  const fs = require('fs');
  const week_end = fs.readFileSync('week_end.txt', 'utf8').trim();
  const status = '${{ job.status }}';
  const runUrl = `${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}`;
  
  // Read scoreboard  // ✅ No duplicate
  ...
```

## Testing Recommendations

### Test the merge fix
```bash
# Run features_scores on a recent week
python -m src.features_scores \
  --universe sp500_universe.csv \
  --week_end 2026-01-16 \
  --lookback_weeks 12
```

Expected output:
- No `KeyError: 'symbol'`
- Clear assertions if columns are missing
- Warning message if all NS_raw are NaN

### Test GitHub Actions
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('.github/workflows/weekly.yml'))"
```

Or trigger the workflow and check the notification step completes without JavaScript errors.

## Root Cause Analysis

### Why did groupby().apply() cause this?

When using `.apply()` with a grouped DataFrame:
- Even with `group_keys=False`, pandas may set the grouping column(s) as the index in the result
- This depends on what the applied function returns
- The `add_roll()` function returns a DataFrame with modified columns but doesn't explicitly manage the index
- Result: `symbol` becomes an index instead of a column

### Why not caught earlier?

The code worked in initial testing because:
- Small test datasets may have had different pandas behavior
- The error only manifests when `cur_counts` actually has data
- The merge happens later in the pipeline, not immediately after the groupby

### Prevention

Added two layers of defense:
1. **Explicit `.reset_index(drop=True)`** after groupby().apply()
2. **Fallback check** before merge: if symbol is in index but not columns, reset it
3. **Assertions** that clearly identify the issue if it recurs

## Files Modified

1. [`src/features_scores.py`](src/features_scores.py)
   - Line ~394: Added `.reset_index(drop=True)` after groupby
   - Line ~398: Added fallback index check
   - Line ~468: Added All-NaN warning for NS_raw
   - Line ~544: Added assertions before merge

2. [`.github/workflows/weekly.yml`](../.github/workflows/weekly.yml)
   - Line ~124: Removed duplicate `const fs` declaration

## Related Issues

- The All-NaN novelty score issue should be investigated separately:
  - Check if embeddings are being generated correctly
  - Verify historical data exists for the lookback period
  - Review embedding vector quality
