# Weeks Log System

## Overview

The weeks log system maintains a canonical, sorted, deduplicated ledger of all weekly pipeline runs in `data/live/weeks_log.csv`.

## Components

### 1. `src/log_week_decision.py`

**Purpose:** Log a single week's decision (TRADE or SKIP) to the weeks log.

**Usage:**
```bash
python -m src.log_week_decision --week_end 2026-01-23
```

**Behavior:**
- Reads basket and report metadata for the specified week
- Calculates turnover metrics by comparing with the most recent prior TRADE week
- Appends one row to `data/live/weeks_log.csv`
- Skips if the week is already logged (duplicate protection)
- Uses `QUOTE_ALL` for CSV safety (prevents corruption from special characters)

**Schema:**
```csv
week_ending_date,action,basket_size,overlap_pct,turnover_pct,num_clusters,avg_novelty_z,avg_event_intensity_z,recap_pct,is_low_info,num_positions,logged_at,skip_reason
```

### 2. `src/rebuild_weeks_log.py`

**Purpose:** Rebuild the entire weeks log from scratch by discovering all weeks.

**Usage:**
```bash
python -m src.rebuild_weeks_log
```

**Behavior:**
- Discovers all weeks from `data/derived/reports/week_ending=*`
- Computes each row using the same logic as `log_week_decision`
- Writes a fresh, sorted, deduplicated `weeks_log.csv`
- **Idempotent:** Can be run multiple times safely
- Guarantees:
  - Sorted by `week_ending_date`
  - No duplicate weeks (unique key: `week_ending_date`)
  - Canonical column order
  - Safe CSV quoting (`QUOTE_ALL`)

**When to use:**
- After backfilling historical weeks
- To repair corrupted weeks_log.csv
- To ensure the log matches all available weeks
- In CI/CD to guarantee complete ledger in artifacts

### 3. CI Integration

The rebuild step is automatically run in `.github/workflows/weekly.yml` after each weekly pipeline run:

```yaml
- name: Rebuild weeks log (full)
  run: |
    python -m src.rebuild_weeks_log
```

This ensures:
- Every artifact ZIP contains a complete, canonical weeks_log.csv
- No drift between individual week appends and the complete ledger
- Automatic repair of any corruption or duplicates

## Backfilling Historical Weeks

To backfill missing weeks:

```bash
# 1. Run log_week_decision for each missing week
for W in 2025-12-04 2025-12-18 2025-12-25; do
  python -m src.log_week_decision --week_end "$W"
done

# 2. Rebuild to ensure canonical format
python -m src.rebuild_weeks_log

# 3. Verify the result
wc -l data/live/weeks_log.csv
tail -n 10 data/live/weeks_log.csv
```

Expected: `N+1` lines (N weeks + 1 header)

## Verification

Quick check script:

```python
import pandas as pd

df = pd.read_csv("data/live/weeks_log.csv")

print(f"✅ Weeks log verified")
print(f"   Rows: {len(df)}")
print(f"   Unique weeks: {df['week_ending_date'].nunique()}")
print(f"   Date range: {df['week_ending_date'].min()} to {df['week_ending_date'].max()}")
print(f"\n   Actions:")
print(df["action"].value_counts())

# Check for duplicates
dups = df[df.duplicated(subset=["week_ending_date"], keep=False)]
if len(dups) > 0:
    print(f"\n❌ WARNING: {len(dups)} duplicate week(s)!")
else:
    print(f"\n✅ No duplicates")
```

## Troubleshooting

### File location
- Correct path: `data/live/weeks_log.csv`
- Not: `data/derived/live/weeks_log.csv`

### Duplicate weeks
If you see duplicates:
```bash
python -m src.rebuild_weeks_log
```
This will deduplicate automatically (last write wins).

### Corrupted CSV (newlines, broken quotes)
The normalization logic in `log_week_decision.load_and_normalize_weeks_log()` handles:
- Parse errors (falls back to python engine)
- Missing columns (adds with empty values)
- Broken quoting (rewrites with `QUOTE_ALL`)

Or simply rebuild:
```bash
python -m src.rebuild_weeks_log
```

### Wrong row count
1. Check for hidden weeks:
   ```bash
   ls data/derived/reports/ | grep week_ending
   ```

2. Compare with weeks_log:
   ```bash
   wc -l data/live/weeks_log.csv
   ```

3. Rebuild to sync:
   ```bash
   python -m src.rebuild_weeks_log
   ```

## Design Rationale

### Why both append and rebuild?

- **`log_week_decision`**: Fast, single-week operation for live pipeline
- **`rebuild_weeks_log`**: Complete rebuild for CI/CD artifacts and backfill

This dual approach gives:
- **Speed**: Append is fast for single weeks
- **Reliability**: Rebuild guarantees correctness for artifacts
- **Flexibility**: Easy to backfill or repair

### Why QUOTE_ALL?

Prevents CSV corruption from:
- Newlines in skip_reason text
- Commas in metadata
- Special characters in cluster descriptions

### Why idempotent rebuild?

Allows:
- Safe repeated runs in CI/CD
- Easy recovery from corruption
- Simple backfill workflow (run rebuild at the end)

## Future Enhancements

Potential improvements:

1. **Per-regime ledgers**: Track weeks by regime ID
2. **Version field**: Track when rows were last updated
3. **Checksum**: SHA256 of critical fields for tamper detection
4. **Archive old logs**: Move to `data/derived/archive/`
