# ERROR Action Handling Guide

## Overview

The `weeks_log.csv` schema now supports three action types:
- **TRADE**: Normal trading week with basket execution
- **SKIP**: Week skipped due to skip rules (low info, etc.)
- **ERROR**: Pipeline failure after scores generation

## ERROR Action Purpose

The ERROR action prevents **silent week loss** when:
1. Scores are successfully generated
2. Downstream steps fail (report_weekly, export_basket, etc.)
3. Week would otherwise disappear from weeks_log.csv

Without ERROR logging, failed weeks vanish from tracking, creating gaps in historical data and making debugging difficult.

## Implementation

### Where ERROR is Logged

**File**: `src/run_weekly_pipeline.py`
**Function**: `log_error_week()`

```python
# Wrapped around report_weekly step
try:
    sh([py, "-m", "src.report_weekly", ...])
except subprocess.CalledProcessError as e:
    scores_path = Path(f"data/derived/scores_weekly/regime={regime}/week_ending={week_end}/scores_weekly.parquet")
    if scores_path.exists():
        log_error_week(week_end, "ERROR_POST_SCORES", str(e))
    raise
```

**Behavior**:
- Logs ERROR to weeks_log.csv
- Still fails the pipeline (re-raises exception)
- Provides visibility into what failed

### ERROR Week Structure

```csv
week_ending_date,action,basket_size,overlap_pct,turnover_pct,...,skip_reason
2026-01-23,ERROR,0,0.0,0.0,...,"ERROR_POST_SCORES | regime=news-novelty-v1 | scores=data/derived/scores_weekly/regime=news-novelty-v1/week_ending=2026-01-23/scores_weekly.parquet | FileNotFoundError: Missing required file: /path/to/clusters.parquet"
```

All numeric fields are zero, `skip_reason` contains structured error info:
- **Error type**: What failed (e.g., ERROR_POST_SCORES)
- **Regime**: Which regime was being processed
- **Scores path**: Path to the scores file that exists (proves scores succeeded)
- **Error class**: Exception type and concise message

This structured format makes debugging trivial - you can see exactly:
- What step failed
- Which regime configuration was used
- Where the successful scores are located
- What specific file/issue caused the failure

## Files Updated for ERROR Support

### ✅ Core Tracking Files

1. **`src/run_weekly_pipeline.py`**
   - Added `log_error_week()` function
   - Wraps report_weekly in try-except
   - Logs ERROR if scores exist but report fails

2. **`src/rebuild_weeks_log.py`**
   - Handles ERROR action in `compute_week_row()`
   - Counts ERROR weeks in summary stats
   - Preserves ERROR during rebuild (if basket exists)

3. **`src/analyze_performance.py`**
   - Counts ERROR weeks separately
   - Excludes ERROR from skip rate calculation
   - Shows ERROR count in summary if any exist

4. **`src/update_weekly_pnl.py`**
   - Logs ERROR weeks to weekly_pnl.csv
   - Similar to SKIP handling but marked as ERROR

### ⚠️ Rebuild Behavior

**IMPORTANT**: `rebuild_weeks_log.py` rebuilds from baskets and reports.

- ERROR weeks in weeks_log.csv **won't have baskets**
- Rebuilding will **drop ERROR weeks** (expected behavior)
- ERROR is a runtime failure marker, not a permanent state
- If you rebuild, investigate and fix ERROR weeks first

**Workflow**:
```bash
# 1. Check for ERROR weeks (now with detailed debugging info)
cat data/live/weeks_log.csv | grep ERROR

# Example output:
# "2026-01-23","ERROR",...,"ERROR_POST_SCORES | regime=news-novelty-v1 | scores=data/derived/scores_weekly/regime=news-novelty-v1/week_ending=2026-01-23/scores_weekly.parquet | FileNotFoundError: Missing required file: clusters.parquet"

# 2. Extract debugging info from skip_reason:
#    - Regime: news-novelty-v1
#    - Scores exist at: data/derived/scores_weekly/regime=news-novelty-v1/week_ending=2026-01-23/scores_weekly.parquet
#    - Issue: clusters.parquet missing

# 3. Check what's missing
ls -la data/derived/news_clusters/week_ending=2026-01-23/

# 4. Fix by running the missing step
python -m src.cluster_news --week_end 2026-01-23

# 5. Retry the failed step
python -m src.report_weekly --week_end 2026-01-23 --regime news-novelty-v1

# 6. Once fixed, rebuild to clean up
python -m src.rebuild_weeks_log
```

## Backward Compatibility

### Legacy Week Data

Old weeks (e.g., 2025-11-27) may have **incomplete data**:
- Missing `clusters.parquet`
- Missing `rep_enriched.parquet`
- Scores exist in non-regime paths

**Result**: report_weekly fails with clear error message:
```
Missing required files for week 2025-11-27:
  - /path/to/news_clusters/week_ending=2025-11-27/clusters.parquet
  - /path/to/rep_enriched/week_ending=2025-11-27/rep_enriched.parquet

This may indicate incomplete data (e.g., legacy weeks missing clusters/enriched data).
Either complete the pipeline for this week or exclude it from processing.
```

**Solution**: 
- Complete the pipeline for that week, OR
- Accept that legacy weeks are incomplete and exclude from processing

### Path Fallback

report_weekly.py and export_basket.py now include backward compatibility:
```python
# Try regime-specific path first
scores_p = paths.scores_dir / f"regime={regime}" / f"week_ending={week_end}" / "scores_weekly.parquet"

# Fall back to legacy non-regime path
if not scores_p.exists():
    scores_p_legacy = paths.scores_dir / f"week_ending={week_end}" / "scores_weekly.parquet"
    if scores_p_legacy.exists():
        scores_p = scores_p_legacy
```

## Testing ERROR Handling

### Simulate an ERROR

```python
# Temporarily break report_weekly to test ERROR logging
# Run pipeline - it will fail but log ERROR
python -m src.run_weekly_pipeline --week_end 2026-01-23 --regime news-novelty-v1

# Verify ERROR was logged
tail data/live/weeks_log.csv
```

### Verify ERROR Counts

```python
python -m src.analyze_performance
# Should show: "ERROR weeks: 1 (excluded from skip rate)"
```

### Clean Up Test ERROR

```bash
# Fix the issue
python -m src.report_weekly --week_end 2026-01-23 --regime news-novelty-v1

# Rebuild to remove ERROR
python -m src.rebuild_weeks_log
```

## Dashboard/Visualization Considerations

Any code that processes `weeks_log.csv` should handle ERROR explicitly:

### ❌ Bad (silently drops ERROR)
```python
trade_weeks = df[df['action'] == 'TRADE']
skip_weeks = df[df['action'] == 'SKIP']
# ERROR weeks disappear!
```

### ✅ Good (explicit handling)
```python
trade_weeks = df[df['action'] == 'TRADE']
skip_weeks = df[df['action'] == 'SKIP']
error_weeks = df[df['action'] == 'ERROR']

if len(error_weeks) > 0:
    print(f"⚠️ {len(error_weeks)} weeks had errors")
```

## Action Schema Summary

| Action | Meaning | basket_size | skip_reason | Frequency |
|--------|---------|-------------|-------------|-----------|
| TRADE | Normal trading week | >0 | "" | ~15-20% |
| SKIP | Skipped by rules | 0 | Rule explanation | ~80-85% |
| ERROR | Pipeline failure | 0 | Error message | Rare (~0%) |

## Maintenance Notes

1. **Monitor ERROR weeks**: Should be rare (~0%)
2. **Investigate immediately**: ERROR indicates pipeline issues
3. **Don't accumulate**: Fix and rebuild to clean up
4. **Check CI/CD logs**: GitHub Actions should flag ERROR weeks

## Related Files

- `src/run_weekly_pipeline.py` - ERROR logging
- `src/rebuild_weeks_log.py` - ERROR rebuild handling
- `src/analyze_performance.py` - ERROR stats reporting
- `src/update_weekly_pnl.py` - ERROR P&L logging
- `data/live/weeks_log.csv` - Master log with ERROR rows
- `WEEKS_LOG_SYSTEM.md` - Original weeks_log documentation
