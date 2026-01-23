# Finnhub Ingestion Tuning Guide

## Default Settings (Tuned for Production)

The defaults have been **tuned to prevent 40+ minute death-marches** on doomed runs:

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `max_retries` | 4 | 30/60/120/240s = ~7.5 min max vs 16 min (was 5 retries) |
| `qps_limit` | 0.5 | 30 calls/min - conservative until you observe clean weeks |
| `coverage_threshold` | 0.75 | Fail if <75% coverage (tighten from initial 60% after stable) |
| `fast_fail_threshold` | 0.3 | Abort if <30% coverage after 100 symbols |
| `fast_fail_check_interval` | 100 | Check early to avoid processing 503 symbols at 30% coverage |

## Why These Defaults?

### 1. Reduced Max Retries (4 vs 5)

**Problem**: With 5 retries and exponential backoff (30/60/120/240/480s), a single symbol can take **~16 minutes** in the worst case:
- Total wait: 30 + 60 + 120 + 240 + 480 = **930 seconds (~15.5 min)** + jitter
- If 100 symbols hit this, that's **26+ hours** of waiting

**Solution**: 4 retries max (30/60/120/240s) = **~7.5 minutes** max per symbol
- Still generous for temporary throttles
- Prevents extreme edge cases
- If Finnhub is throttling that hard, you need to reduce QPS, not retry more

### 2. Conservative QPS Limit (0.5 vs 1.0)

**Problem**: Finnhub free tier is "60 calls/min", but:
- Bursts can trigger rate limiting
- Actual safe sustained rate may be lower
- Starting aggressive (1.0 QPS) leads to cascading 429s

**Solution**: Start at 0.5 QPS (30 calls/min)
- Well under the stated limit
- Observe a few clean weeks
- Gradually increase to 0.8-0.9 QPS if stable

### 3. Higher Coverage Threshold (0.75 vs 0.6)

**Problem**: 60% coverage means **200+ symbols can fail** silently
- Downstream analysis is degraded
- Hard to debug which symbols are missing

**Solution**: 75% coverage requirement
- Start at 60% for first 1-2 weeks (learning period)
- Tighten to 75% once stable
- Forces you to fix root causes instead of accepting poor data

### 4. Fast-Fail Mode (NEW)

**Problem**: Previous behavior would process all 503 symbols even if coverage is dismal
- 150/503 symbols (30% coverage) still takes **~50 minutes** to discover
- Wastes API quota and time

**Solution**: Check coverage after first 100 symbols
- If <30% coverage at symbol 100, **abort immediately**
- Prevents "death-march" scenarios
- Saves 400+ API calls and 40+ minutes

Example fast-fail output:
```
  [100/503] XYZ... ✓ 5 articles

❌ FAST-FAIL: Coverage too low after 100 symbols
   Coverage: 28.0% < fast-fail threshold 30%
   Only 28/100 symbols have news
   Aborting to prevent death-march (remaining: 403 symbols)
```

## Usage Examples

### Basic Usage (New Defaults)
```bash
# Conservative settings - recommended for production
python src/ingest_company_news.py \
  --universe sp500_universe.csv \
  --week_end 2026-01-16

# Equivalent to:
# --qps_limit 0.5 (30 calls/min)
# --coverage_threshold 0.75 (75% required)
# --fast_fail_threshold 0.3 (abort if <30% after 100)
```

### Initial Calibration (First 1-2 Weeks)
```bash
# More lenient to establish baseline
python src/ingest_company_news.py \
  --universe sp500_universe.csv \
  --week_end 2026-01-16 \
  --coverage_threshold 0.6 \
  --qps_limit 0.5
```

### Aggressive (If You Have Paid Tier)
```bash
# Finnhub paid tier: 300 calls/min
python src/ingest_company_news.py \
  --universe sp500_universe.csv \
  --week_end 2026-01-16 \
  --qps_limit 4.5 \
  --coverage_threshold 0.8 \
  --fast_fail_threshold 0.4
```

### Emergency: Already Hitting Rate Limits
```bash
# Ultra-conservative
python src/ingest_company_news.py \
  --universe sp500_universe.csv \
  --week_end 2026-01-16 \
  --qps_limit 0.3 \
  --coverage_threshold 0.5 \
  --fast_fail_threshold 0.2
```

## QPS Limiting: Important Details

### Global QPS Applies to All Requests

Every request attempt (including retries) respects the QPS limit **indirectly**:

1. **Normal operation**: Sleep `1.0 / qps_limit` between symbols
2. **During retry**: Exponential backoff replaces normal sleep
3. **After retry succeeds**: Resume normal QPS pacing

This ensures:
- Total API load stays controlled
- Backoff times don't compound with QPS sleep
- System recovers gracefully after rate limits

### Example Timeline

With `qps_limit=0.5` (2 second sleep between symbols):

```
Symbol 1: Request → Success → Sleep 2s
Symbol 2: Request → 429 → Backoff 30s → Retry → Success → Sleep 2s
Symbol 3: Request → Success → Sleep 2s
```

Notice: The 30s backoff **replaces** the normal 2s sleep during retry.

## Coverage Guardrail: Workflow Integration

### GitHub Actions Integration

The coverage guardrail integrates with GitHub Actions via:

1. **Exit code 1** on failure (stops workflow)
2. **Error annotation** for GitHub UI visibility:
   ```
   ::error::DATA INCOMPLETE - Coverage 57.1% below threshold 75.0%
   ```

### Downstream Steps Won't Run on Failure

The workflow is structured so that if news ingestion fails:
- ❌ Clustering doesn't run
- ❌ Enrichment doesn't run
- ❌ Scoring doesn't run
- ❌ Artifacts aren't uploaded
- ✅ Notification issue is created (has `if: always()`)

This prevents cascading failures from incomplete data.

### Example Failure Notification

GitHub issue will show:
```markdown
# Weekly Scan Failed - 2026-01-16

**Status**: failure

## Error Summary
::error::DATA INCOMPLETE - Coverage 57.1% below threshold 75.0%

❌ DATA INCOMPLETE — RATE LIMITED
   Coverage 57.1% < threshold 75.0%
   Only 287/503 symbols have news data
   Failed symbols: AAPL, MSFT, GOOGL, ...
```

## Tuning Roadmap

### Phase 1: Calibration (Weeks 1-2)
- Use default settings
- Monitor coverage in logs
- Adjust QPS if hitting rate limits
- Keep `coverage_threshold=0.6` (learning)

### Phase 2: Stabilization (Weeks 3-4)
- Increase `coverage_threshold` to 0.75
- Fine-tune `qps_limit` based on observations
- If always clean: try 0.7-0.8 QPS
- If occasional 429s: stay at 0.5 QPS

### Phase 3: Optimization (Ongoing)
- Implement movement filtering (see below)
- Reduce universe to ~100-200 active symbols
- Can increase QPS since total calls are lower
- Can increase coverage_threshold to 0.9+

## Next: Movement Filtering (The Biggest Win)

Once you've stabilized the current setup, implement movement filtering to dramatically reduce API load.

### Implementation Plan

1. **Use candle data to score symbols**:
   ```python
   def score_movement(symbol, candles_df, week_end_date):
       # Get last week's data
       week_data = candles_df[
           (candles_df['symbol'] == symbol) &
           (candles_df['date'] >= week_end_date - timedelta(days=7)) &
           (candles_df['date'] <= week_end_date)
       ]
       
       # 1-week return z-score
       returns = (week_data['c'].iloc[-1] / week_data['c'].iloc[0] - 1)
       return_zscore = abs(returns - mean_return) / std_return
       
       # Volume z-score
       vol_ratio = week_data['v'].mean() / hist_volume_avg
       vol_zscore = abs(vol_ratio - 1.0) / std_vol_ratio
       
       # Gap/ATR
       gap = abs(week_data['o'].iloc[0] - week_data['c'].shift(1).iloc[0])
       atr = calculate_atr(week_data)
       gap_score = gap / atr if atr > 0 else 0
       
       # Combined movement score
       return 0.4 * return_zscore + 0.4 * vol_zscore + 0.2 * gap_score
   ```

2. **Filter to top N symbols**:
   ```python
   # Score all symbols
   scored = [(sym, score_movement(sym, candles, week_end)) for sym in universe]
   scored.sort(key=lambda x: x[1], reverse=True)
   
   # Keep top 100-200
   candidate_symbols = [sym for sym, score in scored[:150]]
   ```

3. **Update coverage calculation**:
   ```python
   # Coverage now based on N candidates, not full 503
   coverage = len(symbols_with_news) / len(candidate_symbols)
   ```

### Expected Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| API calls | 503 | ~150 | **70% reduction** |
| Runtime (0.5 QPS) | ~17 min | ~5 min | **71% faster** |
| Cost per week | High | Low | **70% cheaper** |
| Coverage quality | Diluted | Focused | **Better signal** |

## Monitoring & Alerts

Add these checks to your workflow:

### 1. Coverage Trends
Track coverage over time:
```bash
# In notification issue
Coverage history:
- 2026-01-09: 96.8% ✓
- 2026-01-16: 94.2% ✓
- 2026-01-23: 57.1% ✗ (FAILED)
```

### 2. Retry Frequency
Count retry attempts:
```bash
Total requests: 503
Retry attempts: 47 (9.3%)
Fast-fails: 0
```

### 3. QPS Violations
Detect if you're hitting rate limits frequently:
```bash
429 errors: 12 (2.4% of requests)
→ Consider reducing qps_limit to 0.4
```

## Troubleshooting

### Scenario: Coverage Always <75%

**Diagnosis**: Your QPS is still too high, or Finnhub is throttling hard

**Solution**:
1. Reduce `qps_limit` to 0.3 or 0.4
2. Temporarily lower `coverage_threshold` to 0.6
3. Check if you're on the right Finnhub tier

### Scenario: Fast-Fail Triggers Too Often

**Diagnosis**: Early symbols aren't representative, or system is overloaded

**Solution**:
1. Increase `fast_fail_check_interval` to 150
2. Lower `fast_fail_threshold` to 0.2
3. Check if specific symbol patterns are failing

### Scenario: Runs Take Too Long

**Diagnosis**: QPS is too conservative, or universe is too large

**Solution**:
1. If coverage is always >90%, increase `qps_limit` to 0.7
2. Implement movement filtering to reduce universe
3. Consider parallel requests with semaphore (advanced)

## Summary

The tuned defaults prioritize:
- ⚡ **Fast failure** over slow discovery (fast-fail mode)
- 🎯 **Quality** over quantity (75% coverage threshold)
- 🛡️ **Stability** over speed (0.5 QPS conservative limit)
- ⏱️ **Bounded retries** to prevent hour-long waits (4 retries max)

This ensures your runs either **succeed cleanly in ~17 min** or **fail fast in <10 min**, never burning 40+ minutes on a doomed run.
