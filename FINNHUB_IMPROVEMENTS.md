# Finnhub News Ingestion Improvements

## Summary

Enhanced the Finnhub company news ingestion (`src/ingest_company_news.py`) with retry logic, coverage guardrails, and request load reduction to prevent cascading 429 rate limit failures.

## Problems Addressed

### Before
- **Rapid 429 cascades**: After hitting a rate limit, the script continued firing new requests rapidly, causing more 429s
- **No retry logic**: Failed requests were simply skipped with "Error: ..." messages
- **No data quality checks**: Jobs could complete with incomplete data without warning
- **Fixed rate limiting**: Simple pause every 50 symbols, not adaptive

### After
- **Exponential backoff with jitter**: Retries same symbol with increasing delays (30s, 60s, 120s, 240s, 480s)
- **Coverage guardrails**: Job fails if less than threshold % of symbols have news data
- **Configurable QPS limiting**: Adaptive sleep time based on your API plan
- **Optional movement filtering**: Reduce API load by fetching news only for interesting symbols

## Changes Made

### A) Retry with Exponential Backoff

**Function**: `fetch_company_news()`

```python
# New parameter: max_retries (default: 5)
for attempt in range(max_retries):
    response = requests.get(url, headers=headers, params=params, timeout=30)
    
    # Handle 429 rate limit with exponential backoff + jitter
    if response.status_code == 429:
        base_wait = 30 * (2 ** attempt)  # 30s, 60s, 120s, 240s, 480s
        jitter = random.uniform(0, 5)  # Add 0-5s jitter
        wait_time = base_wait + jitter
        print(f"⚠️  Rate limited (429), retry {attempt+1}/{max_retries}, waiting {wait_time:.1f}s...")
        time.sleep(wait_time)
        continue  # Retry same symbol
```

**Key improvements**:
- Retries the **same symbol** instead of moving to next
- Exponential backoff prevents rapid re-requests
- Jitter prevents "thundering herd" when multiple requests hit limit simultaneously
- Only gives up after 5 failed attempts
- Clear progress messages show retry attempts

### B) Coverage Guardrail

**Function**: `main()`

```python
# Track success/failure
symbols_with_news = set()
failed_symbols = []

for symbol in symbols:
    try:
        df = fetch_company_news(symbol, from_date, to_date, api_key)
        if not df.empty:
            symbols_with_news.add(symbol)
    except Exception as e:
        failed_symbols.append(symbol)

# Check coverage
coverage = len(symbols_with_news) / len(symbols)

if coverage < coverage_threshold:
    print(f"❌ DATA INCOMPLETE — RATE LIMITED")
    print(f"   Coverage {coverage*100:.1f}% < threshold {coverage_threshold*100:.0f}%")
    print(f"   Only {len(symbols_with_news)}/{len(symbols)} symbols have news data")
    sys.exit(1)  # Fail the run
```

**Key improvements**:
- Tracks which symbols succeeded vs failed
- Configurable threshold (default: 60%)
- **Exits with error code 1** if insufficient coverage
- Clear error message: **"DATA INCOMPLETE — RATE LIMITED"**
- Lists failed symbols for debugging

### C) Request Load Reduction

**Three options** (in order of impact):

#### Option 1: Configurable QPS Limit

```bash
# Slow down to 0.5 calls/sec (30 calls/min)
python src/ingest_company_news.py \
    --universe sp500_universe.csv \
    --week_end 2026-01-16 \
    --qps_limit 0.5
```

- Adaptive sleep time based on `qps_limit` parameter
- Default: 1.0 QPS (60 calls/min for Finnhub free tier)
- Safe approach: Start at 0.5 QPS if you're hitting limits

#### Option 2: Movement Filtering (Placeholder)

```bash
# Only fetch news for symbols with significant movement
python src/ingest_company_news.py \
    --universe sp500_universe.csv \
    --week_end 2026-01-16 \
    --filter_by_movement
```

**Function**: `filter_symbols_by_movement()`
- Currently a placeholder that returns all symbols
- Framework ready for implementation:
  - Fetch recent candles for all symbols
  - Filter by price change > 5% OR volume > 1.5x average
  - Only fetch news for filtered symbols

**To implement**:
```python
def filter_symbols_by_movement(symbols, week_end_date, api_key, 
                                vol_threshold=1.5, price_threshold=0.05):
    # 1. Fetch last 2 weeks of candles for all symbols
    # 2. Calculate: price_change = (close - prev_close) / prev_close
    # 3. Calculate: vol_ratio = volume / avg_volume_20d
    # 4. Keep symbols where: abs(price_change) > 5% OR vol_ratio > 1.5x
    # 5. Return filtered list
    pass
```

#### Option 3: Market-Level News (Future)

Instead of per-symbol news:
- Fetch general market news feed once
- Fetch company-specific news only for basket candidates
- Reduces API calls from 500+ to ~50

## Usage Examples

### Basic Usage
```bash
# Default: 60% coverage threshold, 1.0 QPS
python src/ingest_company_news.py \
    --universe sp500_universe.csv \
    --week_end 2026-01-16
```

### Stricter Coverage
```bash
# Require 80% of symbols to have news
python src/ingest_company_news.py \
    --universe sp500_universe.csv \
    --week_end 2026-01-16 \
    --coverage_threshold 0.8
```

### Slower Rate (Safer)
```bash
# Slow down to 0.5 QPS (30 calls/min)
python src/ingest_company_news.py \
    --universe sp500_universe.csv \
    --week_end 2026-01-16 \
    --qps_limit 0.5
```

### Combined Settings
```bash
# Strict coverage + slow rate
python src/ingest_company_news.py \
    --universe sp500_universe.csv \
    --week_end 2026-01-16 \
    --coverage_threshold 0.8 \
    --qps_limit 0.5
```

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--universe` | str | (required) | Path to universe CSV with 'symbol' column |
| `--week_end` | str | (required) | Week ending date YYYY-MM-DD |
| `--coverage_threshold` | float | 0.6 | Minimum fraction of symbols that must have news (0.0-1.0) |
| `--filter_by_movement` | flag | False | Only fetch news for symbols with significant price/volume movement |
| `--qps_limit` | float | 1.0 | Query-per-second rate limit |

## Expected Behavior

### Success Case
```
📰 Fetching news for 503 symbols from 2026-01-10 to 2026-01-16
   Rate limit: 1.0 calls/sec | Coverage threshold: 60%
  [1/503] AAPL... ✓ 12 articles
  [2/503] MSFT... ✓ 8 articles
  ...
  [503/503] ZTS... ✓ 5 articles

📊 Coverage Report:
   Total symbols: 503
   Symbols with news: 487
   Failed symbols: 0
   Coverage: 96.8%

✅ Saved 4,321 news articles to data/derived/news_raw/week_ending=2026-01-16/company_news.parquet
```

### Rate Limit Case (with retry)
```
  [123/503] NVDA... ⚠️  Rate limited (429), retry 1/5, waiting 32.3s...
  [123/503] NVDA... ⚠️  Rate limited (429), retry 2/5, waiting 64.7s...
  [123/503] NVDA... ✓ 15 articles
```

### Failure Case (insufficient coverage)
```
📊 Coverage Report:
   Total symbols: 503
   Symbols with news: 287
   Failed symbols: 216
   Coverage: 57.1%

❌ DATA INCOMPLETE — RATE LIMITED
   Coverage 57.1% < threshold 60.0%
   Only 287/503 symbols have news data
   Failed symbols: AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, BRK.B, UNH, JNJ ... and 206 more

# Exit code: 1
```

## Testing

Run the test suite to verify all improvements:

```bash
python test_news_ingestion_improvements.py
```

Tests verify:
- ✅ Retry logic with exponential backoff
- ✅ Coverage guardrails
- ✅ Request load reduction options
- ✅ CLI arguments

## Migration Notes

### For Existing Workflows

The script is **backward compatible** with default parameters:

```bash
# Old command (still works)
python src/ingest_company_news.py --universe sp500_universe.csv --week_end 2026-01-16

# Now includes retry + 60% coverage threshold
```

### Recommended Settings

**For Finnhub Free Tier (60 calls/min)**:
```bash
--qps_limit 0.9  # Slightly under limit for safety
--coverage_threshold 0.6  # Allow 40% missing (some symbols may have no news)
```

**For Finnhub Paid Tier (300 calls/min)**:
```bash
--qps_limit 4.5  # 270 calls/min with margin
--coverage_threshold 0.8  # Stricter requirement
```

**If Hitting Rate Limits**:
```bash
--qps_limit 0.5  # Very conservative
--coverage_threshold 0.5  # More lenient
```

## Implementation Details

### Exponential Backoff Formula

```
wait_time = (30 * 2^attempt) + jitter
```

| Attempt | Base Wait | With Jitter (0-5s) | Cumulative Max |
|---------|-----------|-------------------|----------------|
| 1 | 30s | 30-35s | 35s |
| 2 | 60s | 60-65s | 100s |
| 3 | 120s | 120-125s | 225s |
| 4 | 240s | 240-245s | 470s |
| 5 | 480s | 480-485s | 955s (~16min) |

### Why Jitter?

Without jitter, if 100 requests hit rate limit simultaneously:
- All wait exactly 30s
- All retry at the same time
- Hit rate limit again → cascading failure

With jitter (0-5s random):
- Requests spread out over 5-second window
- Natural throttling
- Smoother API usage

## Future Enhancements

1. **Implement movement filtering**:
   - Fetch candles for all symbols
   - Filter by price/volume criteria
   - Reduce API calls by ~70-90%

2. **Add response caching**:
   - Cache news responses for 1 hour
   - Avoid duplicate requests in case of retries

3. **Parallel requests with semaphore**:
   - Use `asyncio` with rate-limiting semaphore
   - Faster ingestion while respecting limits

4. **Adaptive QPS**:
   - Start at configured QPS
   - Slow down if 429s occur
   - Speed up if successful

## Related Files

- [`src/ingest_company_news.py`](src/ingest_company_news.py) - Main implementation
- [`src/ingest_market_candles.py`](src/ingest_market_candles.py) - Similar pattern for candle ingestion
- [`test_news_ingestion_improvements.py`](test_news_ingestion_improvements.py) - Test suite

## See Also

- [Finnhub API Documentation](https://finnhub.io/docs/api)
- [Exponential Backoff Best Practices](https://cloud.google.com/iot/docs/how-tos/exponential-backoff)
