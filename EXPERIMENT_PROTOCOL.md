# Experiment Protocol Lock

This document explains why CONFIG.yaml exists and how it prevents accidental parameter drift during testing.

## The Problem: Parameter Drift

**What is parameter drift?**
- Week 1: You get -2% returns, wonder if basket should be 15 instead of 20
- Week 3: Hit rate is 40%, maybe increase recap threshold to 60%?
- Week 6: Big loss week, consider adding sector caps
- Week 10: Results look better with tweaked thresholds

**Why this is dangerous:**
1. You're **optimizing on noise** (8-12 weeks is too short for statistical significance)
2. You **lose scientific validity** (can't distinguish luck from skill)
3. You create **survivorship bias** (only the "winning" parameters survive)
4. You'll **never know** if the original strategy actually worked

## The Solution: CONFIG.yaml

**Lock parameters BEFORE testing:**
```yaml
experiment:
  name: "news-novelty-v1"
  version: "1.0.0"
  
basket:
  size: 20  # LOCKED - don't change during test
  
signals:
  recap_pct_threshold: 0.50  # LOCKED
```

**Every week, config is logged in report_meta.json:**
```json
{
  "week_ending_date": "2026-01-16",
  "config_snapshot": {
    "experiment_name": "news-novelty-v1",
    "experiment_version": "1.0.0",
    "basket_size": 20,
    "recap_pct_threshold": 0.5,
    ...
  }
}
```

## What CONFIG.yaml Locks

### Data Sources
- Market data provider (Finnhub)
- News feed (Finnhub Company News)
- Embedding model (OpenAI text-embedding-3-small)
- Universe (S&P 500 from sp500_universe.csv)

### Signal Generation
- Max clusters per symbol: **1**
- PRICE_ACTION_RECAP threshold: **50%**
- Minimum clusters for TRADE: **10**
- Neutralize PRICE_ACTION_RECAP: **true**
- Exclude SPY from ranking: **true**

### Basket Construction
- Size: **20 positions**
- Weighting: **equal_weight** (5% each)
- Sector cap: **none** (yet)
- Conviction filter: **none** (yet)

### Skip Logic
- Enabled: **true**
- Skip if >50% PRICE_ACTION_RECAP
- Skip if <10 clusters
- Skip if both novelty and event intensity weak

### Execution
- Entry: **Monday open**
- Exit: **Friday close**
- Benchmark: **SPY**

## How to Use

### During Testing (8-12 weeks)
**DO NOT MODIFY CONFIG.yaml**

If you're tempted to change something, ask:
- Is this because of recent results? → **Don't change**
- Would I have made this change before seeing results? → **Probably still don't change**
- Is this fixing a bug vs tuning a parameter? → **Bug fixes OK, tuning is not**

### Starting a New Experiment

If you want to test different parameters:

1. **Complete current experiment** (full 8-12 weeks)
2. **Analyze results** using src/analyze_performance.py
3. **Document learnings** in a summary
4. **Create new config**:
   ```yaml
   experiment:
     name: "news-novelty-v2"  # NEW VERSION
     version: "2.0.0"
   ```
5. **Start fresh** with new start_date
6. **Compare v1 vs v2** after both complete

### Making Changes (The Right Way)

**Example: You want to test basket_size = 15**

❌ **WRONG:**
```bash
# Edit CONFIG.yaml and change basket_size to 15
# Continue with week 7
```

✅ **RIGHT:**
```bash
# Week 12: Complete news-novelty-v1
python -m src.analyze_performance

# Document v1 results
# Create news-novelty-v2 with basket_size: 15
# Start new 12-week test
```

## Checking for Drift

**Verify config hasn't changed:**
```bash
# Check current config
cat CONFIG.yaml

# Check what was used in week 1
cat data/derived/reports/week_ending=2026-01-16/report_meta.json | grep config_snapshot

# Check what was used in week 8
cat data/derived/reports/week_ending=2026-03-06/report_meta.json | grep config_snapshot
```

**They should be IDENTICAL.**

## Why This Matters

**Scientific integrity:**
- Can you trust your results if you kept adjusting knobs?
- Would you trust a drug trial where dosage changed every week?

**Real-world applicability:**
- If you tune on these 12 weeks, it won't work on the next 12
- You're just curve-fitting to random noise

**Decision confidence:**
- After 12 weeks with locked params, you can say "this strategy works/doesn't work"
- With drift, you can only say "this worked when I got lucky with parameter timing"

## The Golden Rule

> **"If you wouldn't have changed it before seeing the results, don't change it after."**

## Exceptions (Rare)

**OK to change CONFIG.yaml if:**
1. **Bug fix**: Data pipeline was broken (not working as intended)
2. **External change**: Finnhub changed API, forced to adapt
3. **Calculation error**: Discovered math error in scoring (not just "tuning")

**Document all changes:**
- Update version number
- Add comment explaining why
- Consider restarting the experiment

## Bottom Line

CONFIG.yaml is your **insurance policy** against self-deception. Lock it, log it, don't touch it.
