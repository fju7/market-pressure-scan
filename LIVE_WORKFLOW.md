# Live Trading Workflow

This document describes the automated weekly operating routine for live signal testing.

## Quick Start

### One-time setup
```bash
python -m src.init_live_ledger
```

This creates three ledger files:
- `data/live/trades_log.csv` - Executed trades
- `data/live/weekly_pnl.csv` - Weekly returns
- `data/live/weeks_log.csv` - All week decisions (TRADE or SKIP)

## Weekly Cycle

### Friday (or Weekend) - Generate Signals

```bash
python -m src.weekly_workflow --week_end 2026-01-16 --mode friday
```

This will:
1. ✅ Run complete pipeline (news → clustering → scoring → report)
2. ✅ Export basket with skip-week logic
3. ✅ Log decision to `weeks_log.csv`
4. ✅ Display next steps

**Review:**
- Report: `data/derived/reports/week_ending=YYYY-MM-DD/weekly_report.md`
- Basket: `data/derived/baskets/week_ending=YYYY-MM-DD/basket.csv`
- Metadata: `data/derived/reports/week_ending=YYYY-MM-DD/report_meta.json`

**Decision:**
- If `action=SKIP`: Do nothing, week is logged
- If `action=TRADE`: Proceed to Monday execution

---

### Monday - Execute Trades (if TRADE)

```bash
python -m src.weekly_workflow \
  --week_end 2026-01-16 \
  --mode monday \
  --execution_date 2026-01-20 \
  --account_value 100000
```

This will:
1. ✅ Fetch live prices from Finnhub
2. ✅ Calculate position sizes based on weights
3. ✅ Log trades to `trades_log.csv`

**Manual steps:**
- Execute trades in your brokerage
- If fill prices differ from logged prices, manually update `trades_log.csv`

---

### Friday Close - Record P&L

**For both TRADE and SKIP weeks:**
```bash
python -m src.weekly_workflow \
  --week_end 2026-01-16 \
  --mode friday_close
```

This will:
1. ✅ For TRADE weeks: Calculate returns using Monday open → Friday close from candles
2. ✅ For SKIP weeks: Log SKIP row with zeros and skip reason
3. ✅ Calculate SPY benchmark returns
4. ✅ Log to `weekly_pnl.csv`
5. ✅ Generate performance scoreboard (instant trajectory view)

**No manual input needed** - all prices pulled from `candles_daily.parquet`

**Quick scoreboard view:**
```bash
cat data/live/scoreboard.csv
cat data/derived/scoreboards/latest_scoreboard.md
```

---

## Quick Performance Check

**10-second trajectory view:**
```bash
python -m src.scoreboard
```

Shows:
- Total weeks (TRADE/SKIP split)
- Cumulative returns (basket vs SPY)
- Hit rate and max drawdown
- Time in market %

**Example output:**
```
📊 PERFORMANCE SCOREBOARD
══════════════════════════════════════════════════════════════════════

📅 Weeks: 5 total (3 TRADE, 2 SKIP)
⏱️  Time in Market: 60.0%

💰 Returns:
   Basket:    +2.91%
   SPY:       +3.21%
   Active:    -0.30%

📈 Performance:
   Hit Rate:        66.7%
   Avg Active/Week: +0.32%
   Max Drawdown:    -0.60%
```

---

## Performance Analysis

After 8-12 weeks, analyze results:

```bash
python -m src.analyze_performance
```

**Metrics tracked:**
- Hit rate (positive vs negative weeks)
- Average return per trade
- Active return vs SPY
- Sharpe ratio & Information ratio
- Maximum drawdown
- % of weeks skipped (quality filter effectiveness)
- Signal quality correlation (recap % vs returns)

## Manual Alternative

If you prefer step-by-step control:

### Friday
```bash
python -m src.run_weekly_pipeline --week_end YYYY-MM-DD --max_clusters_per_symbol 1 --skip_backtest
python -m src.log_week_decision --week_end YYYY-MM-DD
```

### Monday (if TRADE)
```bash
python -m src.log_trades \
  --week_end YYYY-MM-DD \
  --execution_date YYYY-MM-DD \
  --account_value XXXXX
```

### Friday Close
```bash
python -m src.update_weekly_pnl --week_end YYYY-MM-DD
python -m src.scoreboard  # Optional: regenerate scoreboard
```

## Data Files

### Input (from pipeline)
- `basket.csv` - Trading signals (TRADE or SKIP)
- `report_meta.json` - Signal quality metadata
- `candles_daily.parquet` - Price data

### Output (ledgers)
- `trades_log.csv` - All executed trades (optional, for real fills)
- `weekly_pnl.csv` - Position-level returns (auto-calculated)
- `weeks_log.csv` - All week decisions + quality metrics
- `scoreboard.csv` - Single-row performance summary
- `latest_scoreboard.md` - Formatted performance report

## Key Principles

1. **No tuning** - Do not adjust parameters based on recent performance
2. **No second-guessing** - Trust the SKIP logic
3. **Consistent execution** - Same routine every week
4. **Track everything** - Even SKIPs matter for analysis
5. **Let it run** - Need 8-12 weeks minimum for meaningful results

## What We're Testing

- Does novelty-based event filtering provide edge?
- Is the SKIP logic effective at avoiding low-quality weeks?
- Can we beat SPY after accounting for turnover/friction?
- What's the real drawdown distribution?

**Success criteria** (after 12 weeks):
- Positive active return vs SPY
- Hit rate > 50%
- Information ratio > 0.5
- Skip rate appropriate (not too high/low)
