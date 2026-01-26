# Copilot Instructions for market-pressure-scan

## Project Overview
- This repo implements a weekly, news-driven market pressure signal (UPS/DPS) for live and backtest trading.
- All experiment parameters are locked in `CONFIG.yaml` and must not be changed during the test period.
- The main workflow is automated, with manual override options for step-by-step control.

## Key Workflows
- **Weekly Pipeline:**
  - Run with `python -m src.run_weekly_pipeline --week_end YYYY-MM-DD --skip_backtest`
  - Or use the full workflow in `src/weekly_workflow.py` (see `LIVE_WORKFLOW.md` for details)
- **Live Trading Routine:**
  - One-time setup: `python -m src.init_live_ledger`
  - Weekly cycle: `python -m src.weekly_workflow --week_end YYYY-MM-DD --mode friday|monday|friday_close`
  - Manual steps: Use `src/log_week_decision.py`, `src/log_trades.py`, `src/update_weekly_pnl.py` for granular control
- **Performance Analysis:**
  - Quick view: `python -m src.scoreboard`
  - Full analysis: `python -m src.analyze_performance`

## Data Flow & Structure
- **Input:**
  - News, candles, and signals ingested via `src/ingest_company_news.py`, `src/ingest_market_candles.py`
  - Main pipeline output: `data/derived/reports/`, `data/derived/baskets/`
- **Ledgers:**
  - Trades, P&L, and week decisions logged in `data/live/`
  - Scoreboard and reports in `data/derived/`

## Conventions & Patterns
- **No tuning:** Parameters in `CONFIG.yaml` are fixed for the experiment duration.
- **SKIP logic:** Automated skip-week logic is central; do not override unless manually testing.
- **Consistent file formats:** All logs are CSV or Markdown; see `LIVE_WORKFLOW.md` for file details.
- **Modular scripts:** Each major step (ingest, score, log, report) is a separate script in `src/`.
- **Manual override:** All automated steps have manual script alternatives for debugging or custom runs.

## Integration Points
- **Finnhub API:** Requires `FINNHUB_API_KEY` (set as env var)
- **OpenAI API:** For news enrichment, set `OPENAI_API_KEY`
- **External data:** Price data in `candles_daily.parquet`, news in CSV/JSON formats

## Examples
- Run full weekly pipeline:
  ```bash
  python -m src.run_weekly_pipeline --week_end 2026-01-24 --skip_backtest
  ```
- Manual step-by-step:
  ```bash
  python -m src.run_weekly_pipeline --week_end YYYY-MM-DD --max_clusters_per_symbol 1 --skip_backtest
  python -m src.log_week_decision --week_end YYYY-MM-DD
  python -m src.log_trades --week_end YYYY-MM-DD --execution_date YYYY-MM-DD --account_value XXXXX
  python -m src.update_weekly_pnl --week_end YYYY-MM-DD
  python -m src.scoreboard
  ```

## Key Files & References
- `README.md`, `LIVE_WORKFLOW.md`, `EXPERIMENT_PROTOCOL.md`, `CONFIG.yaml`
- Main scripts: `src/weekly_workflow.py`, `src/run_weekly_pipeline.py`, `src/scoreboard.py`, `src/analyze_performance.py`
- Data: `data/live/`, `data/derived/`

---

**For AI agents:**
- Always follow locked config and experiment protocol.
- Use the documented workflow unless explicitly instructed otherwise.
- Reference `LIVE_WORKFLOW.md` for step-by-step routines and file conventions.
- When in doubt, prefer automation but provide manual alternatives for debugging.
