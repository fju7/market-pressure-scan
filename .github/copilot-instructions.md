


# Copilot Instructions for market-pressure-scan

## Project Architecture & Purpose
- Implements a weekly, news-driven market pressure signal (UPS/DPS) for live and backtest trading.
- All experiment parameters are **locked** in `CONFIG.yaml` for the entire test period (8-12 weeks). **No tuning allowed.**
- Modular, script-based workflow: each major step (ingest, cluster, enrich, score, log, report) is a separate script in `src/`.
- Every automated step has a manual override script for debugging or stepwise control.

## Essential Workflows
- **Weekly pipeline:**
  - Main: `python -m src.run_weekly_pipeline --week_end YYYY-MM-DD --skip_backtest`
  - Orchestrated: `src/weekly_workflow.py` automates the full Friday→Monday→Friday cycle (see `LIVE_WORKFLOW.md`).
- **Live trading routine:**
  - One-time setup: `python -m src.init_live_ledger`
  - Weekly: `python -m src.weekly_workflow --week_end YYYY-MM-DD --mode friday|monday|friday_close`
  - Manual alternatives: `src/log_week_decision.py`, `src/log_trades.py`, `src/update_weekly_pnl.py`
- **Performance analysis:**
  - Quick: `python -m src.scoreboard`
  - Full: `python -m src.analyze_performance`
- **CI/CD:**
  - The `.github/workflows/weekly.yml` workflow automates the full pipeline, artifact upload, and notification issue creation.

## Data Flow & Structure
- **Input:**
  - News: `src/ingest_company_news.py` (Finnhub API, requires `FINNHUB_API_KEY`)
  - Market candles: `src/ingest_market_candles.py`
- **Main pipeline outputs:**
  - Reports: `data/derived/reports/`
  - Baskets: `data/derived/baskets/`
- **Ledgers:**
  - Trades, P&L, week decisions: `data/live/`
  - Scoreboard and reports: `data/derived/`
- **Artifacts:**
  - All logs and reports are CSV or Markdown. See `LIVE_WORKFLOW.md` for file conventions.
  - Artifacts are uploaded by the CI workflow and linked in notification issues.

## Project Conventions & Patterns
- **No tuning:** Never change `CONFIG.yaml` during a test. All changes = new experiment.
- **Skip logic:** Automated skip-week logic is central. Do not override unless manually testing/debugging.
- **Canonical paths:** Use helper functions in `src/derived_paths.py` for artifact locations.
- **Config snapshotting:** Each report logs the config used for reproducibility (see `report_meta.json`).
- **Schemas & regimes:** Scoring schemas are in `configs/scoring_schemas/`, regime configs in `config/regimes/`.

## Integration Points
- **Finnhub API:** Set `FINNHUB_API_KEY` in env for news/price ingestion.
- **OpenAI API:** Set `OPENAI_API_KEY` for news enrichment/classification.
- **External data:** Price data in `candles_daily.parquet`, news in CSV/JSON.

## Examples
- Full weekly pipeline:
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
- Scoring schemas: `configs/scoring_schemas/`
- Regime configs: `config/regimes/`
- CI workflow: `.github/workflows/weekly.yml`

---

**For AI agents:**
- Always follow locked config and experiment protocol. Never change `CONFIG.yaml` during a test.
- Use the documented workflow unless explicitly instructed otherwise.
- Reference `LIVE_WORKFLOW.md` for step-by-step routines and file conventions.
- Prefer automation, but provide manual alternatives for debugging.
- When in doubt, check `EXPERIMENT_PROTOCOL.md` for scientific rationale and drift prevention.
