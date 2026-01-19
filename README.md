# market-pressure-scan

Research repo for a weekly news-driven market pressure signal (UPS/DPS).

## Quick Start

1. **Set API keys:**
   ```bash
   export FINNHUB_API_KEY="your_key"
   export OPENAI_API_KEY="your_key"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Review experiment protocol:**
   ```bash
   cat CONFIG.yaml  # Locked parameters
   cat EXPERIMENT_PROTOCOL.md  # Why parameters are locked
   ```

4. **Run weekly pipeline:**
   ```bash
   python -m src.run_weekly_pipeline --week_end 2026-01-24 --skip_backtest
   ```

5. **Follow weekly workflow:**
   See [LIVE_WORKFLOW.md](LIVE_WORKFLOW.md) for complete trading routine.

## Key Documents

- **[CONFIG.yaml](CONFIG.yaml)** - Locked experiment parameters (DO NOT MODIFY during testing)
- **[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)** - Why config is locked and how to avoid drift
- **[LIVE_WORKFLOW.md](LIVE_WORKFLOW.md)** - Weekly operating routine
- **[LIVE_TEST_PLAYBOOK.md](LIVE_TEST_PLAYBOOK.md)** - Testing playbook

## Experiment Design

**Hypothesis:** Novelty-based event filtering provides edge over SPY  
**Duration:** 8-12 weeks minimum  
**Parameters:** Locked in CONFIG.yaml (v1.0.0)  
**Success Criteria:** Positive active return, >50% hit rate, IR >0.5  

**NO TUNING** - Parameters stay fixed for entire test period.
