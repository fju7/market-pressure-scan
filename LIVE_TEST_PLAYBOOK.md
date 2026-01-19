# Live Test Playbook (v1)
**Strategy:** Weekly Top-20 UPS (S&P 500)  
**Entry/Exit:** Monday open → Friday close  
**Rebalance:** Weekly (every Monday)  
**Objective:** Validate signal integrity + execution realism (not maximize profit)

---

## 1) What this test is trying to prove
This live test is successful if it shows:

1. The pipeline can run reliably each week (data → report → basket → trade log)
2. The system produces *repeatable, interpretable* rankings (not noise-hunting)
3. Live execution costs (slippage/spreads) do not destroy the signal
4. We can learn from mistakes using an **error ledger** without "moving the goalposts"

This is **not** a test of whether we get rich quickly.

---

## 2) Test length and "no early stopping"
- **Minimum:** 12 weeks
- **Preferred:** 24 weeks

Rules:
- Do not stop early because it's "working great"
- Do not stop early because it's "obviously broken"
- Do not change the model during the test window (see Section 8)

---

## 3) Capital and risk
### Recommended research capital
- **$25k–$50k** total (choose one and keep it constant)
- Separate mental bucket: "research capital"

### Position sizing
- **Equal-weight** across selected names
- If fewer than 20 qualify (e.g., sector cap), equal-weight the remainder

### No leverage (v1)
- No margin, no options, no shorting

---

## 4) Trading rules (no discretion)
### Basket construction
- Universe: **S&P 500**
- Selection: **Top 20 by UPS_adj**
- Sector cap: **max 5 names per sector**
- Weighting: equal-weight

### Execution
- Trades placed **Monday at market open**
- Order type: **market orders** (v1 realism)
- Always trade the full rebalance (do not "wait for a better fill")

### Forbidden discretionary overrides
Do not skip or override a trade because:
- "I already own it"
- "I don't like the company"
- "This feels crowded"
- "The market seems risky this week"
- "I think it's due for a pullback"

Allowed overrides (rare):
- Data integrity failure (symbol missing prices, malformed report)
- Corporate actions that break pricing (halt, delisting, etc.)

When an override occurs:
- record it in the **Override Log** (Section 7)

---

## 5) What you run each week (repeatable checklist)
### Friday evening / weekend (after the signal cutoff week ends)
1. Generate report:
   - `python -m src.report_weekly --week_end YYYY-MM-DD`
2. Review Top-20 UPS table and stock cards (10 minutes max)
3. Save the report (it should already be in `data/derived/reports/...`)

### Monday morning (execution)
1. Confirm basket tickers + weights
2. Place rebalance trades at market open
3. Record all fills in the trade log (Section 6)

### Friday after close (review)
1. Record weekly performance vs SPY
2. Update error ledger for the 2–3 biggest contributors and detractors
3. No model changes (only classify issues)

---

## 6) Logging requirements (these make the test "real")
You must keep three logs:

### A) Trade Log (required)
For each trade:
- date/time
- symbol
- buy/sell
- shares
- fill price
- broker-reported fees
- notes (optional)

### B) Portfolio Snapshot (weekly)
- signal_week_end (Friday)
- hold_entry_day (actual trading day used)
- hold_exit_day (actual trading day used)
- basket constituents + weights
- weekly return (gross and net)
- SPY return

### C) Error Ledger (weekly)
For each notable mover (winner/loser):
- symbol
- week
- what the report "thought" (conviction + rationale)
- what happened (price move)
- failure tag(s) (see Section 9)
- one-sentence hypothesis (no fixes yet)

---

## 7) Overrides (strongly discouraged)
If you override the system, log:
- date
- symbol(s)
- reason (choose one):
  - DATA_FAILURE
  - CORPORATE_ACTION
  - COMPLIANCE/BROKER_RESTRICTION
  - HUMAN_DISCRETION (discouraged)
- expected impact (one sentence)

If overrides exceed **2 in 12 weeks**, the test is considered contaminated.

---

## 8) "Freeze rules" (to prevent overfitting)
During the live test, you may not change:
- scoring formulas
- thresholds
- feature definitions
- selection rules (Top-20, sector cap)
- entry/exit convention (Mon open → Fri close)
- transaction cost assumptions used for analysis

You may change:
- bug fixes
- logging/report formatting
- speed improvements
- retry logic / rate limiting
- data validation

All model changes go into a **vNext backlog** and are only applied after the test window ends.

---

## 9) Failure tags (use these in the Error Ledger)
Pick 1–2 tags per issue:

**Data / pipeline**
- INGEST_GAP
- DEDUPE_CLUSTER_ERROR
- MISCLASSIFIED_EVENT
- MISCLASSIFIED_SENTIMENT
- MISSING_PRICE_DATA

**Signal / model**
- PRICE_ACTION_DOMINATED
- ECHO_NO_INFORMATION
- LATE_NEWS (after cutoff)
- WRONG_DIRECTION
- WRONG_MAGNITUDE
- SECTOR_CROWDING
- VOLATILITY_REGIME

**Execution**
- SLIPPAGE_TOO_HIGH
- SPREAD_TOO_WIDE
- LIQUIDITY_EVENT

**Human**
- OVERRIDE_CONTAMINATION
- EMOTIONAL_INTERFERENCE

---

## 10) What counts as "success" at the end of the test
After 12–24 weeks, the test is successful if at least **two** are true:

- Live results roughly track paper/backtest results (execution isn't killing it)
- Active return vs SPY is not obviously negative after costs
- Biggest failures cluster into fixable categories (not random)
- You find yourself checking the report before making other trades (usefulness)
- The system reduces cognitive load (less news-chasing)

It is possible to lose money and still have a "successful" test if learning is strong and the system is reliable.

---

## 11) Decision at the end
Choose one:
- **Continue v1** (extend to 24 weeks)
- **Fork v2** (apply backlog changes and start a new test window)
- **Stop** (if failures are unfixable or execution destroys any edge)

Record the decision and reasoning in `DECISIONS.md`.
