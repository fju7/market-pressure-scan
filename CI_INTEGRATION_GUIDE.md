# GitHub Actions Integration Example

## Adding Regime System to weekly.yml

### Option 1: Run Both Regimes in Parallel

```yaml
- name: Run weekly pipeline (v1 baseline)
  env:
    FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    WEEK_END="${{ steps.week_end.outputs.week_end }}"
    
    # Score with v1 (baseline)
    python -m src.features_scores \
      --universe sp500_universe.csv \
      --week_end "$WEEK_END" \
      --regime news-novelty-v1
    
    # Audit v1
    python scripts/audit_regime_run.py \
      --week_end "$WEEK_END" \
      --regime news-novelty-v1 \
      --fail_on_error

- name: Run weekly pipeline (v1b divergence-heavy)
  env:
    FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    WEEK_END="${{ steps.week_end.outputs.week_end }}"
    
    # Score with v1b
    python -m src.features_scores \
      --universe sp500_universe.csv \
      --week_end "$WEEK_END" \
      --regime news-novelty-v1b
    
    # Audit v1b
    python scripts/audit_regime_run.py \
      --week_end "$WEEK_END" \
      --regime news-novelty-v1b \
      --fail_on_error
```

### Option 2: Run Single Regime (Simpler)

```yaml
- name: Run weekly pipeline
  env:
    FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    REGIME_ID: news-novelty-v1b  # Or set via workflow_dispatch input
  run: |
    WEEK_END="${{ steps.week_end.outputs.week_end }}"
    
    # Score with chosen regime
    python -m src.features_scores \
      --universe sp500_universe.csv \
      --week_end "$WEEK_END" \
      --regime "$REGIME_ID"
    
    # Audit
    python scripts/audit_regime_run.py \
      --week_end "$WEEK_END" \
      --regime "$REGIME_ID" \
      --fail_on_error
```

### Updated Artifact Upload

```yaml
- name: Upload week artifacts
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: week_${{ steps.week_end.outputs.week_end }}_run_${{ github.run_id }}
    path: |
      week_end.txt
      # v1 artifacts
      data/derived/scores_weekly/regime=news-novelty-v1/week_ending=${{ steps.week_end.outputs.week_end }}/scores_weekly.parquet
      data/derived/scores_weekly/regime=news-novelty-v1/week_ending=${{ steps.week_end.outputs.week_end }}/report_meta.json
      data/derived/scores_weekly/regime=news-novelty-v1/week_ending=${{ steps.week_end.outputs.week_end }}/schema_used.yaml
      data/derived/features_weekly/regime=news-novelty-v1/week_ending=${{ steps.week_end.outputs.week_end }}/features_weekly.parquet
      # v1b artifacts
      data/derived/scores_weekly/regime=news-novelty-v1b/week_ending=${{ steps.week_end.outputs.week_end }}/scores_weekly.parquet
      data/derived/scores_weekly/regime=news-novelty-v1b/week_ending=${{ steps.week_end.outputs.week_end }}/report_meta.json
      data/derived/scores_weekly/regime=news-novelty-v1b/week_ending=${{ steps.week_end.outputs.week_end }}/schema_used.yaml
      data/derived/features_weekly/regime=news-novelty-v1b/week_ending=${{ steps.week_end.outputs.week_end }}/features_weekly.parquet
      # Other artifacts
      data/live/scoreboard.csv
      data/live/weeks_log.csv
```

### workflow_dispatch Input for Regime Selection

```yaml
on:
  workflow_dispatch:
    inputs:
      week_end:
        description: "Week ending date (YYYY-MM-DD)"
        required: false
        default: ""
      regime:
        description: "Regime ID (news-novelty-v1 or news-novelty-v1b)"
        required: false
        default: "news-novelty-v1"
        type: choice
        options:
          - news-novelty-v1
          - news-novelty-v1b
```

## Minimal Integration (Fastest Path to Production)

Replace current `src.run_weekly_pipeline` call with:

```yaml
- name: Run weekly pipeline
  env:
    FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    WEEK_END="${{ steps.week_end.outputs.week_end }}"
    
    # Use features_scores directly instead of run_weekly_pipeline
    python -m src.features_scores \
      --universe sp500_universe.csv \
      --week_end "$WEEK_END" \
      --regime news-novelty-v1
    
    # Audit
    python scripts/audit_regime_run.py \
      --week_end "$WEEK_END" \
      --regime news-novelty-v1 \
      --fail_on_error
    
    # Continue with rest of pipeline
    python -m src.export_basket --week_end "$WEEK_END" --regime news-novelty-v1
    python -m src.report_weekly --week_end "$WEEK_END" --regime news-novelty-v1
    python -m src.trader_sheet --week_end "$WEEK_END" --regime news-novelty-v1
```

## Notification Update (Show Regime)

```javascript
const meta_path = `data/derived/scores_weekly/regime=${regime}/week_ending=${week_end}/report_meta.json`;
const meta = JSON.parse(fs.readFileSync(meta_path, 'utf8'));

const body = `@fju7 — Your weekly market pressure scan is ready!

## 📊 Weekly Market Pressure Scan Complete

**Week Ending:** ${week_end}
**Regime:** ${meta.regime_id}
**Schema Hash:** ${meta.schema_hash}
**Status:** ${status === 'success' ? '✅ Success' : '❌ Failed'}

...
`;
```

## Testing Integration Locally

```bash
# Simulate CI environment
export FINNHUB_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
export WEEK_END="2026-01-09"

# Run scoring
python -m src.features_scores \
  --universe sp500_universe.csv \
  --week_end "$WEEK_END" \
  --regime news-novelty-v1

# Audit
python scripts/audit_regime_run.py \
  --week_end "$WEEK_END" \
  --regime news-novelty-v1 \
  --fail_on_error

echo "✅ Integration test passed"
```

## Rollout Strategy

### Phase 1: Shadow v1b (Recommended)
1. Keep v1 as production
2. Run v1b in parallel (no decisions based on it yet)
3. Collect ledger data for 4-8 weeks
4. Compare trade/skip decisions

### Phase 2: A/B Test
1. Alternate weeks: v1 odd weeks, v1b even weeks
2. Track performance in ledger
3. Analyze which regime trades better

### Phase 3: Commit to Winner
1. Set winning regime as default
2. Keep both running for reference
3. Continue ledger tracking

## Ledger Analysis in CI

```yaml
- name: Update regime comparison ledger
  run: |
    WEEK_END="${{ steps.week_end.outputs.week_end }}"
    
    # Append to ledger
    python - <<PY
import pandas as pd
import json
from pathlib import Path

week = "$WEEK_END"
ledger_path = Path("data/derived/regime_ledger.csv")

# Extract metadata
v1_meta = json.loads(Path(f"data/derived/scores_weekly/regime=news-novelty-v1/week_ending={week}/report_meta.json").read_text())
v1b_meta = json.loads(Path(f"data/derived/scores_weekly/regime=news-novelty-v1b/week_ending={week}/report_meta.json").read_text())

# Load scores to determine skip/trade
v1_scores = pd.read_parquet(f"data/derived/scores_weekly/regime=news-novelty-v1/week_ending={week}/scores_weekly.parquet")
v1b_scores = pd.read_parquet(f"data/derived/scores_weekly/regime=news-novelty-v1b/week_ending={week}/scores_weekly.parquet")

# Create record
record = {
    "week_end": week,
    "v1_n_candidates": len(v1_scores),
    "v1b_n_candidates": len(v1b_scores),
    "v1_schema_hash": v1_meta["schema_hash"],
    "v1b_schema_hash": v1b_meta["schema_hash"],
    "git_sha": v1_meta["git_sha"],
}

# Append to ledger
if ledger_path.exists():
    ledger = pd.read_csv(ledger_path)
    ledger = pd.concat([ledger, pd.DataFrame([record])], ignore_index=True)
else:
    ledger = pd.DataFrame([record])

ledger.to_csv(ledger_path, index=False)
print(f"✅ Updated ledger: {ledger_path}")
PY
```

## Recommended Workflow Structure

```yaml
jobs:
  score-regimes:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        regime: [news-novelty-v1, news-novelty-v1b]
    steps:
      - name: Score with ${{ matrix.regime }}
        run: |
          python -m src.features_scores \
            --universe sp500_universe.csv \
            --week_end "$WEEK_END" \
            --regime ${{ matrix.regime }}
      
      - name: Audit ${{ matrix.regime }}
        run: |
          python scripts/audit_regime_run.py \
            --week_end "$WEEK_END" \
            --regime ${{ matrix.regime }} \
            --fail_on_error
```

This runs both regimes in parallel using GitHub Actions matrix strategy.
