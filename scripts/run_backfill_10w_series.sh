#!/usr/bin/env bash
unset GITHUB_TOKEN
set -euo pipefail

# -------------------------
# CONFIG
# -------------------------
START_AT_ET="2026-02-06 00:00:00"
INTERVAL_MIN=1
REF="main"
WORKFLOW_FILE=".github/workflows/backfill_2weeks.yml"
REGIME="news-novelty-v1"
SCHEMA="news-novelty-v1b"
UNIVERSE="sp500_universe.csv"
FROM_STAGE="candles"
FORCE="true"

PAIRS=(
  "2025-07-04 2025-06-27"
  "2025-06-20 2025-06-13"
  "2025-06-06 2025-05-30"
  "2025-05-23 2025-05-16"
  "2025-05-09 2025-05-02"
)

DOWNLOAD_SCRIPT="scripts/download_backfill_artifacts.sh"
POLL_SEC=60

# -------------------------
# Helpers
# -------------------------
epoch_now() { date +%s; }

epoch_from_et() {
python - <<PY
from datetime import datetime
from zoneinfo import ZoneInfo
dt = datetime.strptime("${START_AT_ET}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("America/New_York"))
print(int(dt.timestamp()))
PY
}

utc_iso_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

sleep_until_epoch() {
  local target="$1"
  local now delta
  now="$(epoch_now)"
  delta=$(( target - now ))
  if (( delta > 0 )); then
    echo "Sleeping ${delta}s until $(date -d "@$target" "+%Y-%m-%d %H:%M:%S %Z") ..."
    sleep "$delta"
  else
    echo "Target time already passed by $((-delta))s; continuing."
  fi
}

find_run_id_after() {
  local title="$1"
  local after_utc="$2"
  gh run list \
    --workflow "$WORKFLOW_FILE" \
    --limit 60 \
    --json databaseId,displayTitle,createdAt \
    -q ".[] | select(.displayTitle == \"$title\") | select(.createdAt >= \"$after_utc\") | .databaseId" \
  | head -n 1
}

find_active_run_id() {
  local title="$1"
  gh run list \
    --workflow "$WORKFLOW_FILE" \
    --limit 60 \
    --json databaseId,displayTitle,status \
    -q ".[] | select(.displayTitle == \"$title\") | select(.status == \"in_progress\" or .status == \"queued\" or .status == \"pending\") | .databaseId" \
  | head -n 1
}

find_successful_run_id() {
  local title="$1"
  gh run list \
    --workflow "$WORKFLOW_FILE" \
    --limit 60 \
    --json databaseId,displayTitle,conclusion \
    -q ".[] | select(.displayTitle == \"$title\") | select(.conclusion == \"success\") | .databaseId" \
  | head -n 1
}

wait_for_run() {
  local run_id="$1"
  echo "Waiting for run ${run_id} to complete..."
  while true; do
    local conclusion
    conclusion="$(gh run view "$run_id" --json conclusion -q .conclusion 2>/dev/null || echo "")"
    if [[ -n "$conclusion" ]]; then
      echo "Run ${run_id} concluded: ${conclusion}"
      if [[ "$conclusion" != "success" ]]; then
        echo "❌ Run failed with conclusion=${conclusion}. Stopping."
        exit 1
      fi
      break
    fi
    sleep "$POLL_SEC"
  done
}

# -------------------------
# MAIN
# -------------------------
echo "=== Backfill 10 weeks series ==="
echo "Start time: ${START_AT_ET}"
echo "Pairs: ${#PAIRS[@]}"
echo ""

TARGET_EPOCH="$(epoch_from_et)"
sleep_until_epoch "$TARGET_EPOCH"

for pair in "${PAIRS[@]}"; do
  read -r w1 w2 <<< "$pair"
  
  RUN_TITLE="backfill-2w ${w1}_${w2} regime=${REGIME} schema=${SCHEMA}"
  echo ""
  echo "========================================"
  echo "PAIR: ${w1} + ${w2}"
  echo "Run title: ${RUN_TITLE}"
  echo "========================================"
  
  # Check for successful run first
  SUCCESSFUL_RUN_ID="$(find_successful_run_id "$RUN_TITLE")"
  if [[ -n "$SUCCESSFUL_RUN_ID" ]]; then
    echo "✅ Found successful run ${SUCCESSFUL_RUN_ID} for this pair. Reusing it."
    RUN_ID="$SUCCESSFUL_RUN_ID"
  else
    # Check for active run
    ACTIVE_RUN_ID="$(find_active_run_id "$RUN_TITLE")"
    if [[ -n "$ACTIVE_RUN_ID" ]]; then
      echo "⚠️  Found active run ${ACTIVE_RUN_ID} for this pair. Reusing it."
      RUN_ID="$ACTIVE_RUN_ID"
    else
      # Dispatch new run
      DISPATCH_TIME_UTC="$(utc_iso_now)"
      echo "Dispatching workflow for ${w1} + ${w2} at ${DISPATCH_TIME_UTC} ..."
      
      gh workflow run "$WORKFLOW_FILE" \
        --ref "$REF" \
        --field week_end_1="$w1" \
        --field week_end_2="$w2" \
        --field regime="$REGIME" \
        --field schema="$SCHEMA" \
        --field universe="$UNIVERSE" \
        --field from_stage="$FROM_STAGE" \
        --field force="$FORCE"
      
      sleep 10
      
      echo "Locating run..."
      RUN_ID=""
      for attempt in {1..12}; do
        RUN_ID="$(find_run_id_after "$RUN_TITLE" "$DISPATCH_TIME_UTC")"
        if [[ -n "$RUN_ID" ]]; then
          echo "Found run: ${RUN_ID}"
          break
        fi
        echo "Attempt ${attempt}/12: not found yet, sleeping..."
        sleep 5
      done
      
      if [[ -z "$RUN_ID" ]]; then
        echo "❌ Could not locate run after 60s. Aborting."
        exit 1
      fi
    fi
  fi
  
  # Wait for completion if not already successful
  if [[ -z "$SUCCESSFUL_RUN_ID" ]]; then
    wait_for_run "$RUN_ID"
  fi
  
  echo "Downloading artifacts for run ${RUN_ID}..."
  bash "$DOWNLOAD_SCRIPT" "$RUN_TITLE"
  
  echo "✅ Completed pair ${w1} + ${w2}"
  
  # Wait interval before next dispatch
  NEXT_EPOCH=$(( $(epoch_now) + INTERVAL_MIN * 60 ))
  sleep_until_epoch "$NEXT_EPOCH"
done

echo ""
echo "=== ALL 10 WEEKS COMPLETE ==="
echo "Run backtest and evaluation to verify:"
echo "  python -m src.backtest_weekly"
echo "  python -m src.research.eval_complete_weeks --calendar_mode strict --no_normalize_week_end"
