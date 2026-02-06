#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# CONFIG
# -------------------------
START_AT_ET="2026-02-04 20:00:00"   # 8:00pm ET
INTERVAL_MIN=135                   # 2h15m spacing (earliest)
REF="main"
WORKFLOW_FILE=".github/workflows/backfill_2weeks.yml"

REGIME="news-novelty-v1"
SCHEMA="news-novelty-v1b"
UNIVERSE="sp500_universe.csv"
FROM_STAGE="candles"
FORCE="true"

# 7 runs x 2 weeks = 14 weeks (back from 2025-10-10)
PAIRS=(
  "2025-10-10 2025-10-03"
  "2025-09-26 2025-09-19"
  "2025-09-12 2025-09-05"
  "2025-08-29 2025-08-22"
  "2025-08-15 2025-08-08"
  "2025-08-01 2025-07-25"
  "2025-07-18 2025-07-11"
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

# Find the newest run matching the exact displayTitle AND createdAt >= DISPATCH_TIME_UTC
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

# Find an active (pending/in_progress/queued) run matching the exact displayTitle.
find_active_run_id() {
  local title="$1"

  gh run list     --workflow "$WORKFLOW_FILE"     --limit 60     --json databaseId,displayTitle,status     -q ".[] | select(.displayTitle == \"$title\") | select(.status == \"in_progress\" or .status == \"queued\" or .status == \"pending\") | .databaseId"   | head -n 1
}

# Find the newest SUCCESS completed run matching the exact displayTitle.
find_success_run_id() {
  local title="$1"

  gh run list     --workflow "$WORKFLOW_FILE"     --limit 80     --json databaseId,displayTitle,status,conclusion,createdAt     -q ".[] | select(.displayTitle == \"$title\") | select(.status == \"completed\") | select(.conclusion == \"success\") | .databaseId"   | head -n 1
}



wait_for_run_id() {
  local title="$1"
  local after_utc="$2"

  local run_id=""
  for _ in $(seq 1 60); do
    run_id="$(find_run_id_after "$title" "$after_utc" || true)"
    if [[ -n "${run_id:-}" ]]; then
      echo "$run_id"
      return 0
    fi
    echo "Waiting for run to appear in GitHub UI..." >&2
    sleep 10
  done

  echo "ERROR: Could not find run for title after dispatch time:"
  echo "  title=$title"
  echo "  after_utc=$after_utc"
  exit 1
}

wait_for_complete() {
  local run_id="$1"
  while true; do
    local status concl
    status="$(gh run view "$run_id" --json status -q .status)"
    concl="$(gh run view "$run_id" --json conclusion -q .conclusion)"
    echo "run_id=$run_id status=$status conclusion=$concl"
    if [[ "$status" == "completed" ]]; then
      if [[ "$concl" != "success" ]]; then
        echo "WARNING: run completed with conclusion=$concl (still downloading artifacts)."
      fi
      break
    fi
    sleep "$POLL_SEC"
  done
}

# -------------------------
# Preflight
# -------------------------
if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: gh CLI not found."
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: gh not authenticated. Run: gh auth login"
  exit 1
fi
if [[ ! -x "$DOWNLOAD_SCRIPT" ]]; then
  echo "ERROR: downloader not found or not executable: $DOWNLOAD_SCRIPT"
  echo "Fix: chmod +x $DOWNLOAD_SCRIPT"
  exit 1
fi

echo "Workflow: ${WORKFLOW_FILE}  ref: ${REF}"
echo "Regime:   ${REGIME}"
echo "Schema:   ${SCHEMA}"
echo "Universe: ${UNIVERSE}"
echo "Stage:    ${FROM_STAGE}"
echo "Force:    ${FORCE}"
echo "Runs:     ${#PAIRS[@]} (2 weeks each)"
echo

START_EPOCH="$(epoch_from_et)"

# -------------------------
# Series loop
# -------------------------
for i in "${!PAIRS[@]}"; do
  read -r W1 W2 <<<"${PAIRS[$i]}"

  # Earliest dispatch time for this slot
  SLOT_EPOCH=$(( START_EPOCH + i*INTERVAL_MIN*60 ))

  echo "=================================================================="
  echo "Run $((i+1))/${#PAIRS[@]}  week_end_1=${W1}  week_end_2=${W2}"
  echo "=================================================================="

  sleep_until_epoch "$SLOT_EPOCH"

  TITLE="backfill-2w ${W1}_${W2} regime=${REGIME} schema=${SCHEMA}"
  DISPATCH_UTC="$(utc_iso_now)"

  echo "Dispatching at (UTC): $DISPATCH_UTC"
  echo "Expected run-name/title: $TITLE"

  # Prefer: attach to active run -> reuse successful completed run -> otherwise dispatch.
  EXISTING_ID="$(find_active_run_id "$TITLE" || true)"
  if [[ -n "${EXISTING_ID:-}" ]]; then
    RUN_ID="$EXISTING_ID"
    echo "Found existing active run_id: $RUN_ID (skipping dispatch)"
  else
    SUCCESS_ID="$(find_success_run_id "$TITLE" || true)"
    if [[ -n "${SUCCESS_ID:-}" ]]; then
      RUN_ID="$SUCCESS_ID"
      echo "Found existing successful run_id: $RUN_ID (skipping dispatch)"
    else
      gh workflow run "${WORKFLOW_FILE}" \
        --ref "${REF}" \
        -f week_end_1="${W1}" \
        -f week_end_2="${W2}" \
        -f regime="${REGIME}" \
        -f schema="${SCHEMA}" \
        -f universe="${UNIVERSE}" \
        -f from_stage="${FROM_STAGE}" \
        -f force="${FORCE}"

      echo "Submitted workflow dispatch. Locating run id..."
      RUN_ID="$(wait_for_run_id "$TITLE" "$DISPATCH_UTC")"
      echo "Found run_id: $RUN_ID"
    fi
  fi

  echo "Waiting for completion..."
  wait_for_complete "$RUN_ID"

  echo "Downloading artifacts..."
  "$DOWNLOAD_SCRIPT" "$TITLE" "$RUN_ID"

  echo
done

echo "✅ All 7 runs dispatched, completed, and artifacts downloaded."
