#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_FILE=".github/workflows/backfill_2weeks.yml"
OUTROOT="artifacts/backfills"
POLL_SEC=60

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: gh CLI not found."
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: gh not authenticated. Run: gh auth login"
  exit 1
fi

run_id_for_title_success() {
  local title="$1"
  gh run list \
    --workflow "$WORKFLOW_FILE" \
    --limit 80 \
    --json databaseId,displayTitle,conclusion,createdAt,status \
    -q ".[] | select(.displayTitle == \"$title\") | select(.conclusion == \"success\") | .databaseId" \
  | head -n 1
}

run_id_for_title_any() {
  local title="$1"
  gh run list \
    --workflow "$WORKFLOW_FILE" \
    --limit 80 \
    --json databaseId,displayTitle,conclusion,createdAt,status \
    -q ".[] | select(.displayTitle == \"$title\") | .databaseId" \
  | head -n 1
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
        echo "WARNING: run completed with conclusion=$concl"
      fi
      break
    fi
    sleep "$POLL_SEC"
  done
}

download_artifacts() {
  local run_id="$1"
  local outdir="$2"
  mkdir -p "$outdir"

  gh run view "$run_id" --json databaseId,displayTitle,createdAt,startedAt,updatedAt,status,conclusion,headSha,event,workflowName \
    > "$outdir/run_meta.json"

  gh run download "$run_id" --dir "$outdir"

  echo "Downloaded artifacts to: $outdir"
  find "$outdir" -maxdepth 3 -type f | sed 's|^|  |'
}

usage() {
  echo "Usage:"
  echo "  $0 \"backfill-2w 2025-10-10_2025-10-03 regime=news-novelty-v1 schema=news-novelty-v1b\""
  exit 2
}

[[ $# -ge 1 ]] || usage
TITLE="$1"
RUN_ID="${2:-}"

if [[ -z "${RUN_ID:-}" ]]; then
  RUN_ID="$(run_id_for_title_success "$TITLE" || true)"
fi
if [[ -z "${RUN_ID:-}" ]]; then
  RUN_ID="$(run_id_for_title_any "$TITLE" || true)"
fi

if [[ -z "${RUN_ID:-}" ]]; then
  echo "ERROR: could not find run with title exactly:"
  echo "  $TITLE"
  echo
  echo "Tip: list recent runs:"
  echo "  gh run list --workflow $WORKFLOW_FILE --limit 10"
  exit 1
fi

echo "Found run_id: $RUN_ID for title: $TITLE"
wait_for_complete "$RUN_ID"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SAFE_TITLE="$(echo "$TITLE" | tr ' /:=,' '_' | tr -cd 'A-Za-z0-9_.-_' )"
OUTDIR="$OUTROOT/${SAFE_TITLE}__run${RUN_ID}__${STAMP}"

download_artifacts "$RUN_ID" "$OUTDIR"
