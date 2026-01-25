#!/bin/bash
# Package all artifacts for a given week into a zip file
# Usage: ./scripts/package_week_artifacts.sh YYYY-MM-DD [output_name]

set -euo pipefail

WEEK_END="${1:-}"
OUTPUT_NAME="${2:-week_${WEEK_END}_artifacts.zip}"

if [[ -z "$WEEK_END" ]]; then
    echo "❌ ERROR: Week ending date required"
    echo "Usage: $0 YYYY-MM-DD [output_name]"
    exit 1
fi

echo "📦 Packaging artifacts for week_ending=$WEEK_END"

# Find all directories matching this week
WEEK_DIRS=$(find data -type d -name "week_ending=$WEEK_END" 2>/dev/null || true)

if [[ -z "$WEEK_DIRS" ]]; then
    echo "⚠️  No artifacts found for week_ending=$WEEK_END"
    exit 1
fi

echo "Found directories:"
echo "$WEEK_DIRS" | sed 's/^/  - /'

# Create zip with all week artifacts
# Also include live data and week_end.txt if they exist
zip -r "$OUTPUT_NAME" \
    $(find data -type d -name "week_ending=$WEEK_END") \
    $(find data/live -type f 2>/dev/null || true) \
    week_end.txt \
    2>/dev/null || true

if [[ -f "$OUTPUT_NAME" ]]; then
    SIZE=$(du -h "$OUTPUT_NAME" | cut -f1)
    echo "✅ Created: $OUTPUT_NAME ($SIZE)"
    echo ""
    echo "Contents:"
    unzip -l "$OUTPUT_NAME" | tail -n +4 | head -n -2
else
    echo "❌ Failed to create archive"
    exit 1
fi
