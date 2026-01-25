#!/bin/bash
# Quick verification script for weeks_log.csv
# Usage: ./verify_weeks_log.sh

set -euo pipefail

echo "🔍 Weeks Log Verification"
echo "========================="
echo ""

# Check if file exists
if [ ! -f "data/live/weeks_log.csv" ]; then
    echo "❌ File not found: data/live/weeks_log.csv"
    exit 1
fi

# Line count
LINE_COUNT=$(wc -l < data/live/weeks_log.csv)
ROW_COUNT=$((LINE_COUNT - 1))  # Subtract header

echo "📊 Basic Stats:"
echo "   Total lines: $LINE_COUNT (${ROW_COUNT} weeks + 1 header)"
echo ""

# Python analysis
python - <<'PY'
import pandas as pd

df = pd.read_csv("data/live/weeks_log.csv")

print(f"📅 Date Range:")
print(f"   {df['week_ending_date'].min()} → {df['week_ending_date'].max()}")
print()

print(f"📈 Action Breakdown:")
for action, count in df["action"].value_counts().items():
    pct = (count / len(df)) * 100
    print(f"   {action:6s}: {count:3d} ({pct:5.1f}%)")
print()

# Check for duplicates
dups = df[df.duplicated(subset=["week_ending_date"], keep=False)]
if len(dups) > 0:
    print(f"❌ Duplicates Found: {len(dups)} rows")
    print(dups[["week_ending_date", "action"]])
    print()
else:
    print("✅ No duplicates")
    print()

# Check for missing weeks
weeks = pd.to_datetime(df["week_ending_date"])
week_diff = weeks.diff().dt.days
gaps = week_diff[week_diff > 7]
if len(gaps) > 0:
    print(f"⚠️  Gaps Found: {len(gaps)} week(s) with >7 day spacing")
    for idx in gaps.index:
        prev_week = weeks.iloc[idx - 1].strftime("%Y-%m-%d")
        curr_week = weeks.iloc[idx].strftime("%Y-%m-%d")
        gap_days = int(week_diff.iloc[idx])
        print(f"   {prev_week} → {curr_week} ({gap_days} days)")
    print()
else:
    print("✅ No gaps detected (all weeks are ≤7 days apart)")
    print()

# Check column completeness
print(f"📋 Schema:")
print(f"   Columns: {len(df.columns)}")
null_counts = df.isnull().sum()
if null_counts.any():
    print(f"   ⚠️  Null values found:")
    for col, count in null_counts[null_counts > 0].items():
        print(f"      {col}: {count}")
else:
    print(f"   ✅ No null values")
print()

# Summary for TRADE weeks
trade_df = df[df["action"] == "TRADE"]
if len(trade_df) > 0:
    print(f"📊 TRADE Weeks Summary:")
    print(f"   Count: {len(trade_df)}")
    print(f"   Avg basket size: {trade_df['basket_size'].astype(float).mean():.1f}")
    if trade_df['turnover_pct'].astype(float).sum() > 0:
        print(f"   Avg turnover: {trade_df['turnover_pct'].astype(float).mean():.1f}%")
        print(f"   Avg overlap: {trade_df['overlap_pct'].astype(float).mean():.1f}%")
    print()

print("=" * 50)
print("✅ Verification complete")
PY
