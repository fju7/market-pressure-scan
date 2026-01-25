#!/usr/bin/env python
"""Quick verification script for weeks_log.csv"""

import pandas as pd

df = pd.read_csv("data/live/weeks_log.csv")

print(f"✅ Verified weeks_log.csv")
print(f"   Rows: {len(df)}")
print(f"   Unique weeks: {df['week_ending_date'].nunique()}")
print(f"   Date range: {df['week_ending_date'].min()} to {df['week_ending_date'].max()}")
print(f"\n   Actions breakdown:")
print(df["action"].value_counts().to_string())

# Check for duplicates
dups = df[df.duplicated(subset=["week_ending_date"], keep=False)]
if len(dups) > 0:
    print(f"\n❌ WARNING: {len(dups)} duplicate week(s) found!")
    print(dups[["week_ending_date", "action"]])
else:
    print(f"\n✅ No duplicates found")
