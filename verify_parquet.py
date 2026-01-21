#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

bt_file = Path("data/derived/backtest/bt_weekly.parquet")
if bt_file.exists():
    df = pd.read_parquet(bt_file)
    print("✓ Columns:", list(df.columns))
    print("\nData shape:", df.shape)
    print("\nFirst row:")
    for col in df.columns:
        print(f"  {col}: {df[col].iloc[0]}")
else:
    print("✗ File not found:", bt_file)
