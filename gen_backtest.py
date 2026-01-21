#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
import numpy as np

# Create output directories
backtest_dir = Path('data/derived/backtest')
scores_dir = Path('data/derived/scores_weekly')
backtest_dir.mkdir(parents=True, exist_ok=True)
scores_dir.mkdir(parents=True, exist_ok=True)

# Generate synthetic backtest data
bt_rows = []
start_date = date(2026, 1, 3)
for i in range(4):
    week_end = start_date + timedelta(days=7*i)
    week_str = week_end.isoformat()
    entry_target = week_end + timedelta(days=3)
    exit_target = week_end + timedelta(days=10)
    
    gross_ret = np.random.normal(0.01 + 0.005*i, 0.02)
    spy_ret = np.random.normal(0.005, 0.015)
    turnover = np.random.uniform(0.3, 0.7)
    tcost = turnover * 30 / 10000
    
    bt_rows.append({
        'signal_week_end': week_str,
        'hold_entry_target': entry_target.isoformat(),
        'hold_exit_target': exit_target.isoformat(),
        'n_positions': int(np.random.randint(15, 21)),
        'missing_returns': int(np.random.randint(0, 3)),
        'gross_return': float(gross_ret),
        'turnover': float(turnover),
        'tcost': float(tcost),
        'net_return': float(gross_ret - tcost),
        'spy_return': float(spy_ret),
        'active_net_return': float(gross_ret - tcost - spy_ret),
    })

bt_df = pd.DataFrame(bt_rows)
bt_path = backtest_dir / 'bt_weekly.parquet'
bt_df.to_parquet(bt_path, index=False)
print('✓ Generated', bt_path)
print('\nColumns:', list(bt_df.columns))
print('\nData:')
print(bt_df.to_string())
