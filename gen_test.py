import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta

# Create directories
Path('data/derived/backtest').mkdir(parents=True, exist_ok=True)
Path('data/derived/scores_weekly').mkdir(parents=True, exist_ok=True)

# Generate synthetic data
rows = []
for i in range(4):
    week_end = date(2026, 1, 3) + timedelta(days=7*i)
    rows.append({
        'signal_week_end': week_end.isoformat(),
        'hold_entry_target': (week_end + timedelta(days=3)).isoformat(),
        'hold_exit_target': (week_end + timedelta(days=10)).isoformat(),
        'n_positions': 18,
        'missing_returns': 0,
        'gross_return': 0.012,
        'turnover': 0.5,
        'tcost': 0.0015,
        'net_return': 0.0105,
        'spy_return': 0.008,
        'active_net_return': 0.0025,
    })

df = pd.DataFrame(rows)
df.to_parquet('data/derived/backtest/bt_weekly.parquet')

# Output results
with open('_gen_output.txt', 'w') as f:
    f.write('Columns:\n')
    f.write(str(list(df.columns)) + '\n\n')
    f.write('Data:\n')
    f.write(df.to_string())
    f.write('\n\nFile created:\n')
    f.write(str(Path('data/derived/backtest/bt_weekly.parquet').absolute()))
