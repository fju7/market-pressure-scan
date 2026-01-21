#!/usr/bin/env python3
"""Generate complete test data structure for end-to-end ingestion testing."""
import json
import csv
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import numpy as np

def gen_test_data():
    """Generate 4 weeks of synthetic backtest data."""
    backtest_dir = Path("data/derived/backtest")
    scores_dir = Path("data/derived/scores_weekly")
    backtest_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate bt_weekly.parquet if not exists
    bt_file = backtest_dir / "bt_weekly.parquet"
    if not bt_file.exists():
        rows = []
        for i in range(4):
            week_end = date(2026, 1, 3) + timedelta(days=7*i)
            rows.append({
                'signal_week_end': week_end.isoformat(),
                'hold_entry_target': (week_end + timedelta(days=3)).isoformat(),
                'hold_exit_target': (week_end + timedelta(days=10)).isoformat(),
                'n_positions': 18,
                'missing_returns': 0,
                'gross_return': 0.012 + 0.002*i,
                'turnover': 0.5,
                'tcost': 0.0015,
                'net_return': 0.0105 + 0.002*i,
                'spy_return': 0.008,
                'active_net_return': 0.0025 + 0.002*i,
            })
        df = pd.DataFrame(rows)
        df.to_parquet(bt_file, index=False)
        print(f"✓ Created {bt_file}")
    
    # Generate week folders
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'FB', 'NFLX', 'GOOG', 'AMD',
               'ADBE', 'CRM', 'INTC', 'BKNG', 'PYPL', 'CMPR', 'WDAY', 'SQ', 'ZM', 'OKTA']
    
    for i in range(4):
        week_end = date(2026, 1, 3) + timedelta(days=7*i)
        week_folder = scores_dir / f"week_ending={week_end.isoformat()}"
        week_folder.mkdir(parents=True, exist_ok=True)
        
        # report_meta.json
        meta = {
            "week_ending": week_end.isoformat(),
            "universe_type": "test",
            "num_symbols_covered": 20,
            "week_type": "CLEAN_TRADE"
        }
        with (week_folder / "report_meta.json").open("w") as f:
            json.dump(meta, f, indent=2)
        
        # basket.csv
        with (week_folder / "basket.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["symbol"])
            writer.writeheader()
            for sym in symbols:
                writer.writerow({"symbol": sym})
        
        # scores_weekly.parquet (dummy)
        scores_df = pd.DataFrame({
            "symbol": symbols,
            "score": np.random.normal(0, 1, len(symbols))
        })
        scores_df.to_parquet(week_folder / "scores_weekly.parquet", index=False)
        
        # ops_compact_friday.log (dummy)
        (week_folder / "ops_compact_friday.log").write_text("# Log entry\n")
        
        print(f"✓ Created week folder: {week_folder}")
    
    print("\n✓ Test data generation complete!")

if __name__ == "__main__":
    gen_test_data()
