#!/usr/bin/env python
"""
Generate minimal synthetic backtest output for dashboard testing.
This creates realistic-looking bt_weekly.parquet and week folders.
"""
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
import numpy as np

def generate_test_backtest():
    """Generate synthetic backtest data matching bt_weekly.parquet schema."""
    
    # Create output directories
    backtest_dir = Path("data/derived/backtest")
    scores_dir = Path("data/derived/scores_weekly")
    backtest_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate 4 weeks of synthetic backtest data
    bt_rows = []
    pos_rows = []
    
    start_date = date(2026, 1, 3)  # Friday
    for i in range(4):
        week_end = start_date + timedelta(days=7*i)
        week_str = week_end.isoformat()
        
        # Entry/exit dates (Monday open to Friday close of following week)
        entry_target = week_end + timedelta(days=3)  # Monday
        exit_target = week_end + timedelta(days=10)  # next Friday
        
        # Synthetic returns (positive alpha trend)
        gross_ret = np.random.normal(0.01 + 0.005*i, 0.02)  # trending positive
        spy_ret = np.random.normal(0.005, 0.015)
        turnover = np.random.uniform(0.3, 0.7)
        tcost = turnover * 30 / 10000
        
        bt_rows.append({
            "signal_week_end": week_str,
            "hold_entry_target": entry_target.isoformat(),
            "hold_exit_target": exit_target.isoformat(),
            "n_positions": np.random.randint(15, 21),
            "missing_returns": np.random.randint(0, 3),
            "gross_return": float(gross_ret),
            "turnover": float(turnover),
            "tcost": float(tcost),
            "net_return": float(gross_ret - tcost),
            "spy_return": float(spy_ret),
            "active_net_return": float(gross_ret - tcost - spy_ret),
        })
        
        # Generate synthetic positions for this week
        n_pos = bt_rows[-1]["n_positions"]
        for j in range(n_pos):
            symbol = f"TEST{j%20:02d}"  # TEST00-TEST19
            ret_symbol = np.random.normal(gross_ret / n_pos, 0.015)
            
            pos_rows.append({
                "signal_week_end": week_str,
                "hold_entry_target": entry_target.isoformat(),
                "hold_exit_target": exit_target.isoformat(),
                "symbol": symbol,
                "sector": np.random.choice(["Technology", "Healthcare", "Finance", "Energy"]),
                "weight": 1.0 / n_pos,
                "UPS_adj": float(np.random.uniform(-1, 3)),
                "ret_mon_open_fri_close": float(ret_symbol),
            })
        
        # Create week folder with minimal files for ingestion
        week_folder = scores_dir / f"week_ending={week_str}"
        week_folder.mkdir(parents=True, exist_ok=True)
        
        # Create basket.csv
        basket_df = pd.DataFrame({
            "symbol": [f"TEST{j%20:02d}" for j in range(n_pos)],
            "weight": [1.0 / n_pos] * n_pos,
        })
        basket_df.to_csv(week_folder / "basket.csv", index=False)
        
        # Create scores_weekly.parquet (dummy)
        scores_df = pd.DataFrame({
            "symbol": [f"TEST{j%20:02d}" for j in range(20)],
            "UPS_adj": np.random.uniform(-1, 3, 20),
            "sector": np.random.choice(["Technology", "Healthcare", "Finance", "Energy"], 20),
        })
        scores_df.to_parquet(week_folder / "scores_weekly.parquet")
        
        # Create report_meta.json
        import json
        meta = {
            "week_ending": week_str,
            "experiment": "news-novelty-v1",
            "baseline_version": "V1",
            "universe_type": "test",
            "num_symbols_covered": 20,
            "week_type": "CLEAN_TRADE" if np.random.random() > 0.2 else "CLEAN_SKIP",
            "skip_reasons": {},
        }
        with open(week_folder / "report_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        # Create dummy ops log
        (week_folder / "ops_compact_friday.log").touch()
    
    # Write parquet files
    bt_df = pd.DataFrame(bt_rows)
    pos_df = pd.DataFrame(pos_rows)
    
    bt_path = backtest_dir / "bt_weekly.parquet"
    pos_path = backtest_dir / "bt_positions.parquet"
    
    bt_df.to_parquet(bt_path, index=False)
    pos_df.to_parquet(pos_path, index=False)
    
    print(f"✓ Created: {bt_path}")
    print(f"✓ Created: {pos_path}")
    print(f"✓ Created 4 week folders in {scores_dir}")
    print(f"\n=== bt_weekly.parquet ===")
    print(bt_df.to_string())
    print(f"\n=== Columns ===")
    print(list(bt_df.columns))

if __name__ == "__main__":
    generate_test_backtest()
