"""
Initialize live trading ledger CSV files

Run this once to create the empty CSV files with proper headers.
"""

from pathlib import Path
import pandas as pd

base = Path("data/live")
base.mkdir(parents=True, exist_ok=True)

trades = base / "trades_log.csv"
pnl = base / "weekly_pnl.csv"

if not trades.exists():
    pd.DataFrame(columns=[
        "week_ending_date", "symbol", "action", "weight_target",
        "entry_date", "entry_price", "shares", "notes"
    ]).to_csv(trades, index=False)
    print(f"✅ Created {trades}")
else:
    print(f"⏭️  {trades} already exists")

if not pnl.exists():
    pd.DataFrame(columns=[
        "week_ending_date", "symbol", "entry_price", "exit_price",
        "return_pct", "benchmark_return_pct", "active_return_pct", "notes"
    ]).to_csv(pnl, index=False)
    print(f"✅ Created {pnl}")
else:
    print(f"⏭️  {pnl} already exists")

print("\n📋 Ledger files ready:")
print(f"  - {trades}")
print(f"  - {pnl}")
