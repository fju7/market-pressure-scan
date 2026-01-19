"""
Complete weekly workflow automation

This orchestrates the entire Friday → Monday → Friday cycle.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str):
    """Run a command and handle errors"""
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        sys.exit(1)
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Automated weekly trading workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Weekly Operating Routine:

FRIDAY (or Weekend) - Signal Generation:
  python -m src.weekly_workflow --week_end YYYY-MM-DD --mode friday
  
  This will:
  1. Run the full pipeline (news, clustering, scoring, report)
  2. Export basket (with skip-week logic)
  3. Log the week's decision to weeks_log.csv
  4. Display next steps

MONDAY - Trade Execution (if TRADE):
  python -m src.weekly_workflow --week_end YYYY-MM-DD --mode monday \\
    --execution_date YYYY-MM-DD --account_value XXXXX
  
  This will:
  1. Fetch live prices from Finnhub
  2. Calculate position sizes
  3. Log trades to trades_log.csv
  
  Then manually execute the trades and update entry_price if needed.

FRIDAY - P&L Recording (if TRADE):
  python -m src.weekly_workflow --week_end YYYY-MM-DD --mode friday_close \\
    --exit_date YYYY-MM-DD
  
  This will:
  1. Read entry prices from trades_log
  2. Get exit prices from candles_daily
  3. Calculate returns vs SPY
  4. Log to weekly_pnl.csv

ANYTIME - Performance Analysis:
  python -m src.analyze_performance
  
  Shows hit rate, returns, drawdowns, skip rate, etc.
        """
    )
    
    ap.add_argument("--week_end", required=True, help="Week ending date YYYY-MM-DD")
    ap.add_argument(
        "--mode",
        required=True,
        choices=["friday", "monday", "friday_close"],
        help="Workflow stage: friday (signal), monday (execute), friday_close (record P&L)"
    )
    ap.add_argument("--execution_date", help="Trade execution date (for monday mode)")
    ap.add_argument("--account_value", type=float, help="Account value (for monday mode)")
    ap.add_argument("--exit_date", help="Exit date for P&L (for friday_close mode)")
    ap.add_argument("--max_clusters_per_symbol", type=int, default=1)
    
    args = ap.parse_args()
    
    py = sys.executable
    
    if args.mode == "friday":
        # Friday: Generate signals
        print("\n🏁 FRIDAY: Signal Generation")
        print(f"   Week ending: {args.week_end}")
        
        # Run pipeline
        run_command(
            [py, "-m", "src.run_weekly_pipeline",
             "--week_end", args.week_end,
             "--max_clusters_per_symbol", str(args.max_clusters_per_symbol),
             "--skip_backtest"],
            "Running weekly pipeline"
        )
        
        # Log decision
        run_command(
            [py, "-m", "src.log_week_decision",
             "--week_end", args.week_end],
            "Logging week decision"
        )
        
        print("\n" + "="*70)
        print("✅ FRIDAY workflow complete!")
        print("="*70)
        print("\n📋 Next steps:")
        print("   1. Review the report:")
        print(f"      data/derived/reports/week_ending={args.week_end}/weekly_report.md")
        print("   2. Check basket.csv:")
        print(f"      data/derived/baskets/week_ending={args.week_end}/basket.csv")
        print("   3. If action=TRADE, prepare for Monday execution")
        print("   4. If action=SKIP, no further action needed\n")
    
    elif args.mode == "monday":
        # Monday: Execute trades
        if not args.execution_date:
            print("❌ --execution_date required for monday mode")
            sys.exit(1)
        if not args.account_value:
            print("❌ --account_value required for monday mode")
            sys.exit(1)
        
        print("\n💼 MONDAY: Trade Execution")
        print(f"   Week ending: {args.week_end}")
        print(f"   Execution date: {args.execution_date}")
        print(f"   Account value: ${args.account_value:,.2f}")
        
        # Check if week is TRADE
        basket_path = Path(f"data/derived/baskets/week_ending={args.week_end}/basket.csv")
        if not basket_path.exists():
            print(f"❌ Basket not found: {basket_path}")
            sys.exit(1)
        
        import pandas as pd
        basket = pd.read_csv(basket_path)
        if basket["action"].iloc[0] == "SKIP":
            print("⏭️  This week is SKIP - no trades to execute")
            sys.exit(0)
        
        # Log trades (fetches live prices)
        run_command(
            [py, "-m", "src.log_trades",
             "--week_end", args.week_end,
             "--execution_date", args.execution_date,
             "--account_value", str(args.account_value)],
            "Logging trades with live prices"
        )
        
        print("\n" + "="*70)
        print("✅ MONDAY workflow complete!")
        print("="*70)
        print("\n📋 Next steps:")
        print("   1. Review logged trades:")
        print("      cat data/live/trades_log.csv")
        print("   2. Execute the trades in your brokerage")
        print("   3. If fill prices differ, manually update trades_log.csv")
        print(f"   4. On Friday: run with --mode friday_close --exit_date YYYY-MM-DD\n")
    
    elif args.mode == "friday_close":
        # Friday: Record P&L
        if not args.exit_date:
            print("❌ --exit_date required for friday_close mode")
            sys.exit(1)
        
        print("\n📊 FRIDAY CLOSE: P&L Recording")
        print(f"   Week ending: {args.week_end}")
        print(f"   Exit date: {args.exit_date}")
        
        # Check if trades exist for this week
        trades_path = Path("data/live/trades_log.csv")
        if not trades_path.exists():
            print("❌ No trades_log.csv found - did you execute Monday workflow?")
            sys.exit(1)
        
        import pandas as pd
        trades = pd.read_csv(trades_path)
        week_trades = trades[trades["notes"].str.contains(f"week ending {args.week_end}", na=False)]
        
        if week_trades.empty:
            print(f"⏭️  No trades found for week ending {args.week_end}")
            print("   This week was likely SKIP - no P&L to record")
            sys.exit(0)
        
        # Log P&L
        run_command(
            [py, "-m", "src.log_weekly_pnl",
             "--week_end", args.week_end,
             "--exit_date", args.exit_date],
            "Logging weekly P&L"
        )
        
        print("\n" + "="*70)
        print("✅ FRIDAY CLOSE workflow complete!")
        print("="*70)
        print("\n📋 Week complete! View results:")
        print("   cat data/live/weekly_pnl.csv")
        print("\n📊 Run performance analysis anytime:")
        print("   python -m src.analyze_performance\n")


if __name__ == "__main__":
    main()
