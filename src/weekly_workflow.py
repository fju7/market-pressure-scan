"""
Complete weekly workflow automation

This orchestrates the entire Friday → Monday → Friday cycle.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from pathlib import Path
import pandas as pd

import pandas as pd

def compute_auto_week_end_from_candles(candles_path: Path) -> str:
    """
    Returns last completed trading week end (last trading day in the last completed Mon–Fri window),
    based on candles_daily.parquet dates.
    """
    if not candles_path.exists():
        raise SystemExit(f"❌ candles file not found for auto week_end: {candles_path}")

    candles = pd.read_parquet(candles_path, columns=["date"])
    dmax = pd.to_datetime(candles["date"]).max().normalize()

    # If max candle is Mon-Thu, the current week is not complete yet → use previous week's window
    ref = dmax
    if int(ref.dayofweek) < 4:
        ref = ref - pd.Timedelta(days=7)

    mon = ref - pd.Timedelta(days=int(ref.dayofweek))
    fri = mon + pd.Timedelta(days=4)

    dates = pd.to_datetime(candles["date"]).dt.normalize()
    wdays = dates[(dates >= mon) & (dates <= fri)]
    if wdays.empty:
        raise SystemExit(f"❌ Could not find candle dates for week window {mon.date()}..{fri.date()}")

    return wdays.max().strftime("%Y-%m-%d")


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

FRIDAY - P&L Recording:
  python -m src.weekly_workflow --week_end YYYY-MM-DD --mode friday_close
  
  This will:
  1. For TRADE weeks: Calculate returns using Monday open → Friday close from candles
  2. For SKIP weeks: Log SKIP row with zeros
  3. Calculate SPY benchmark returns
  4. Log to weekly_pnl.csv

ANYTIME - Performance Analysis:
  python -m src.analyze_performance
  
  Shows hit rate, returns, drawdowns, skip rate, etc.
        """
    )
    
    ap.add_argument("--week_end", default="auto", help="Week ending date YYYY-MM-DD (or 'auto')")
    ap.add_argument(
        "--mode",
        required=True,
        choices=["friday", "monday", "friday_close"],
        help="Workflow stage: friday (signal), monday (execute), friday_close (record P&L)"
    )
    ap.add_argument("--execution_date", help="Trade execution date (for monday mode)")
    ap.add_argument("--account_value", type=float, help="Account value (for monday mode)")
    ap.add_argument("--max_clusters_per_symbol", type=int, default=1)
    
    args = ap.parse_args()

    if args.week_end == "auto":
        candles_path = Path("data/derived/market_daily/candles_daily.parquet")
        args.week_end = compute_auto_week_end_from_candles(candles_path)
        print(f"\n🗓️  auto week_end resolved to: {args.week_end} (from {candles_path})")
    
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
        # Friday: Record P&L (automated from candles)
        print("\n📊 FRIDAY CLOSE: P&L Recording")
        print(f"   Week ending: {args.week_end}")
        
        # Check if basket exists
        basket_path = Path(f"data/derived/baskets/week_ending={args.week_end}/basket.csv")
        if not basket_path.exists():
            print(f"❌ Basket not found: {basket_path}")
            print("   Run --mode friday first to generate signals")
            sys.exit(1)
        
        # Update P&L (auto-calculates from candles for TRADE or logs SKIP)
        run_command(
            [py, "-m", "src.update_weekly_pnl",
             "--week_end", args.week_end],
            "Auto-calculating weekly P&L from candles"
        )
        
        # Generate scoreboard
        run_command(
            [py, "-m", "src.scoreboard"],
            "Generating performance scoreboard"
        )
        
        print("\n" + "="*70)
        print("✅ FRIDAY CLOSE workflow complete!")
        print("="*70)
        print("\n📋 Week complete! View results:")
        print("   cat data/live/weekly_pnl.csv")
        print("   cat data/live/scoreboard.csv")
        print("   cat data/derived/scoreboards/latest_scoreboard.md")
        print("\n📊 Deep analysis anytime:")
        print("   python -m src.analyze_performance\n")


if __name__ == "__main__":
    main()
