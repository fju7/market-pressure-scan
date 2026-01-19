"""
Analyze live trading performance metrics

After 8-12 weeks, run this to see:
- Hit rate (positive vs negative weeks)
- Average return per trade
- Active return vs SPY
- Drawdowns
- % of weeks skipped
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def analyze_performance(min_weeks: int = 1) -> None:
    """
    Analyze live trading performance from ledgers
    
    Parameters
    ----------
    min_weeks : int
        Minimum number of weeks required for analysis
    """
    weeks_log_path = Path("data/live/weeks_log.csv")
    pnl_path = Path("data/live/weekly_pnl.csv")
    
    if not weeks_log_path.exists():
        print(f"❌ No weeks log found: {weeks_log_path}")
        return
    
    weeks = pd.read_csv(weeks_log_path)
    
    print("=" * 70)
    print("📊 LIVE TRADING PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Overall stats
    total_weeks = len(weeks)
    trade_weeks = len(weeks[weeks["action"] == "TRADE"])
    skip_weeks = len(weeks[weeks["action"] == "SKIP"])
    skip_rate = (skip_weeks / total_weeks * 100) if total_weeks > 0 else 0
    
    print(f"\n📅 WEEKS SUMMARY")
    print(f"   Total weeks tracked: {total_weeks}")
    print(f"   TRADE weeks: {trade_weeks}")
    print(f"   SKIP weeks: {skip_weeks}")
    print(f"   Skip rate: {skip_rate:.1f}%")
    
    if total_weeks < min_weeks:
        print(f"\n⚠️  Need at least {min_weeks} weeks for analysis (currently {total_weeks})")
        return
    
    # P&L analysis
    if not pnl_path.exists():
        print(f"\n⚠️  No P&L data yet: {pnl_path}")
        return
    
    pnl = pd.read_csv(pnl_path)
    
    if pnl.empty:
        print(f"\n⚠️  P&L file is empty")
        return
    
    # Weekly aggregation (average across positions in each week)
    weekly_pnl = pnl.groupby("week_ending_date").agg({
        "return_pct": "mean",
        "benchmark_return_pct": "first",  # Same for all positions in a week
        "active_return_pct": "mean",
        "symbol": "count",  # Number of positions
    }).rename(columns={"symbol": "num_positions"})
    
    completed_weeks = len(weekly_pnl)
    
    print(f"\n💰 P&L SUMMARY ({completed_weeks} weeks with results)")
    print(f"   Average positions per week: {weekly_pnl['num_positions'].mean():.1f}")
    
    # Returns
    avg_return = weekly_pnl["return_pct"].mean()
    avg_bench = weekly_pnl["benchmark_return_pct"].mean()
    avg_active = weekly_pnl["active_return_pct"].mean()
    
    print(f"\n📈 RETURNS")
    print(f"   Average return: {avg_return:+.2f}%")
    print(f"   Average SPY return: {avg_bench:+.2f}%")
    print(f"   Average active return: {avg_active:+.2f}%")
    
    # Hit rate
    positive_weeks = len(weekly_pnl[weekly_pnl["return_pct"] > 0])
    negative_weeks = len(weekly_pnl[weekly_pnl["return_pct"] <= 0])
    hit_rate = (positive_weeks / completed_weeks * 100) if completed_weeks > 0 else 0
    
    positive_active_weeks = len(weekly_pnl[weekly_pnl["active_return_pct"] > 0])
    active_hit_rate = (positive_active_weeks / completed_weeks * 100) if completed_weeks > 0 else 0
    
    print(f"\n🎯 HIT RATE")
    print(f"   Positive weeks: {positive_weeks}/{completed_weeks} ({hit_rate:.1f}%)")
    print(f"   Negative weeks: {negative_weeks}/{completed_weeks}")
    print(f"   Weeks beating SPY: {positive_active_weeks}/{completed_weeks} ({active_hit_rate:.1f}%)")
    
    # Risk metrics
    return_std = weekly_pnl["return_pct"].std()
    active_std = weekly_pnl["active_return_pct"].std()
    sharpe = (avg_return / return_std) if return_std > 0 else 0
    info_ratio = (avg_active / active_std) if active_std > 0 else 0
    
    print(f"\n📊 RISK METRICS")
    print(f"   Return volatility: {return_std:.2f}%")
    print(f"   Active volatility: {active_std:.2f}%")
    print(f"   Sharpe ratio (weekly): {sharpe:.2f}")
    print(f"   Information ratio: {info_ratio:.2f}")
    
    # Drawdown analysis (cumulative returns)
    weekly_pnl = weekly_pnl.sort_index()
    cum_returns = (1 + weekly_pnl["return_pct"] / 100).cumprod() - 1
    running_max = cum_returns.expanding().max()
    drawdown = cum_returns - running_max
    max_dd = drawdown.min() * 100
    
    print(f"\n📉 DRAWDOWN")
    print(f"   Max drawdown: {max_dd:.2f}%")
    print(f"   Current cumulative return: {cum_returns.iloc[-1] * 100:+.2f}%")
    
    # Best/worst weeks
    best_week = weekly_pnl.nlargest(1, "return_pct")
    worst_week = weekly_pnl.nsmallest(1, "return_pct")
    
    print(f"\n🏆 BEST/WORST WEEKS")
    if not best_week.empty:
        print(f"   Best: {best_week.index[0]} ({best_week['return_pct'].iloc[0]:+.2f}%)")
    if not worst_week.empty:
        print(f"   Worst: {worst_week.index[0]} ({worst_week['return_pct'].iloc[0]:+.2f}%)")
    
    # Signal quality correlation
    trade_weeks_list = weeks[weeks["action"] == "TRADE"]["week_ending_date"].tolist()
    weeks_with_pnl = weekly_pnl.index.tolist()
    
    # Match weeks
    matched_weeks = weeks[weeks["week_ending_date"].isin(weeks_with_pnl)].copy()
    
    if not matched_weeks.empty:
        matched_pnl = weekly_pnl.loc[matched_weeks["week_ending_date"]].copy()
        matched_weeks = matched_weeks.set_index("week_ending_date")
        
        # Correlation: low recap % should lead to better returns
        if "recap_pct" in matched_weeks.columns and len(matched_weeks) > 2:
            recap_corr = matched_weeks["recap_pct"].corr(matched_pnl["return_pct"])
            print(f"\n🔍 SIGNAL QUALITY")
            print(f"   Recap % vs return correlation: {recap_corr:.2f}")
            print(f"   (Negative = good: low recap → better returns)")
    
    print("\n" + "=" * 70)
    
    # Weekly details table
    if completed_weeks <= 12:
        print(f"\n📋 WEEKLY DETAILS")
        display_pnl = weekly_pnl.copy()
        display_pnl["return_pct"] = display_pnl["return_pct"].apply(lambda x: f"{x:+.2f}%")
        display_pnl["active_return_pct"] = display_pnl["active_return_pct"].apply(lambda x: f"{x:+.2f}%")
        print(display_pnl.to_string())
    
    print("\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Analyze live trading performance after 8-12 weeks"
    )
    ap.add_argument(
        "--min_weeks",
        type=int,
        default=1,
        help="Minimum number of weeks required (default: 1)",
    )
    
    args = ap.parse_args()
    
    analyze_performance(args.min_weeks)
