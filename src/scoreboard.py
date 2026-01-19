"""
Lightweight performance scoreboard

Reads weekly_pnl.csv and generates:
- data/live/scoreboard.csv (single-row summary)
- data/derived/scoreboards/latest_scoreboard.md (formatted report)

Shows strategy trajectory in 10 seconds.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_scoreboard(pnl_path: Path) -> dict:
    """
    Calculate performance metrics from weekly_pnl.csv
    
    Parameters
    ----------
    pnl_path : Path
        Path to weekly_pnl.csv
    
    Returns
    -------
    dict
        Performance metrics
    """
    if not pnl_path.exists():
        raise FileNotFoundError(f"Weekly P&L file not found: {pnl_path}")
    
    df = pd.read_csv(pnl_path)
    
    if df.empty:
        return {
            "total_weeks": 0,
            "trade_weeks": 0,
            "skip_weeks": 0,
            "cum_basket_return_pct": 0.0,
            "cum_spy_return_pct": 0.0,
            "cum_active_return_pct": 0.0,
            "hit_rate_pct": 0.0,
            "avg_active_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "time_in_market_pct": 0.0,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Count weeks
    weeks = df.groupby("week_ending_date").first()
    total_weeks = len(weeks)
    skip_weeks = (weeks["symbol"] == "SKIP").sum()
    trade_weeks = total_weeks - skip_weeks
    
    # For TRADE weeks, aggregate by week (equal-weight positions)
    trade_df = df[df["symbol"] != "SKIP"].copy()
    
    if trade_df.empty:
        # All SKIP weeks
        return {
            "total_weeks": total_weeks,
            "trade_weeks": 0,
            "skip_weeks": skip_weeks,
            "cum_basket_return_pct": 0.0,
            "cum_spy_return_pct": 0.0,
            "cum_active_return_pct": 0.0,
            "hit_rate_pct": 0.0,
            "avg_active_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "time_in_market_pct": 0.0,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Weekly aggregation (equal-weight basket)
    weekly = trade_df.groupby("week_ending_date").agg({
        "return_pct": "mean",  # Equal-weight basket return
        "benchmark_return_pct": "first",  # SPY is same for all positions
        "active_return_pct": "mean"  # Equal-weight active return
    }).reset_index()
    
    # Cumulative returns (compound)
    weekly["cum_basket"] = (1 + weekly["return_pct"] / 100).cumprod()
    weekly["cum_spy"] = (1 + weekly["benchmark_return_pct"] / 100).cumprod()
    
    cum_basket_return = (weekly["cum_basket"].iloc[-1] - 1) * 100
    cum_spy_return = (weekly["cum_spy"].iloc[-1] - 1) * 100
    cum_active_return = cum_basket_return - cum_spy_return
    
    # Hit rate (% of TRADE weeks with positive basket return)
    hit_rate = (weekly["return_pct"] > 0).sum() / len(weekly) * 100 if len(weekly) > 0 else 0.0
    
    # Average weekly active return
    avg_active_return = weekly["active_return_pct"].mean()
    
    # Max drawdown (basket)
    weekly["peak"] = weekly["cum_basket"].cummax()
    weekly["drawdown"] = (weekly["cum_basket"] / weekly["peak"] - 1) * 100
    max_drawdown = weekly["drawdown"].min()
    
    # Time in market
    time_in_market = trade_weeks / total_weeks * 100 if total_weeks > 0 else 0.0
    
    return {
        "total_weeks": total_weeks,
        "trade_weeks": trade_weeks,
        "skip_weeks": skip_weeks,
        "cum_basket_return_pct": cum_basket_return,
        "cum_spy_return_pct": cum_spy_return,
        "cum_active_return_pct": cum_active_return,
        "hit_rate_pct": hit_rate,
        "avg_active_return_pct": avg_active_return,
        "max_drawdown_pct": max_drawdown,
        "time_in_market_pct": time_in_market,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def write_scoreboard_csv(metrics: dict, output_path: Path):
    """Write scoreboard to CSV"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([metrics])
    df.to_csv(output_path, index=False)
    
    print(f"✓ Scoreboard CSV written: {output_path}")


def write_scoreboard_md(metrics: dict, output_path: Path):
    """Write formatted markdown scoreboard"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    md = f"""# Performance Scoreboard
**Last Updated:** {metrics['last_updated']}

## Summary

| Metric | Value |
|--------|-------|
| **Total Weeks** | {metrics['total_weeks']} |
| **TRADE Weeks** | {metrics['trade_weeks']} |
| **SKIP Weeks** | {metrics['skip_weeks']} |
| **Time in Market** | {metrics['time_in_market_pct']:.1f}% |

## Returns

| Metric | Value |
|--------|-------|
| **Cumulative Basket Return** | {metrics['cum_basket_return_pct']:+.2f}% |
| **Cumulative SPY Return** | {metrics['cum_spy_return_pct']:+.2f}% |
| **Cumulative Active Return** | {metrics['cum_active_return_pct']:+.2f}% |
| **Avg Weekly Active Return** | {metrics['avg_active_return_pct']:+.2f}% |

## Risk Metrics

| Metric | Value |
|--------|-------|
| **Hit Rate** (TRADE weeks) | {metrics['hit_rate_pct']:.1f}% |
| **Max Drawdown** (basket) | {metrics['max_drawdown_pct']:.2f}% |

## Quick Assessment

"""
    
    # Add quick assessment
    if metrics['total_weeks'] < 8:
        md += "⏳ **Early days** - Need 8-12 weeks for meaningful conclusions\n\n"
    else:
        if metrics['cum_active_return_pct'] > 0:
            md += "✅ **Beating benchmark** - Active return positive\n"
        else:
            md += "⚠️ **Underperforming** - Active return negative\n"
        
        if metrics['hit_rate_pct'] >= 50:
            md += f"✅ **Hit rate acceptable** - {metrics['hit_rate_pct']:.1f}% winning weeks\n"
        else:
            md += f"⚠️ **Low hit rate** - {metrics['hit_rate_pct']:.1f}% winning weeks\n"
        
        if metrics['time_in_market_pct'] < 50:
            md += f"📊 **Selective filter** - Only trading {metrics['time_in_market_pct']:.1f}% of weeks\n"
        else:
            md += f"📊 **Active approach** - Trading {metrics['time_in_market_pct']:.1f}% of weeks\n"
    
    md += f"\n---\n*Based on {metrics['trade_weeks']} TRADE weeks and {metrics['skip_weeks']} SKIP weeks*\n"
    
    output_path.write_text(md)
    print(f"✓ Scoreboard markdown written: {output_path}")


def display_scoreboard(metrics: dict):
    """Display scoreboard to terminal"""
    print("\n" + "="*70)
    print("📊 PERFORMANCE SCOREBOARD")
    print("="*70)
    print(f"\n📅 Weeks: {metrics['total_weeks']} total ({metrics['trade_weeks']} TRADE, {metrics['skip_weeks']} SKIP)")
    print(f"⏱️  Time in Market: {metrics['time_in_market_pct']:.1f}%")
    print(f"\n💰 Returns:")
    print(f"   Basket:  {metrics['cum_basket_return_pct']:+7.2f}%")
    print(f"   SPY:     {metrics['cum_spy_return_pct']:+7.2f}%")
    print(f"   Active:  {metrics['cum_active_return_pct']:+7.2f}%")
    print(f"\n📈 Performance:")
    print(f"   Hit Rate:        {metrics['hit_rate_pct']:.1f}%")
    print(f"   Avg Active/Week: {metrics['avg_active_return_pct']:+.2f}%")
    print(f"   Max Drawdown:    {metrics['max_drawdown_pct']:.2f}%")
    print(f"\n🕐 Last Updated: {metrics['last_updated']}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate performance scoreboard")
    parser.add_argument(
        "--pnl_path",
        type=str,
        default="data/live/weekly_pnl.csv",
        help="Path to weekly_pnl.csv"
    )
    parser.add_argument(
        "--skip_markdown",
        action="store_true",
        help="Skip markdown output (CSV only)"
    )
    args = parser.parse_args()
    
    pnl_path = Path(args.pnl_path)
    
    # Calculate metrics
    metrics = calculate_scoreboard(pnl_path)
    
    # Display to terminal
    display_scoreboard(metrics)
    
    # Write CSV
    csv_path = Path("data/live/scoreboard.csv")
    write_scoreboard_csv(metrics, csv_path)
    
    # Write markdown (optional)
    if not args.skip_markdown:
        md_path = Path("data/derived/scoreboards/latest_scoreboard.md")
        write_scoreboard_md(metrics, md_path)


if __name__ == "__main__":
    main()
