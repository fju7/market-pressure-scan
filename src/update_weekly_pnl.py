"""
Auto-compute weekly P&L from candles data (Friday close)

For TRADE weeks:
- Reads basket.csv to get positions
- Pulls Monday open (entry) and Friday close (exit) from candles
- Computes equal-weight basket return
- Computes SPY benchmark return
- Appends to weekly_pnl.csv

For SKIP weeks:
- Appends SKIP row with zeros and skip reason
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def get_monday_friday_dates(week_end: str) -> tuple[str, str]:
    """
    Get Monday and Friday dates for a given week ending.
    
    Parameters
    ----------
    week_end : str
        Week ending date (YYYY-MM-DD), typically a Friday
    
    Returns
    -------
    tuple[str, str]
        Monday date, Friday date (YYYY-MM-DD)
    """
    friday = pd.Timestamp(week_end)
    monday = friday - pd.Timedelta(days=4)  # Friday - 4 days = Monday
    return monday.strftime("%Y-%m-%d"), friday.strftime("%Y-%m-%d")


def get_price_from_candles(symbol: str, date: str, price_type: str = "open") -> float | None:
    """
    Get price from candles_daily.parquet
    
    Parameters
    ----------
    symbol : str
        Stock symbol
    date : str
        Date (YYYY-MM-DD)
    price_type : str
        "open" or "close"
    
    Returns
    -------
    float | None
        Price or None if not found
    """
    candles_path = Path("data/derived/market_daily/candles_daily.parquet")
    if not candles_path.exists():
        return None
    
    df = pd.read_parquet(candles_path)
    df = df[(df["symbol"] == symbol) & (df["date"] == date)]
    
    if df.empty:
        return None
    
    if price_type == "open":
        return df["o"].iloc[0]
    elif price_type == "close":
        return df["c"].iloc[0]
    else:
        return None


def update_weekly_pnl(week_end: str) -> None:
    """
    Update weekly_pnl.csv with P&L calculation
    
    Parameters
    ----------
    week_end : str
        Week ending date (YYYY-MM-DD)
    """
    # Read basket
    basket_path = Path(f"data/derived/baskets/week_ending={week_end}/basket.csv")
    if not basket_path.exists():
        raise FileNotFoundError(f"Basket not found: {basket_path}")
    
    basket = pd.read_csv(basket_path)
    action = basket["action"].iloc[0]
    
    # Get Monday and Friday dates
    monday_date, friday_date = get_monday_friday_dates(week_end)
    
    # Initialize weekly_pnl.csv if needed
    pnl_path = Path("data/live/weekly_pnl.csv")
    pnl_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not pnl_path.exists():
        pnl_df = pd.DataFrame(columns=[
            "week_ending_date", "symbol", "entry_price", "exit_price",
            "return_pct", "benchmark_return_pct", "active_return_pct", "notes"
        ])
        pnl_df.to_csv(pnl_path, index=False)
    
    # Read existing weekly_pnl
    pnl_df = pd.read_csv(pnl_path)
    
    # Check if week already logged
    if week_end in pnl_df["week_ending_date"].values:
        print(f"Week {week_end} already logged in weekly_pnl.csv. Skipping.")
        return
    
    if action == "SKIP":
        # Log SKIP week
        skip_reason = basket["notes"].iloc[0] if "notes" in basket.columns else "Low information week"
        skip_row = {
            "week_ending_date": week_end,
            "symbol": "SKIP",
            "entry_price": 0.0,
            "exit_price": 0.0,
            "return_pct": 0.0,
            "benchmark_return_pct": 0.0,
            "active_return_pct": 0.0,
            "notes": skip_reason
        }
        pnl_df = pd.concat([pnl_df, pd.DataFrame([skip_row])], ignore_index=True)
        pnl_df.to_csv(pnl_path, index=False)
        print(f"✓ Week {week_end}: SKIP logged (reason: {skip_reason})")
        return
    
    # TRADE week - calculate P&L
    positions = basket[basket["action"] == "BUY"].copy()
    
    if positions.empty:
        print(f"No BUY positions in basket for week {week_end}")
        return
    
    # Calculate returns for each position
    returns = []
    for _, row in positions.iterrows():
        symbol = row["symbol"]
        entry_price = get_price_from_candles(symbol, monday_date, "open")
        exit_price = get_price_from_candles(symbol, friday_date, "close")
        
        if entry_price is None or exit_price is None:
            print(f"⚠️  Missing candle data for {symbol}: entry={entry_price}, exit={exit_price}")
            continue
        
        ret_pct = ((exit_price - entry_price) / entry_price) * 100
        returns.append({
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return_pct": ret_pct
        })
    
    if not returns:
        print(f"No valid returns calculated for week {week_end}")
        return
    
    # Calculate benchmark return (SPY)
    spy_entry = get_price_from_candles("SPY", monday_date, "open")
    spy_exit = get_price_from_candles("SPY", friday_date, "close")
    
    if spy_entry is None or spy_exit is None:
        print(f"⚠️  Missing SPY candle data: entry={spy_entry}, exit={spy_exit}")
        benchmark_return = 0.0
    else:
        benchmark_return = ((spy_exit - spy_entry) / spy_entry) * 100
    
    # Log each position
    for ret in returns:
        active_return = ret["return_pct"] - benchmark_return
        log_row = {
            "week_ending_date": week_end,
            "symbol": ret["symbol"],
            "entry_price": ret["entry_price"],
            "exit_price": ret["exit_price"],
            "return_pct": ret["return_pct"],
            "benchmark_return_pct": benchmark_return,
            "active_return_pct": active_return,
            "notes": f"Auto-calculated from candles (Mon open → Fri close)"
        }
        pnl_df = pd.concat([pnl_df, pd.DataFrame([log_row])], ignore_index=True)
    
    # Save
    pnl_df.to_csv(pnl_path, index=False)
    
    # Calculate and display summary
    basket_return = sum(r["return_pct"] for r in returns) / len(returns)
    avg_active = basket_return - benchmark_return
    
    print(f"✓ Week {week_end}: TRADE P&L logged")
    print(f"  Positions: {len(returns)}")
    print(f"  Basket return: {basket_return:.2f}%")
    print(f"  SPY return: {benchmark_return:.2f}%")
    print(f"  Active return: {avg_active:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Update weekly P&L from candles")
    parser.add_argument(
        "--week_end",
        type=str,
        required=True,
        help="Week ending date (YYYY-MM-DD)"
    )
    args = parser.parse_args()
    
    update_weekly_pnl(args.week_end)


if __name__ == "__main__":
    main()
