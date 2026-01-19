"""
Log weekly P&L from executed trades

This script:
1. Reads trades_log.csv to get entry prices for a given week
2. Fetches exit prices (Friday close or current price) from Finnhub
3. Calculates returns vs SPY benchmark
4. Appends to data/live/weekly_pnl.csv
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests


def fetch_quote(symbol: str, api_key: str) -> dict | None:
    """
    Fetch current quote from Finnhub
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol
    api_key : str
        Finnhub API key
    
    Returns
    -------
    dict or None
        Quote data with keys: c (current), o (open), h (high), l (low), pc (prev close), t (timestamp)
    """
    url = "https://finnhub.io/api/v1/quote"
    params = {"symbol": symbol, "token": api_key}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("c") and data["c"] > 0:
            return data
        else:
            print(f"⚠️  {symbol}: Invalid quote data from Finnhub: {data}")
            return None
            
    except Exception as e:
        print(f"⚠️  {symbol}: Failed to fetch quote: {e}")
        return None


def log_weekly_pnl(
    week_end: str,
    exit_date: str,
    notes: str = "",
) -> None:
    """
    Log weekly P&L for positions from a given week
    
    Parameters
    ----------
    week_end : str
        Week ending date (YYYY-MM-DD)
    exit_date : str
        Exit date for P&L calculation (YYYY-MM-DD), typically Friday
    notes : str
        Additional notes
    """
    # Load trades log
    trades_path = Path("data/live/trades_log.csv")
    if not trades_path.exists():
        raise FileNotFoundError(f"Trades log not found: {trades_path}")
    
    trades = pd.read_csv(trades_path)
    
    # Find trades for this week (look for week_end in notes field)
    week_trades = trades[trades["notes"].str.contains(f"week ending {week_end}", na=False)].copy()
    
    if week_trades.empty:
        print(f"⏭️  No trades found for week ending {week_end}")
        return
    
    # Group by symbol and get entry price (should be one entry per symbol per week)
    # Use entry_price column
    symbol_entries = {}
    for _, row in week_trades.iterrows():
        symbol = row["symbol"]
        entry_price = row.get("entry_price")
        
        if pd.notna(entry_price) and entry_price > 0:
            symbol_entries[symbol] = float(entry_price)
        else:
            print(f"⚠️  {symbol}: Missing or invalid entry price in trades_log")
    
    if not symbol_entries:
        print(f"⏭️  No valid entry prices found for week ending {week_end}")
        return
    
    # Load market candles for exit prices
    candles_path = Path("data/derived/market_daily/candles_daily.parquet")
    if not candles_path.exists():
        raise FileNotFoundError(f"Market candles not found: {candles_path}")
    
    candles = pd.read_parquet(candles_path)
    candles["date"] = pd.to_datetime(candles["date"])
    exit_dt = pd.to_datetime(exit_date)
    trade_dt = pd.to_datetime(week_trades.iloc[0]["entry_date"])
    
    print(f"📊 Calculating P&L for {len(symbol_entries)} symbols...")
    
    pnl_entries = []
    symbols = list(symbol_entries.keys())
    
    for i, symbol in enumerate(symbols):
        entry_price = symbol_entries[symbol]
        
        # Get exit price from candles (close price on exit_date)
        symbol_candles = candles[candles["symbol"] == symbol].copy()
        exit_candle = symbol_candles[symbol_candles["date"] == exit_dt]
        
        if not exit_candle.empty:
            exit_price = exit_candle.iloc[0]["close"]
            return_pct = ((exit_price - entry_price) / entry_price) * 100
            
            pnl_entries.append({
                "symbol": symbol,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": return_pct,
            })
            
            print(f"  [{i+1}/{len(symbols)}] {symbol}: ${entry_price:.2f} → ${exit_price:.2f} ({return_pct:+.2f}%)")
        else:
            print(f"  [{i+1}/{len(symbols)}] {symbol}: ❌ No candle data for {exit_date}")
    
    if not pnl_entries:
        print("⏭️  No valid exit prices found in candles data")
        return
    
    # Calculate SPY benchmark return
    print("\n📊 Calculating SPY benchmark return...")
    spy_candles = candles[candles["symbol"] == "SPY"].copy()
    
    # Get SPY price on trade date (entry)
    spy_entry = spy_candles[spy_candles["date"] <= trade_dt].tail(1)
    spy_exit = spy_candles[spy_candles["date"] == exit_dt]
    
    benchmark_return_pct = 0.0
    
    if not spy_entry.empty and not spy_exit.empty:
        spy_entry_price = spy_entry.iloc[0]["close"]
        spy_exit_price = spy_exit.iloc[0]["close"]
        benchmark_return_pct = ((spy_exit_price - spy_entry_price) / spy_entry_price) * 100
        print(f"  SPY: ${spy_entry_price:.2f} → ${spy_exit_price:.2f} ({benchmark_return_pct:+.2f}%)")
    else:
        print("  ⚠️  SPY benchmark return unavailable (missing candle data)")
    
    # Add benchmark return and active return to all entries
    for entry in pnl_entries:
        entry["week_ending_date"] = week_end
        entry["benchmark_return_pct"] = benchmark_return_pct
        entry["active_return_pct"] = entry["return_pct"] - benchmark_return_pct
        entry["notes"] = notes
    
    # Create DataFrame
    pnl_df = pd.DataFrame(pnl_entries)
    
    # Reorder columns
    pnl_df = pnl_df[[
        "week_ending_date",
        "symbol",
        "entry_price",
        "exit_price",
        "return_pct",
        "benchmark_return_pct",
        "active_return_pct",
        "notes",
    ]]
    
    # Append to weekly_pnl.csv
    pnl_path = Path("data/live/weekly_pnl.csv")
    pnl_path.parent.mkdir(parents=True, exist_ok=True)
    
    if pnl_path.exists():
        # Append mode
        pnl_df.to_csv(pnl_path, mode="a", header=False, index=False)
        print(f"\n✅ Appended {len(pnl_df)} P&L record(s) to {pnl_path}")
    else:
        # Create new file with header
        pnl_df.to_csv(pnl_path, index=False)
        print(f"\n✅ Created {pnl_path} with {len(pnl_df)} P&L record(s)")
    
    # Display summary
    print("\nWeekly P&L summary:")
    print(pnl_df.to_string(index=False))
    
    avg_return = pnl_df["return_pct"].mean()
    avg_active = pnl_df["active_return_pct"].mean()
    print(f"\nAverage return: {avg_return:+.2f}%")
    print(f"Average active return: {avg_active:+.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Log weekly P&L from trades_log.csv"
    )
    ap.add_argument("--week_end", required=True, help="Week ending date YYYY-MM-DD")
    ap.add_argument(
        "--exit_date",
        required=True,
        help="Exit date for P&L calculation YYYY-MM-DD (typically Friday)",
    )
    ap.add_argument(
        "--notes",
        default="",
        help="Additional notes for this P&L record",
    )
    
    args = ap.parse_args()
    
    log_weekly_pnl(
        week_end=args.week_end,
        exit_date=args.exit_date,
        notes=args.notes,
    )
