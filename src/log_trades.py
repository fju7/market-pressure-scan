"""
Log executed trades from basket.csv to trades_log.csv

This script:
1. Reads basket.csv for a given week
2. If action=TRADE, appends one row per symbol to data/live/trades_log.csv
3. Calculates shares based on account value and target weights
4. Fetches live prices from Finnhub if not provided
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


def fetch_live_price(symbol: str, api_key: str) -> float | None:
    """
    Fetch current quote price from Finnhub
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol
    api_key : str
        Finnhub API key
    
    Returns
    -------
    float or None
        Current price (c field), or None if error
    """
    url = "https://finnhub.io/api/v1/quote"
    params = {"symbol": symbol, "token": api_key}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # Finnhub quote endpoint returns: c (current), h (high), l (low), o (open), pc (prev close), t (timestamp)
        current_price = data.get("c")
        
        if current_price and current_price > 0:
            return float(current_price)
        else:
            print(f"⚠️  {symbol}: Invalid price data from Finnhub: {data}")
            return None
            
    except Exception as e:
        print(f"⚠️  {symbol}: Failed to fetch price: {e}")
        return None


def fetch_live_prices(symbols: list[str], api_key: str) -> dict[str, float]:
    """
    Fetch live prices for multiple symbols with rate limiting
    
    Parameters
    ----------
    symbols : list of str
        List of stock tickers
    api_key : str
        Finnhub API key
    
    Returns
    -------
    dict
        Symbol -> price mapping
    """
    prices = {}
    
    for i, symbol in enumerate(symbols):
        price = fetch_live_price(symbol, api_key)
        if price:
            prices[symbol] = price
            print(f"  [{i+1}/{len(symbols)}] {symbol}: ${price:.2f}")
        else:
            print(f"  [{i+1}/{len(symbols)}] {symbol}: ❌ Price unavailable")
        
        # Rate limiting: Finnhub free tier = 60 calls/min
        if i < len(symbols) - 1:
            time.sleep(1.1)  # ~55 calls/min to be safe
    
    return prices


def log_trades(
    week_end: str,
    execution_date: str,
    execution_time: str,
    account_value: float,
    fill_prices: dict[str, float] | None = None,
    fee_per_trade: float = 0.0,
    side: str = "BUY",
    fetch_prices: bool = True,
) -> None:
    """
    Log trades from basket to trades_log.csv
    
    Parameters
    ----------
    week_end : str
        Week ending date (YYYY-MM-DD)
    execution_date : str
        Actual execution date (YYYY-MM-DD), typically Monday
    execution_time : str
        Execution time (HH:MM) - not currently logged
    account_value : float
        Total account value for position sizing
    fill_prices : dict, optional
        Symbol -> fill price mapping. If None and fetch_prices=True, fetches from Finnhub
    fee_per_trade : float
        Fee per trade execution - not currently logged
    side : str
        BUY or SELL
    fetch_prices : bool
        Whether to fetch live prices from Finnhub (default: True)
    """
    basket_path = Path(f"data/derived/baskets/week_ending={week_end}/basket.csv")
    if not basket_path.exists():
        raise FileNotFoundError(f"Basket not found: {basket_path}")
    
    basket = pd.read_csv(basket_path)
    
    # Check if this is a TRADE action
    if basket["action"].iloc[0] == "SKIP":
        print(f"⏭️  Week {week_end}: Action=SKIP, no trades to log")
        return
    
    # Filter to TRADE rows only
    trades = basket[basket["action"] == "TRADE"].copy()
    
    if trades.empty:
        print(f"⏭️  Week {week_end}: No TRADE rows found in basket")
        return
    
    # Fetch live prices if not provided
    if fill_prices is None and fetch_prices:
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            raise ValueError("FINNHUB_API_KEY environment variable not set")
        
        symbols = trades["symbol"].tolist()
        print(f"📊 Fetching live prices for {len(symbols)} symbols...")
        fill_prices = fetch_live_prices(symbols, api_key)
    
    if fill_prices is None:
        fill_prices = {}
    
    # Prepare trades_log entries
    log_entries = []
    for _, row in trades.iterrows():
        symbol = row["symbol"]
        weight = row["weight"]
        
        # Calculate position size in dollars
        position_value = account_value * weight
        
        # Get fill price (if provided) or leave as placeholder
        fill_price = fill_prices.get(symbol) if fill_prices else None
        
        # Calculate shares (round down to whole shares)
        if fill_price and fill_price > 0:
            shares = int(position_value / fill_price)
        else:
            shares = None  # User must fill manually
        
        log_entries.append({
            "week_ending_date": week_end,
            "symbol": symbol,
            "action": side,
            "weight_target": weight,
            "entry_date": execution_date,
            "entry_price": fill_price,
            "shares": shares,
            "notes": f"Monday rebalance week ending {week_end}",
        })
    
    log_df = pd.DataFrame(log_entries)
    
    # Append to trades_log.csv
    log_path = Path("data/live/trades_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    if log_path.exists():
        # Append mode
        log_df.to_csv(log_path, mode="a", header=False, index=False)
        print(f"✅ Appended {len(log_df)} trade(s) to {log_path}")
    else:
        # Create new file with header
        log_df.to_csv(log_path, index=False)
        print(f"✅ Created {log_path} with {len(log_df)} trade(s)")
    
    # Display summary
    print("\nTrade log summary:")
    print(log_df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Log executed trades from basket.csv to trades_log.csv"
    )
    ap.add_argument("--week_end", required=True, help="Week ending date YYYY-MM-DD")
    ap.add_argument(
        "--execution_date",
        required=True,
        help="Trade execution date YYYY-MM-DD (typically Monday)",
    )
    ap.add_argument(
        "--execution_time",
        default="09:30",
        help="Trade execution time HH:MM (default: 09:30)",
    )
    ap.add_argument(
        "--account_value",
        type=float,
        required=True,
        help="Total account value for position sizing",
    )
    ap.add_argument(
        "--fee_per_trade",
        type=float,
        default=0.0,
        help="Fee per trade execution (default: 0.0)",
    )
    ap.add_argument(
        "--side",
        default="BUY",
        choices=["BUY", "SELL"],
        help="Trade side: BUY or SELL (default: BUY)",
    )
    ap.add_argument(
        "--no_fetch_prices",
        action="store_true",
        help="Do not fetch live prices from Finnhub (manual fill required)",
    )
    
    args = ap.parse_args()
    
    log_trades(
        week_end=args.week_end,
        execution_date=args.execution_date,
        execution_time=args.execution_time,
        account_value=args.account_value,
        fee_per_trade=args.fee_per_trade,
        side=args.side,
        fill_prices=None,
        fetch_prices=not args.no_fetch_prices,
    )
