"""
Ingest daily market candles for universe + SPY using Finnhub API.
Stores results in data/derived/market_daily/candles_daily.parquet
"""

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


def fetch_candles(symbol: str, from_ts: int, to_ts: int, api_key: str) -> pd.DataFrame:
    """Fetch daily candles from Finnhub API."""
    url = "https://finnhub.io/api/v1/stock/candle"
    headers = {"X-Finnhub-Token": api_key}
    params = {
        "symbol": symbol,
        "resolution": "D",
        "from": from_ts,
        "to": to_ts
    }
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    
    data = response.json()
    
    if data.get("s") == "no_data":
        print(f"  ⚠ No data for {symbol}")
        return pd.DataFrame()
    
    if data.get("s") != "ok":
        print(f"  ⚠ Error for {symbol}: {data}")
        return pd.DataFrame()
    
    df = pd.DataFrame({
        "symbol": symbol,
        "date": pd.to_datetime(data["t"], unit="s").date,
        "o": data["o"],
        "h": data["h"],
        "l": data["l"],
        "c": data["c"],
        "v": data["v"]
    })
    
    return df


def main(universe_path: str, week_end: str):
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("FINNHUB_API_KEY environment variable not set")
    
    # Parse week_end and determine date range
    week_end_date = datetime.strptime(week_end, "%Y-%m-%d").date()
    
    # Fetch data for last 90 days (enough for weekly analysis)
    from_date = week_end_date - timedelta(days=90)
    to_date = week_end_date
    
    from_ts = int(datetime.combine(from_date, datetime.min.time()).timestamp())
    to_ts = int(datetime.combine(to_date, datetime.max.time()).timestamp())
    
    # Load universe
    universe_df = pd.read_csv(universe_path)
    
    if "symbol" not in universe_df.columns:
        raise ValueError(f"Universe file must have 'symbol' column: {universe_path}")
    
    symbols = list(universe_df["symbol"].unique())
    
    # Always include SPY for baseline
    if "SPY" not in symbols:
        symbols.append("SPY")
    
    print(f"📊 Fetching candles for {len(symbols)} symbols from {from_date} to {to_date}")
    
    all_candles = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {symbol}...", end=" ")
        
        try:
            df = fetch_candles(symbol, from_ts, to_ts, api_key)
            if not df.empty:
                all_candles.append(df)
                print(f"✓ {len(df)} bars")
            else:
                print("(skipped)")
            
            # Rate limit: Finnhub free tier is 60 calls/min
            if i % 50 == 0:
                print(f"  💤 Rate limit pause...")
                time.sleep(1)
            else:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    if not all_candles:
        raise RuntimeError("No candle data fetched")
    
    # Combine all data
    combined = pd.concat(all_candles, ignore_index=True)
    
    # Ensure date is datetime for parquet
    combined["date"] = pd.to_datetime(combined["date"])
    
    # Sort by symbol and date
    combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
    
    # Save to parquet
    output_dir = Path("data/derived/market_daily")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "candles_daily.parquet"
    
    combined.to_parquet(output_path, index=False)
    
    print(f"\n✅ Saved {len(combined):,} candle records to {output_path}")
    print(f"   Symbols: {combined['symbol'].nunique()}")
    print(f"   Date range: {combined['date'].min()} to {combined['date'].max()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Ingest market candles from Finnhub")
    ap.add_argument("--universe", required=True, help="Path to universe CSV with 'symbol' column")
    ap.add_argument("--week_end", required=True, help="Week ending date YYYY-MM-DD")
    args = ap.parse_args()
    
    main(args.universe, args.week_end)
