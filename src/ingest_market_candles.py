"""
Ingest daily market candles for universe + SPY using Finnhub API.
Stores results in data/derived/market_daily/candles_daily.parquet
"""

import argparse
from .reuse import should_skip
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.io_atomic import write_parquet_atomic


def fetch_candles(symbol: str, from_ts: int, to_ts: int, api_key: str, max_retries: int = 3) -> pd.DataFrame:
    """Fetch daily candles from Finnhub API with retry logic for 429 errors."""
    url = "https://finnhub.io/api/v1/stock/candle"
    headers = {"X-Finnhub-Token": api_key}
    params = {
        "symbol": symbol,
        "resolution": "D",
        "from": from_ts,
        "to": to_ts
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params)
            
            # Handle 429 rate limit with exponential backoff
            if response.status_code == 429:
                wait_time = 60 * (2 ** attempt)  # 60s, 120s, 240s
                print(f"⚠ Rate limited (429), waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("s") == "no_data":
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
            
        except requests.exceptions.HTTPError as e:
            if attempt == max_retries - 1:
                raise
            print(f"  ⚠ HTTP error (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(5 * (attempt + 1))
    
    return pd.DataFrame()


def main(universe_path: str, week_end: str, force: bool = False):
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
    print(f"   Rate limit: ~1 call/second (60 calls/min for Finnhub free tier)\n")
    
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
            
            # Rate limit: Finnhub free tier is 60 calls/min = 1 call/sec
            # Use 1.1s to be safe and avoid bursts hitting the limit
            time.sleep(1.1)
                
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    if not all_candles:
        raise RuntimeError("No candle data fetched")
    
    # Combine all data
    combined = pd.concat(all_candles, ignore_index=True)
    
    # Normalize Finnhub OHLCV column names to canonical names
    rename_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    combined = combined.rename(columns=rename_map)
    
    # Ensure date is datetime for parquet
    combined["date"] = pd.to_datetime(combined["date"])
    
    # Sort by symbol and date
    combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
    
    # Pre-write verification (invariants check)
    print("\n🔍 Verifying candle integrity...")
    
    # Check 1: No duplicates on (symbol, date)
    dup_count = combined.duplicated(subset=["symbol", "date"]).sum()
    if dup_count > 0:
        raise RuntimeError(f"❌ Found {dup_count} duplicate (symbol, date) pairs!")
    print("   ✓ No duplicates on (symbol, date)")
    
    # Check 2: No null OHLCV values
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    if not set(ohlcv_cols).issubset(combined.columns):
        raise RuntimeError(f"❌ Missing OHLCV columns. Expected {ohlcv_cols}, found: {list(combined.columns)}")
    
    null_counts = combined[ohlcv_cols].isnull().sum()
    if null_counts.any():
        raise RuntimeError(f"❌ Null values found in OHLCV: {null_counts[null_counts > 0].to_dict()}")
    print("   ✓ No null OHLCV values")
    
    # Check 3: Date range includes requested window
    min_date = combined["date"].min()
    max_date = combined["date"].max()
    expected_min = pd.to_datetime(from_date)
    expected_max = pd.to_datetime(to_date)
    
    if min_date > expected_min or max_date < expected_max:
        print(f"   ⚠️  Date range warning: expected {expected_min} to {expected_max}, got {min_date} to {max_date}")
    else:
        print(f"   ✓ Date range covers requested window")
    
    print("✅ Candle integrity verified")
    
    # Save to parquet (atomic write to prevent corruption)
    output_dir = Path("data/derived/market_daily")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "candles_daily.parquet"

    if should_skip(output_path, force):
        print(f"SKIP: {output_path} exists and --force not set.")
        return

    write_parquet_atomic(combined, output_path)

    print(f"\n✓ Saved {len(combined):,} candle records to {output_path}")
    print(f"   Symbols: {combined['symbol'].nunique()}")
    print(f"   Date range: {combined['date'].min()} to {combined['date'].max()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Ingest market candles from Finnhub")
    ap.add_argument("--universe", required=True, help="Path to universe CSV with 'symbol' column")
    ap.add_argument("--week_end", required=True, help="Week ending date YYYY-MM-DD")
    ap.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = ap.parse_args()

    main(args.universe, args.week_end, force=args.force)
