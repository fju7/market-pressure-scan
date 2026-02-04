"""
Ingest daily market candles for universe + SPY using Finnhub API.

Behavior:
- If candles store does NOT exist: build a 90-day history ending at week_end.
- If candles store exists and --force is NOT set: fetch ONLY missing dates from
  (existing max_date + 1) through week_end, then upsert + dedupe.
- If --force is set: refetch a 90-day window ending at week_end and MERGE into store (safe).
- Use --replace_store for a destructive 90-day rebuild that replaces the store.

Store path:
  data/derived/market_daily/candles_daily.parquet
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.io_atomic import write_parquet_atomic


STORE_DIR = Path("data/derived/market_daily")
STORE_PATH = STORE_DIR / "candles_daily.parquet"


def _as_date(x) -> date:
    return pd.to_datetime(x).date()


def fetch_candles(symbol: str, from_ts: int, to_ts: int, api_key: str, max_retries: int = 3) -> pd.DataFrame:
    """Fetch daily candles from Finnhub API with retry logic for 429 errors."""
    url = "https://finnhub.io/api/v1/stock/candle"
    headers = {"X-Finnhub-Token": api_key}
    params = {"symbol": symbol, "resolution": "D", "from": from_ts, "to": to_ts}

    for attempt in range(max_retries):
        response = requests.get(url, headers=headers, params=params)

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

        df = pd.DataFrame(
            {
                "symbol": symbol,
                "date": pd.to_datetime(data["t"], unit="s"),
                "open": data["o"],
                "high": data["h"],
                "low": data["l"],
                "close": data["c"],
                "volume": data["v"],
            }
        )
        return df

    return pd.DataFrame()


def _load_existing_store(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # normalize
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _verify_candles(df: pd.DataFrame, expected_min: date, expected_max: date) -> None:
    print("\n🔍 Verifying candle integrity...")

    dup_count = df.duplicated(subset=["symbol", "date"]).sum()
    if dup_count > 0:
        raise RuntimeError(f"❌ Found {dup_count} duplicate (symbol, date) pairs!")
    print("   ✓ No duplicates on (symbol, date)")

    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [c for c in ohlcv_cols if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"❌ Missing OHLCV columns: {missing_cols}")

    null_counts = df[ohlcv_cols].isnull().sum()
    if null_counts.any():
        raise RuntimeError(f"❌ Null values found in OHLCV: {null_counts[null_counts > 0].to_dict()}")
    print("   ✓ No null OHLCV values")

    min_date = pd.to_datetime(df["date"]).min().date()
    max_date = pd.to_datetime(df["date"]).max().date()

    if min_date > expected_min or max_date < expected_max:
        print(f"   ⚠️  Date range warning: expected [{expected_min}, {expected_max}] got [{min_date}, {max_date}]")
    else:
        print("   ✓ Date range covers requested window")

    print("✅ Candle integrity verified")


def main(universe_path: str, week_end: str, force: bool = False, replace_store: bool = False):
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("FINNHUB_API_KEY environment variable not set")

    week_end_date = datetime.strptime(week_end, "%Y-%m-%d").date()

    # universe symbols
    universe_df = pd.read_csv(universe_path)
    if "symbol" not in universe_df.columns:
        raise ValueError(f"Universe file must have 'symbol' column: {universe_path}")

    symbols = (
        universe_df["symbol"].astype(str).str.upper().str.strip().dropna().unique().tolist()
    )
    if "SPY" not in symbols:
        symbols.append("SPY")

    STORE_DIR.mkdir(parents=True, exist_ok=True)

    existing = _load_existing_store(STORE_PATH)

    if existing is None or existing.empty:
        # no store yet -> build 90d ending week_end
        from_date = week_end_date - timedelta(days=90)
        to_date = week_end_date
        mode = "FULL (90d bootstrap)"
    else:
        existing_max = pd.to_datetime(existing["date"]).max().date()

        if force:
            # IMPORTANT: force must NOT shrink history. We refetch 90d window and merge.
            from_date = week_end_date - timedelta(days=90)
            to_date = week_end_date
            mode = "REFETCH (90d merge)"
        else:
            # incremental from (existing max + 1) to week_end
            from_date = existing_max + timedelta(days=1)
            to_date = week_end_date
            mode = "INCREMENTAL"

        if from_date > to_date:
            print(f"✅ Candles already up-to-date: existing_max={existing_max} >= week_end={week_end_date}")
            print(f"Store: {STORE_PATH}")
            return

    from_ts = int(datetime.combine(from_date, datetime.min.time()).timestamp())
    to_ts = int(datetime.combine(to_date, datetime.max.time()).timestamp())

    print(f"\n📊 Candle ingest mode: {mode}")
    print(f"📊 Fetching candles for {len(symbols)} symbols from {from_date} to {to_date}")
    print("   Rate limit: ~1 call/second (Finnhub free tier)\n")

    all_candles = []

    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {symbol}...", end=" ")
        try:
            df = fetch_candles(symbol, from_ts, to_ts, api_key)
            if not df.empty:
                all_candles.append(df)
                print(f"✓ {len(df)} bars")
            else:
                print("(no_data)")

            time.sleep(1.1)  # avoid Finnhub bursts
        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    new_rows = pd.concat(all_candles, ignore_index=True) if all_candles else pd.DataFrame()

    if existing is None or existing.empty:
        combined = new_rows
    else:
        if replace_store:
            combined = new_rows
        else:
            combined = pd.concat([existing, new_rows], ignore_index=True)
    if combined.empty:
        raise RuntimeError("No candle data available after ingest")

    combined["symbol"] = combined["symbol"].astype(str).str.upper().str.strip()
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["symbol", "date"]).drop_duplicates(subset=["symbol", "date"], keep="last").reset_index(drop=True)

    # Guardrail: never shrink history unless explicitly replacing store
    if existing is not None and not existing.empty and not replace_store:
        old_min = pd.to_datetime(existing["date"]).min().date()
        old_max = pd.to_datetime(existing["date"]).max().date()
        new_min = pd.to_datetime(combined["date"]).min().date()
        new_max = pd.to_datetime(combined["date"]).max().date()
        if new_min > old_min:
            raise RuntimeError(
                f"❌ Refusing to shrink candles store: old_min={old_min} new_min={new_min}. Use --replace_store to override."
            )
        if new_max < old_max:
            raise RuntimeError(
                f"❌ Refusing to shrink candles store: old_max={old_max} new_max={new_max}. Use --replace_store to override."
            )

    expected_min = from_date if (existing is None or existing.empty or replace_store) else min(pd.to_datetime(existing["date"]).min().date(), from_date)
    expected_max = to_date
    _verify_candles(combined, expected_min=expected_min, expected_max=expected_max)

    write_parquet_atomic(combined, STORE_PATH)

    print(f"\n✓ Saved {len(combined):,} candle records to {STORE_PATH}")
    print(f"   Symbols: {combined['symbol'].nunique()}")
    print(f"   Date range: {combined['date'].min()} to {combined['date'].max()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Ingest market candles from Finnhub (incremental)")
    ap.add_argument("--universe", required=True, help="Path to universe CSV with 'symbol' column")
    ap.add_argument("--week_end", required=True, help="Week ending date YYYY-MM-DD")
    ap.add_argument("--force", action="store_true", help="Refetch last 90d ending at week_end and MERGE into store (safe)")
    ap.add_argument("--replace_store", action="store_true", help="DESTRUCTIVE: replace store with 90d history ending at week_end")
    args = ap.parse_args()

    main(args.universe, args.week_end, force=args.force, replace_store=args.replace_store)
