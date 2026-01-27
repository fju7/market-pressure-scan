from __future__ import annotations

import argparse
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests

from src.io_atomic import write_parquet_atomic
from src.reuse import should_skip


FINNHUB_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"


def _parse_week_end(cli_week_end: str | None) -> date:
    """
    Canonical week_end resolution order:
      1) env WEEK_END (CI)
      2) CLI --week_end
    """
    env_we = (os.environ.get("WEEK_END") or "").strip()
    we = env_we or (cli_week_end or "").strip()
    if not we:
        raise ValueError("Missing week_end. Set WEEK_END env var or pass --week_end YYYY-MM-DD.")
    return date.fromisoformat(we)


def _load_universe(universe_path: str) -> List[str]:
    df = pd.read_csv(universe_path)
    if "symbol" not in df.columns:
        raise ValueError(f"Universe file must have 'symbol' column: {universe_path}")
    syms = (
        df["symbol"]
        .astype(str)
        .str.upper()
        .str.strip()
        .dropna()
        .unique()
        .tolist()
    )
    # Finnhub company-news expects equities; keep as-is but remove empties
    syms = [s for s in syms if s]
    return syms


def fetch_company_news(
    symbol: str,
    from_date: date,
    to_date: date,
    api_key: str,
    max_retries: int = 3,
    timeout_s: int = 60,
) -> pd.DataFrame:
    """
    Fetch company news from Finnhub for a symbol.
    Returns a DataFrame with columns:
      symbol, published_utc, headline, summary, source, url, related
    """
    headers = {"X-Finnhub-Token": api_key}
    params = {"symbol": symbol, "from": from_date.isoformat(), "to": to_date.isoformat()}

    for attempt in range(max_retries):
        try:
            r = requests.get(FINNHUB_COMPANY_NEWS_URL, headers=headers, params=params, timeout=timeout_s)

            if r.status_code == 429:
                wait_time = 60 * (2 ** attempt)
                print(f"⚠ Rate limited (429) for {symbol}, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            r.raise_for_status()
            data = r.json()

            if not data:
                return pd.DataFrame()

            records: List[Dict[str, Any]] = []
            for item in data:
                # Finnhub returns epoch seconds in item["datetime"]
                dt_utc = None
                try:
                    dt_utc = datetime.fromtimestamp(int(item.get("datetime", 0)), tz=timezone.utc)
                except Exception:
                    dt_utc = None

                records.append(
                    {
                        "symbol": symbol,
                        "published_utc": dt_utc.isoformat() if dt_utc else None,
                        "headline": (item.get("headline") or "").strip(),
                        "summary": (item.get("summary") or "").strip(),
                        "source": (item.get("source") or "").strip(),
                        "url": (item.get("url") or "").strip(),
                        "related": (item.get("related") or "").strip(),
                    }
                )

            df = pd.DataFrame.from_records(records)
            return df

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"✗ Failed {symbol} after {max_retries} attempts: {e}")
                raise
            backoff = 5 * (attempt + 1)
            print(f"⚠ Request error for {symbol} (attempt {attempt+1}/{max_retries}): {e} — sleeping {backoff}s")
            time.sleep(backoff)

    return pd.DataFrame()


def main(
    week_end: str | None,
    universe_path: str,
    lookback_days: int = 90,
    force: bool = False,
) -> Path:
    api_key = (os.environ.get("FINNHUB_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("FINNHUB_API_KEY environment variable not set")

    week_end_date = _parse_week_end(week_end)
    from_date = week_end_date - timedelta(days=lookback_days)
    to_date = week_end_date

    output_dir = Path("data/derived/company_news") / f"week_ending={week_end_date.isoformat()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "company_news.parquet"

    # --- SKIP GUARD: must be before any fetching ---
    if should_skip(output_path, force):
        print(f"SKIP: {output_path} exists and --force not set.")
        return output_path

    symbols = _load_universe(universe_path)

    print(f"📰 Fetching company news for {len(symbols)} symbols from {from_date} to {to_date}")
    print("   Rate limit: ~1 call/second (Finnhub free tier)\n")

    all_rows: List[pd.DataFrame] = []

    for i, sym in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {sym}...", end=" ")
        df = fetch_company_news(sym, from_date, to_date, api_key)
        if df.empty:
            print("(skipped)")
        else:
            all_rows.append(df)
            print(f"✓ {len(df)} articles")

        time.sleep(1.1)

    if not all_rows:
        # still write an empty parquet for downstream stability
        empty = pd.DataFrame(columns=["symbol","published_utc","headline","summary","source","url","related"])
        write_parquet_atomic(empty, output_path)
        print(f"\n✓ Saved 0 news articles to {output_path}")
        return output_path

    combined = pd.concat(all_rows, ignore_index=True)

    # Normalize + types
    combined["symbol"] = combined["symbol"].astype(str).str.upper().str.strip()
    combined["published_utc"] = pd.to_datetime(combined["published_utc"], utc=True, errors="coerce")
    combined["headline"] = combined["headline"].astype(str).fillna("").str.strip()
    combined["summary"] = combined["summary"].astype(str).fillna("").str.strip()
    combined["url"] = combined["url"].astype(str).fillna("").str.strip()

    # Dedupe: (symbol, url, headline)
    combined = combined.drop_duplicates(subset=["symbol","url","headline"], keep="last")
    combined = combined.sort_values(["symbol","published_utc"], ascending=[True, False]).reset_index(drop=True)

    write_parquet_atomic(combined, output_path)

    print(f"\n✓ Saved {len(combined):,} news articles to {output_path}")
    print(f"   Symbols: {combined['symbol'].nunique()}")
    pu = combined["published_utc"]
    print(f"   Published range: {pu.min()} to {pu.max()}")
    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Ingest company news from Finnhub")
    ap.add_argument("--week_end", required=False, default=None, help="Week ending date YYYY-MM-DD (optional if WEEK_END set)")
    ap.add_argument("--universe", default="sp500_universe.csv", help="Universe CSV with 'symbol' column")
    ap.add_argument("--lookback_days", type=int, default=90)
    ap.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = ap.parse_args()

    main(args.week_end, args.universe, lookback_days=args.lookback_days, force=args.force)
