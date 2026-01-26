"""
Ingest company news for a universe using the Finnhub API.
Stores results in data/derived/company_news/week_ending=YYYY-MM-DD/company_news.parquet (canonical artifact).
"""

import argparse
from .reuse import should_skip
import os
import random
import sys
from src.io_atomic import write_parquet_atomic
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


def fetch_company_news(symbol: str, from_date: str, to_date: str, api_key: str, max_retries: int = 4) -> pd.DataFrame:
    """
    Fetch company news from Finnhub API with retry and exponential backoff.
    
    Args:
        symbol: Stock ticker
        from_date: Start date in YYYY-MM-DD format
        to_date: End date in YYYY-MM-DD format
        api_key: Finnhub API key
        max_retries: Maximum number of retry attempts on 429 errors (default: 4)
    
    Returns:
        DataFrame with columns: symbol, published_utc, headline, summary, source, url
    """
    url = "https://finnhub.io/api/v1/company-news"
    headers = {"X-Finnhub-Token": api_key}
    params = {
        "symbol": symbol,
        "from": from_date,
        "to": to_date
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            # Handle 429 rate limit with exponential backoff + jitter
            if response.status_code == 429:
                base_wait = 30 * (2 ** attempt)  # 30s, 60s, 120s, 240s
                jitter = random.uniform(0, 5)  # Add 0-5s jitter to avoid thundering herd
                wait_time = base_wait + jitter
                print(f"⚠️  Rate limited (429), retry {attempt+1}/{max_retries}, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            write_parquet_atomic(combined, output_path)

            print(f"\nSaved {len(combined):,} news articles to {output_path}")
            
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # Transform to our schema
            records = []
            for item in data:
                records.append({
                    "symbol": symbol,
                    "published_utc": datetime.fromtimestamp(item["datetime"]).isoformat(),
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "related": item.get("related", "")
                })
            
            return pd.DataFrame(records)
                    print(f"\n✓ Saved {len(combined):,} news articles to {output_path}")
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                # Last attempt failed
                print(f"✗ Failed after {max_retries} attempts: {e}")
                raise
            print(f"⚠️  Request error (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(5 * (attempt + 1))
    
    return pd.DataFrame()


def filter_symbols_by_movement(symbols: list, week_end_date, api_key: str, 
                                vol_threshold: float = 1.5, price_threshold: float = 0.05) -> list:
    """
    Filter symbols to only those with significant price/volume movement.
    Reduces API load by focusing on interesting symbols.
    
    Args:
        symbols: List of symbols to filter
        week_end_date: Week ending date
        api_key: Finnhub API key
        vol_threshold: Minimum volume ratio vs 20-day average (default: 1.5x)
        price_threshold: Minimum absolute price change (default: 5%)
    
    Returns:
        Filtered list of symbols
    """
    # This is a placeholder implementation
    # In production, you'd fetch recent candles and filter by:
    # - Price change > price_threshold (e.g., 5%)
    # - Volume > vol_threshold * average volume
    # For now, return all symbols (implement as needed)
    print(f"⚠️  Movement filtering not yet implemented, using all symbols")
    return symbols


def main(universe_path: str, week_end: str, coverage_threshold: float = 0.75, 
         filter_by_movement: bool = False, qps_limit: float = 0.5, fast_fail_threshold: float = 0.3,
         fast_fail_check_interval: int = 100, force: bool = False):
    """
    Ingest company news with retry, backoff, and coverage guardrails.
    
    Args:
        universe_path: Path to universe CSV file
        week_end: Week ending date (YYYY-MM-DD)
        coverage_threshold: Minimum fraction of symbols that must have news (default: 0.75 = 75%)
        filter_by_movement: If True, only fetch news for symbols with significant price/volume movement (default: False)
        qps_limit: Query-per-second rate limit (default: 0.5 = 30 calls/min)
        fast_fail_threshold: If coverage drops below this after fast_fail_check_interval symbols, abort (default: 0.3)
        fast_fail_check_interval: Check coverage after this many symbols (default: 100)
    """
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("FINNHUB_API_KEY environment variable not set")
    
    # Parse week_end and determine the week
    week_end_date = datetime.strptime(week_end, "%Y-%m-%d").date()
    week_start_date = week_end_date - timedelta(days=6)  # 7-day window
    
    from_date = week_start_date.strftime("%Y-%m-%d")
    to_date = week_end_date.strftime("%Y-%m-%d")
    
    # Load universe
    universe_df = pd.read_csv(universe_path)
    
    if "symbol" not in universe_df.columns:
        raise ValueError(f"Universe file must have 'symbol' column: {universe_path}")
    
    symbols = list(universe_df["symbol"].unique())
    
    # Optional: Filter to symbols with movement (reduces API load)
    if filter_by_movement:
        symbols = filter_symbols_by_movement(symbols, week_end_date, api_key)
        print(f"🎯 Filtered to {len(symbols)} symbols with significant movement")
    
    print(f"📰 Fetching news for {len(symbols)} symbols from {from_date} to {to_date}")
    print(f"   Rate limit: {qps_limit} calls/sec | Coverage threshold: {coverage_threshold*100:.0f}%")
    
    all_news = []
    symbols_with_news = set()
    failed_symbols = []
    
    sleep_time = 1.0 / qps_limit  # Convert QPS to sleep time
    
    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {symbol}...", end=" ", flush=True)
        
        try:
            df = fetch_company_news(symbol, from_date, to_date, api_key)
            if not df.empty:
                all_news.append(df)
                symbols_with_news.add(symbol)
                print(f"✓ {len(df)} articles")
            else:
                print("(no news)")
            
            # Adaptive rate limiting
            time.sleep(sleep_time)
                
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed_symbols.append(symbol)
            continue
        
        # Fast-fail check: abort early if coverage is dismal
        if i == fast_fail_check_interval:
            early_coverage = len(symbols_with_news) / i if i > 0 else 0.0
            if early_coverage < fast_fail_threshold:
                error_msg = (
                    f"\n❌ FAST-FAIL: Coverage too low after {i} symbols\n"
                    f"   Coverage: {early_coverage*100:.1f}% < fast-fail threshold {fast_fail_threshold*100:.0f}%\n"
                    f"   Only {len(symbols_with_news)}/{i} symbols have news\n"
                    f"   Aborting to prevent death-march (remaining: {len(symbols)-i} symbols)"
                )
                print(error_msg)
                print("::error::FAST-FAIL - Coverage too low, aborting news ingestion")
                sys.exit(1)
    
    # Coverage guardrail: Check if we have sufficient data
    coverage = len(symbols_with_news) / len(symbols) if symbols else 0.0
    
    print(f"\n📊 Coverage Report:")
    print(f"   Total symbols: {len(symbols)}")
    print(f"   Symbols with news: {len(symbols_with_news)}")
    print(f"   Failed symbols: {len(failed_symbols)}")
    print(f"   Coverage: {coverage*100:.1f}%")
    
    if coverage < coverage_threshold:
        error_msg = (
            f"❌ DATA INCOMPLETE — RATE LIMITED\n"
            f"   Coverage {coverage*100:.1f}% < threshold {coverage_threshold*100:.0f}%\n"
            f"   Only {len(symbols_with_news)}/{len(symbols)} symbols have news data\n"
            f"   Failed symbols: {', '.join(failed_symbols[:10])}"
        )
        if len(failed_symbols) > 10:
            error_msg += f" ... and {len(failed_symbols)-10} more"
        
        print(error_msg)
        print(f"::error::DATA INCOMPLETE - Coverage {coverage*100:.1f}% below threshold {coverage_threshold*100:.0f}%")
        sys.exit(1)
    
    if not all_news:
        print("⚠️  No news data fetched - creating empty placeholder")
        # Create empty DataFrame with correct schema
        combined = pd.DataFrame(columns=["symbol", "published_utc", "headline", "summary", "source", "url", "related"])
    else:
        # Combine all data
        combined = pd.concat(all_news, ignore_index=True)
        
        # Sort by symbol and published date
        combined = combined.sort_values(["symbol", "published_utc"]).reset_index(drop=True)
        
        # --- Relevance filter (v1) - DISABLED for now ---
        # Many feeds include market wrap content; keep items likely related to the symbol.
        # NOTE: Finnhub company-news endpoint already filters by symbol, so additional filtering
        # is too aggressive and removes valid articles. Consider re-enabling with company name matching.
        # text = (combined["headline"].fillna("") + " " + combined["summary"].fillna("")).str.upper()
        # sym = combined["symbol"].astype(str).str.upper().str.strip()
        # rel = combined["related"].fillna("").astype(str).str.upper()
        # mask_ticker_in_text = text.str.contains(sym, regex=False)
        # mask_ticker_in_related = rel.str.contains(sym, regex=False)
        # before_filter = len(combined)
        # combined = combined[mask_ticker_in_text | mask_ticker_in_related].copy()
        # print(f"   Relevance filter: {before_filter:,} → {len(combined):,} articles ({100*len(combined)/before_filter:.1f}% retained)")
        # print(f"[INFO] relevance filter kept {len(combined):,} rows")
    
    # Save to parquet
    output_dir = Path("data/derived/company_news") / f"week_ending={week_end}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "company_news.parquet"

    if should_skip(output_path, force):
        print(f"SKIP: {output_path} exists and --force not set.")
        return

    combined.to_parquet(output_path, index=False)

    print(f"\n 5 Saved {len(combined):,} news articles to {output_path}")
    if not combined.empty:
        print(f"   Symbols: {combined['symbol'].nunique()}")
        print(f"   Sources: {combined['source'].nunique()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        "Ingest company news from Finnhub",
        description="Fetch company news with retry, backoff, and coverage guardrails"
    )
    ap.add_argument("--universe", required=True, help="Path to universe CSV with 'symbol' column")
    ap.add_argument("--week_end", required=True, help="Week ending date YYYY-MM-DD")
    ap.add_argument(
        "--coverage_threshold",
        type=float,
        default=0.75,
        help="Minimum fraction of symbols that must have news (default: 0.75 = 75%%; start at 0.6 for initial weeks)"
    )
    ap.add_argument(
        "--filter_by_movement",
        action="store_true",
        help="Only fetch news for symbols with significant price/volume movement (reduces API load)"
    )
    ap.add_argument(
        "--qps_limit",
        type=float,
        default=0.5,
        help="Query-per-second rate limit (default: 0.5 = 30 calls/min, conservative)"
    )
    ap.add_argument(
        "--fast_fail_threshold",
        type=float,
        default=0.3,
        help="Abort if coverage drops below this after fast_fail_check_interval symbols (default: 0.3 = 30%%)"
    )
    ap.add_argument(
        "--fast_fail_check_interval",
        type=int,
        default=100,
        help="Check coverage after this many symbols for fast-fail (default: 100)"
    )
    ap.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = ap.parse_args()

    main(args.universe, args.week_end, args.coverage_threshold, 
         args.filter_by_movement, args.qps_limit, args.fast_fail_threshold,
         args.fast_fail_check_interval, force=args.force)
