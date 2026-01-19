"""
Ingest company news for universe using Finnhub API.
Stores results in data/derived/news_raw/week_ending=YYYY-MM-DD/company_news.parquet
"""

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


def fetch_company_news(symbol: str, from_date: str, to_date: str, api_key: str) -> pd.DataFrame:
    """
    Fetch company news from Finnhub API.
    
    Args:
        symbol: Stock ticker
        from_date: Start date in YYYY-MM-DD format
        to_date: End date in YYYY-MM-DD format
        api_key: Finnhub API key
    
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
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    
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
            "url": item.get("url", "")
        })
    
    return pd.DataFrame(records)


def main(universe_path: str, week_end: str):
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
    
    print(f"📰 Fetching news for {len(symbols)} symbols from {from_date} to {to_date}")
    
    all_news = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {symbol}...", end=" ")
        
        try:
            df = fetch_company_news(symbol, from_date, to_date, api_key)
            if not df.empty:
                all_news.append(df)
                print(f"✓ {len(df)} articles")
            else:
                print("(no news)")
            
            # Rate limit: Finnhub free tier is 60 calls/min
            if i % 50 == 0:
                print(f"  💤 Rate limit pause...")
                time.sleep(1)
            else:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    if not all_news:
        print("⚠️  No news data fetched - creating empty placeholder")
        # Create empty DataFrame with correct schema
        combined = pd.DataFrame(columns=["symbol", "published_utc", "headline", "summary", "source", "url"])
    else:
        # Combine all data
        combined = pd.concat(all_news, ignore_index=True)
        
        # Sort by symbol and published date
        combined = combined.sort_values(["symbol", "published_utc"]).reset_index(drop=True)
    
    # Save to parquet
    output_dir = Path("data/derived/news_raw") / f"week_ending={week_end}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "company_news.parquet"
    
    combined.to_parquet(output_path, index=False)
    
    print(f"\n✅ Saved {len(combined):,} news articles to {output_path}")
    if not combined.empty:
        print(f"   Symbols: {combined['symbol'].nunique()}")
        print(f"   Sources: {combined['source'].nunique()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Ingest company news from Finnhub")
    ap.add_argument("--universe", required=True, help="Path to universe CSV with 'symbol' column")
    ap.add_argument("--week_end", required=True, help="Week ending date YYYY-MM-DD")
    args = ap.parse_args()
    
    main(args.universe, args.week_end)
