import pandas as pd
from pathlib import Path
import argparse

REQ_CLUSTER_COLS = {"symbol","cluster_id","rep_published_utc","rep_headline","rep_summary","cluster_size","unique_sources"}
REQ_ENRICH_COLS  = {"symbol","cluster_id","embedding","sentiment_json","event_json"}
REQ_CANDLE_COLS  = {"symbol","date","o","h","l","c","v"}

def main(universe, week_end):
    clusters = Path("data/derived/news_clusters") / f"week_ending={week_end}" / "clusters.parquet"
    enrich   = Path("data/derived/rep_enriched") / f"week_ending={week_end}" / "rep_enriched.parquet"
    candles  = Path("data/derived/market_daily") / "candles_daily.parquet"

    for p in [clusters, enrich, candles]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    cdf = pd.read_parquet(clusters)
    edf = pd.read_parquet(enrich)
    mdf = pd.read_parquet(candles)

    missing = REQ_CLUSTER_COLS - set(cdf.columns)
    if missing:
        raise RuntimeError(f"clusters.parquet missing columns: {sorted(missing)}")

    missing = REQ_ENRICH_COLS - set(edf.columns)
    if missing:
        raise RuntimeError(f"rep_enriched.parquet missing columns: {sorted(missing)}")

    missing = REQ_CANDLE_COLS - set(mdf.columns)
    if missing:
        raise RuntimeError(f"candles_daily.parquet missing columns: {sorted(missing)}")

    if "SPY" not in set(mdf["symbol"].unique()):
        raise RuntimeError("candles_daily.parquet must include SPY for AR5 baseline")

    print("✅ Step 1 preflight passed.")
    print(f"clusters rows: {len(cdf):,} | enriched rows: {len(edf):,} | candles rows: {len(mdf):,}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True)
    ap.add_argument("--week_end", required=True)
    args = ap.parse_args()
    main(args.universe, args.week_end)
