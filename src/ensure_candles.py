# src/ensure_candles.py
"""
Ensure required candles are present in store, fetching only missing data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import pandas as pd

from src.market_candles_store import (
    CandleStoreConfig,
    load_existing,
    compute_missing_keys,
    upsert_candles,
    audit_candles,
)

@dataclass(frozen=True)
class CandleRequest:
    symbols: list[str]
    date_min: str  # "YYYY-MM-DD"
    date_max: str  # "YYYY-MM-DD" inclusive

def build_required_keys(req: CandleRequest) -> pd.DataFrame:
    """
    Build all (symbol, date) combinations needed.
    Note: Includes weekends; API will skip non-trading days naturally.
    """
    start = pd.to_datetime(req.date_min).date()
    end = pd.to_datetime(req.date_max).date()
    dates = pd.date_range(start=start, end=end, freq="D")
    dates = [d.date() for d in dates]
    
    # Cartesian product
    keys = [(s, d) for s in req.symbols for d in dates]
    return pd.DataFrame(keys, columns=["symbol", "date"])

def ensure_candles(
    store_path: Path,
    req: CandleRequest,
    fetch_fn: Callable[[list[str], str, str], pd.DataFrame],
    *,
    full_refresh: bool = False,
) -> dict:
    """
    Ensure candles are present in store, fetching only missing data.
    
    Args:
        store_path: Path to candles.parquet
        req: CandleRequest specifying symbols and date range
        fetch_fn: Function (symbols, date_min, date_max) -> DataFrame
        full_refresh: If True, rebuild entire store from scratch
        
    Returns:
        dict with ingestion stats for report_meta
    """
    cfg = CandleStoreConfig(path=store_path)

    required_keys = build_required_keys(req)
    
    if full_refresh:
        print(f"🔄 Full refresh requested - fetching all {len(required_keys)} (symbol, date) pairs")
        missing = required_keys
        existing = pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    else:
        existing = load_existing(cfg)
        missing = compute_missing_keys(existing, required_keys)
        print(f"📊 Store has {len(existing)} rows, need {len(missing)} more")

    fetched = pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    
    if not missing.empty:
        print(f"📥 Fetching {len(missing)} missing candles...")
        # Fetch by unique symbols (API returns all dates in range)
        symbols = sorted(missing["symbol"].unique().tolist())
        fetched = fetch_fn(symbols, req.date_min, req.date_max)
        print(f"✅ Fetched {len(fetched)} candle records")

    n_existing, n_added, n_final = upsert_candles(cfg, fetched, full_refresh=full_refresh)
    
    # Audit final store
    final_df = load_existing(cfg)
    audit = audit_candles(final_df, expected_date_range=(req.date_min, req.date_max))
    
    if not audit["ok"]:
        print(f"⚠️ Candle audit found issues: {audit['issues']}")
    
    return {
        "candles_store_path": str(store_path),
        "n_existing": n_existing,
        "n_added": n_added,
        "n_final": n_final,
        "n_missing_requested": int(len(missing)),
        "full_refresh": full_refresh,
        "audit": audit,
    }
