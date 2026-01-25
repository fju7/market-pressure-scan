# src/market_candles_store.py
"""
Incremental candle storage with atomic writes and deduplication.
Maintains single canonical store, appends only missing data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import pandas as pd

@dataclass(frozen=True)
class CandleStoreConfig:
    path: Path  # e.g. data/derived/market_candles/candles.parquet
    atomic_tmp_suffix: str = ".tmp"

REQUIRED_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize candle schema."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing candle columns: {missing}")
    
    # Normalize types
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"]).dt.date  # store as date
    
    # Rename columns if needed (o->open, h->high, etc.)
    rename_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    return df[REQUIRED_COLUMNS]

def load_existing(cfg: CandleStoreConfig) -> pd.DataFrame:
    """Load existing candles or return empty DataFrame."""
    if not cfg.path.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    df = pd.read_parquet(cfg.path)
    return _ensure_schema(df)

def compute_missing_keys(
    existing: pd.DataFrame,
    required_keys: pd.DataFrame,  # columns: symbol, date
) -> pd.DataFrame:
    """Return (symbol, date) pairs not in existing store."""
    if existing.empty:
        return required_keys.copy()

    ex = existing[["symbol", "date"]].drop_duplicates()
    req = required_keys.drop_duplicates()

    merged = req.merge(ex, on=["symbol", "date"], how="left", indicator=True)
    return merged.loc[merged["_merge"] == "left_only", ["symbol", "date"]]

def atomic_write_parquet(cfg: CandleStoreConfig, df: pd.DataFrame) -> None:
    """Write parquet atomically using temp file + replace."""
    tmp = cfg.path.with_suffix(cfg.path.suffix + cfg.atomic_tmp_suffix)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp, index=False)
    tmp.replace(cfg.path)  # atomic on same filesystem

def upsert_candles(
    cfg: CandleStoreConfig,
    new_rows: pd.DataFrame,
    *,
    full_refresh: bool = False,
) -> Tuple[int, int, int]:
    """
    Upsert candles into store with deduplication.
    
    Args:
        cfg: Store configuration
        new_rows: New candles to add
        full_refresh: If True, replace entire store (ignore existing)
        
    Returns:
        (n_existing, n_added, n_final) counts
    """
    if new_rows.empty:
        existing = load_existing(cfg)
        return (len(existing), 0, len(existing))
    
    new_rows = _ensure_schema(new_rows)

    if full_refresh:
        final = new_rows.drop_duplicates(subset=["symbol", "date"], keep="last")
        final = final.sort_values(["symbol", "date"]).reset_index(drop=True)
        atomic_write_parquet(cfg, final)
        return (0, len(new_rows), len(final))

    existing = load_existing(cfg)
    n_existing = len(existing)

    combined = pd.concat([existing, new_rows], ignore_index=True)
    # "last wins" lets you overwrite bad prior rows if needed
    final = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
    final = final.sort_values(["symbol", "date"]).reset_index(drop=True)

    atomic_write_parquet(cfg, final)
    return (n_existing, len(new_rows), len(final))

def audit_candles(df: pd.DataFrame, expected_date_range: Tuple[str, str] = None) -> dict:
    """
    Audit candle data quality.
    
    Returns:
        dict with audit results and any issues found
    """
    issues = []
    
    # Check for duplicates
    dupes = df.duplicated(subset=["symbol", "date"], keep=False)
    if dupes.any():
        issues.append(f"Found {dupes.sum()} duplicate (symbol, date) keys")
    
    # Check for nulls in OHLCV
    for col in ["open", "high", "low", "close", "volume"]:
        nulls = df[col].isna().sum()
        if nulls > 0:
            issues.append(f"Found {nulls} null values in {col}")
    
    # Check date range
    if expected_date_range:
        start, end = expected_date_range
        start_date = pd.to_datetime(start).date()
        end_date = pd.to_datetime(end).date()
        
        df_dates = pd.to_datetime(df["date"]).dt.date
        out_of_range = ((df_dates < start_date) | (df_dates > end_date)).sum()
        if out_of_range > 0:
            issues.append(f"Found {out_of_range} rows outside expected range [{start}, {end}]")
    
    return {
        "total_rows": len(df),
        "unique_symbols": df["symbol"].nunique(),
        "date_range": [df["date"].min(), df["date"].max()],
        "has_duplicates": dupes.any(),
        "issues": issues,
        "ok": len(issues) == 0
    }
