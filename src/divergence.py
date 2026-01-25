# src/divergence.py
"""Divergence feature computation for regime v1b"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_5d_returns(candles: pd.DataFrame, week_end: str) -> pd.DataFrame:
    """
    Compute 5-trading-day returns ending at week_end.
    
    Args:
        candles: DataFrame with columns: symbol, date, c (close price)
        week_end: YYYY-MM-DD string
        
    Returns:
        DataFrame with columns: symbol, ret_5d
    """
    c = candles.copy()
    c["date"] = pd.to_datetime(c["date"])
    c = c.sort_values(["symbol", "date"])

    # 5 trading days ending at week_end
    we = pd.to_datetime(week_end)
    window = c[c["date"] <= we].groupby("symbol").tail(6)  # 6 points -> 5d return
    window = window.groupby("symbol").agg(
        start_close=("c", "first"),
        end_close=("c", "last"),
    ).reset_index()

    window["ret_5d"] = (window["end_close"] / window["start_close"]) - 1.0
    return window[["symbol", "ret_5d"]]


def add_divergence_feature(
    features: pd.DataFrame, 
    candles: pd.DataFrame, 
    week_end: str,
    novelty_col: str = "NV_raw"
) -> pd.DataFrame:
    """
    Add divergence feature: excess return magnitude when novelty is low.
    
    Divergence = |excess_ret_5d| * (1 - normalized_novelty)
    
    This spikes when:
    - Stock moves significantly vs SPY
    - News novelty is low (market dislocation, not news-driven)
    
    Args:
        features: DataFrame with symbol and novelty_col
        candles: Market candles DataFrame
        week_end: Week ending date
        novelty_col: Column name for novelty score (default: NV_raw)
        
    Returns:
        features DataFrame with added columns: ret_5d, spy_ret_5d, excess_ret_5d, divergence
    """
    rets = compute_5d_returns(candles, week_end)

    # Compute SPY return and merge
    spy = rets[rets["symbol"] == "SPY"][["ret_5d"]].rename(columns={"ret_5d": "spy_ret_5d"})
    spy_ret = float(spy["spy_ret_5d"].iloc[0]) if len(spy) else 0.0

    out = features.merge(rets, on="symbol", how="left")
    out["spy_ret_5d"] = spy_ret
    out["excess_ret_5d"] = out["ret_5d"] - out["spy_ret_5d"]

    # Normalized novelty expected in [0,1]; if not, we'll normalize it
    if novelty_col in out.columns:
        nov = out[novelty_col].fillna(0.0)
        # Normalize to [0,1] if needed
        nov_min, nov_max = nov.min(), nov.max()
        if nov_max > nov_min:
            nov_norm = (nov - nov_min) / (nov_max - nov_min)
        else:
            nov_norm = pd.Series(0.0, index=nov.index)
        nov_norm = nov_norm.clip(0.0, 1.0)
    else:
        nov_norm = pd.Series(0.0, index=out.index)

    # Divergence emphasizes excess moves when novelty is low
    out["divergence"] = out["excess_ret_5d"].abs() * (1.0 - nov_norm)
    
    return out
