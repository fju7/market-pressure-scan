# src/features_scores.py
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil import tz

# Regime system imports
from src.scoring_schema import load_schema, write_schema_provenance
from src.io_atomic import write_parquet_atomic
# src/features_scores.py
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil import tz

from src.io_atomic import write_parquet_atomic
from src.reuse import should_skip
from src.scoring_schema import load_schema, write_schema_provenance

NY = tz.gettz("America/New_York")


# =============================================================================
# Debug helpers
# =============================================================================
# ----------------------------
# Debug helpers
# ----------------------------

def _debug_env_stamp():
    print("=== FEATURES_SCORES DEBUG STAMP ===")
    print("module_file:", __file__)
    print("python:", platform.python_version())
    print("pandas:", pd.__version__, "numpy:", np.__version__)
    try:
        src = Path(__file__).read_bytes()
        print("features_scores.py sha256:", hashlib.sha256(src).hexdigest()[:16])
    except Exception as e:
        print("sha256 read failed:", e)
    print("===================================")

def dump_df(df: pd.DataFrame, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.csv"
    df.to_csv(p, index=True)
    print(f"📦 dumped df -> {p} (shape={df.shape}, index_names={df.index.names})")

# ----------------------------
# Config / paths
# ----------------------------

NY = tz.gettz("America/New_York")

@dataclass(frozen=True)

class Paths:
    root: Path
    derived: Path
    # Canonical artifact locations (single source of truth)
    market_daily_path: Path
    company_news_dir: Path
    news_clusters_dir: Path
    rep_enriched_dir: Path
    out_features_dir: Path
    out_scores_dir: Path
    regime_id: str = "news-novelty-v1"  # For provenance tracking


def default_paths(regime_id: str = "news-novelty-v1", schema_id: str | None = None) -> Paths:
    root = Path(__file__).resolve().parents[1]
    derived = root / "data" / "derived"
    # Canonical artifact locations
    market_daily_path = derived / "market_daily" / "candles_daily.parquet"
    company_news_dir = derived / "company_news"
    news_clusters_dir = derived / "news_clusters"
    rep_enriched_dir = derived / "rep_enriched"
    features_base = derived / "features_weekly" / f"regime={regime_id}"
    scores_base = derived / "scores_weekly" / f"regime={regime_id}"
    if schema_id:
        scores_base = scores_base / f"schema={schema_id}"
    return Paths(
        root=root,
        derived=derived,
        market_daily_path=market_daily_path,
        company_news_dir=company_news_dir,
        news_clusters_dir=news_clusters_dir,
        rep_enriched_dir=rep_enriched_dir,
        out_features_dir=features_base,
        out_scores_dir=scores_base,
        regime_id=regime_id,
    )

# ----------------------------
# Utilities
# ----------------------------

def parse_week_end(s: str) -> datetime:
    # expects YYYY-MM-DD (ET date representing the Friday)
    dt = datetime.fromisoformat(s)
    return datetime(dt.year, dt.month, dt.day, 0, 0, 0)

def week_end_cutoff_utc(week_end_et: datetime) -> datetime:
    # Friday 4pm ET cutoff
    dt_et = datetime(week_end_et.year, week_end_et.month, week_end_et.day, 16, 0, 0, tzinfo=NY)
    return dt_et.astimezone(timezone.utc)

def winsorize(x: np.ndarray, zcap: float = 3.0) -> np.ndarray:
    return np.clip(x, -zcap, zcap)

def zscore_series(s: pd.Series, zcap: float = 3.0) -> pd.Series:
    vals = s.astype(float).to_numpy()
    mu = np.nanmean(vals)
    sd = np.nanstd(vals)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    z = (vals - mu) / sd
    z = winsorize(z, zcap=zcap)
    return pd.Series(z, index=s.index)

def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (m,d), b: (n,d)
    # returns (m,n)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T

def safe_json_load(x) -> dict:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return {}
        return json.loads(x)
    return {}

# ----------------------------
# Loaders
# ----------------------------

def load_universe(universe_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(universe_csv)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    return df[["symbol", "sector"]].drop_duplicates()

def load_week_parquet(base_dir: Path, week_end: str, filename: str) -> pd.DataFrame:
    p = base_dir / f"week_ending={week_end}" / filename
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")
    return pd.read_parquet(p)

def list_prior_weeks(week_end_et: datetime, n_weeks: int) -> List[str]:
    # includes current week_end as week 0
    out = []
    for k in range(n_weeks):
        d = week_end_et - timedelta(days=7 * k)
        out.append(d.date().isoformat())
    return out

# ----------------------------
# Market features (AR5, VS, VR)
# ----------------------------

def market_features_for_week(
    candles_daily: pd.DataFrame,
    universe_syms: List[str],
    week_end: str,
) -> pd.DataFrame:
    """
    candles_daily columns: symbol, date (YYYY-MM-DD), o, h, l, c, v
    Must include SPY.
    Computes as-of week_end (or nearest prior trading day):
      - AR5: 5-day close/close return relative to SPY
      - VS: log( avgVol_5 / avgVol_60 )
      - VR: 60d realized vol percentile across universe
    """
    cd = candles_daily.copy()
    cd["date"] = pd.to_datetime(cd["date"]).dt.date
    target = datetime.fromisoformat(week_end).date()

    # Restrict to universe + SPY
    syms = set(universe_syms) | {"SPY"}
    cd = cd[cd["symbol"].isin(syms)]

    # For each symbol, find last available date <= target
    # We'll compute returns ending at that date.
    last_dates_list = []
    for sym, g in cd.groupby("symbol"):
        d = g.loc[g["date"] <= target, "date"]
        if not d.empty:
            last_dates_list.append({"symbol": sym, "asof_date": d.max()})
    
    last_dates = pd.DataFrame(last_dates_list)
    if last_dates.empty:
        raise RuntimeError("No trading data found for universe symbols.")

    # Build a map for SPY asof_date (use SPY's last date)
    spy_asof = last_dates.loc[last_dates["symbol"] == "SPY", "asof_date"]
    if spy_asof.empty:
        raise RuntimeError("SPY not found in candles_daily.parquet (required for AR5 baseline).")
    spy_asof_date = spy_asof.iloc[0]

    # Use SPY as-of date as the portfolio as-of date; for other symbols we still use their last<=target,
    # but AR5 baseline will use SPY's 5d return ending spy_asof_date.
    # If a symbol's last date differs (rare in S&P), it will be slightly misaligned; acceptable v1.

    # Helper to compute 5d close-to-close return ending at asof_date
    def ret_close(g: pd.DataFrame, end_date: datetime.date, lookback: int) -> Optional[float]:
        gg = g[g["date"] <= end_date].sort_values("date")
        if len(gg) < lookback + 1:
            return None
        c_end = float(gg["close"].iloc[-1])
        c_start = float(gg["close"].iloc[-(lookback + 1)])
        if c_start <= 0:
            return None
        return c_end / c_start - 1.0

    # Helper avg volume
    def avg_vol(g: pd.DataFrame, end_date: datetime.date, lookback: int) -> Optional[float]:
        gg = g[g["date"] <= end_date].sort_values("date")
        if len(gg) < lookback:
            return None
        return float(np.nanmean(gg["volume"].iloc[-lookback:].astype(float).to_numpy()))

    # Helper vol (std of daily close returns) over lookback
    def realized_vol(g: pd.DataFrame, end_date: datetime.date, lookback: int) -> Optional[float]:
        gg = g[g["date"] <= end_date].sort_values("date")
        if len(gg) < lookback + 1:
            return None
        c = gg["close"].astype(float).to_numpy()
        rets = np.diff(c[-(lookback + 1):]) / (c[-(lookback + 1):-1] + 1e-12)
        if rets.size < 2:
            return None
        return float(np.nanstd(rets, ddof=1))

    # Precompute SPY AR5
    spy_g = cd[cd["symbol"] == "SPY"]
    spy_ret5 = ret_close(spy_g, spy_asof_date, lookback=5)
    if spy_ret5 is None:
        raise RuntimeError("Not enough SPY history to compute AR5.")

    rows = []
    for sym in universe_syms:
        g = cd[cd["symbol"] == sym]
        if g.empty:
            continue
        asof_date = last_dates.loc[last_dates["symbol"] == sym, "asof_date"]
        if asof_date.empty:
            continue
        asof_date = asof_date.iloc[0]

        r5 = ret_close(g, asof_date, lookback=5)
        v5 = avg_vol(g, asof_date, lookback=5)
        v60 = avg_vol(g, asof_date, lookback=60)
        rv60 = realized_vol(g, asof_date, lookback=60)

        if r5 is None or v5 is None or v60 is None or rv60 is None or v60 <= 0:
            continue

        ar5 = float(r5 - spy_ret5)
        vs = float(math.log(v5 / v60))
        rows.append((sym, str(asof_date), ar5, vs, rv60))

    mkt = pd.DataFrame(rows, columns=["symbol", "asof_date", "AR5", "VS_raw", "RV60"])
    if mkt.empty:
        raise RuntimeError("Market feature table is empty; check candles_daily data.")

    # Vol regime percentile across universe
    mkt["VR_pct"] = mkt["RV60"].rank(pct=True)

    # Z-scores used later
    mkt["z_AR5"] = zscore_series(mkt["AR5"])
    mkt["z_VS"] = zscore_series(mkt["VS_raw"])
    # We keep RV60 raw + VR_pct; VR penalty uses pct.

    return mkt

# ----------------------------
# News-derived features (NV, NA, NS, SS, EI)
# ----------------------------

def compute_rep_sent_score_final(sent: dict) -> Tuple[float, float, str]:
    """
    Applies v1 penalties + confidence shrinkage.
    Returns (sent_score_final, confidence_adj, driver)
    """
    score = float(sent.get("sent_score", 0.0) or 0.0)
    conf = float(sent.get("confidence", 0.0) or 0.0)
    driver = str(sent.get("sent_driver", ""))
    if driver == "Market/Price-action":
        score *= 0.35
        conf *= 0.70
    elif driver == "Speculation/Opinion":
        score *= 0.65
        conf *= 0.85
    # confidence shrinkage
    score_final = score * conf
    return float(score_final), float(conf), driver

def compute_event_sev_final(evt: dict, unique_sources: float = 1.0) -> float:
    sev = float(evt.get("event_severity", 0.0) or 0.0)
    conf = float(evt.get("event_confidence", 0.0) or 0.0)
    etype = str(evt.get("event_type_primary", ""))
    sev_adj = sev * conf

    if etype == "PRICE_ACTION_RECAP":
        sev_adj = min(sev_adj, 0.5)
    if etype == "MACRO_SECTOR":
        sev_adj *= 0.8

    # light echo weight, capped
    w_echo = 1.0 + 0.10 * math.log(1.0 + max(unique_sources, 0.0))
    w_echo = min(w_echo, 1.3)
    return float(sev_adj * w_echo)

def build_news_feature_panel(
    paths: Paths,
    universe: pd.DataFrame,
    week_end_et: datetime,
    week_end: str,
    lookback_weeks: int = 12,
    novelty_min_history: int = 15,
) -> pd.DataFrame:
    # ...existing code...
    # Echo features
    if "cluster_size" not in df.columns:
        df["cluster_size"] = 1.0
    df["EC_raw_cluster"] = df["cluster_size"].astype(float)

    # --- Counts for NV/NA (weekly proxy) ---
    # Use deduped story clusters per week as count_5d proxy.
    weekly_counts = (
        df.groupby(["week_ending_date", "symbol"], as_index=False)
          .agg(total_clusters=("cluster_id", "nunique"),
               total_raw_echo=("cluster_size", "sum"),
               EC_raw=("EC_raw_cluster", "mean"),
               unique_sources_mean=("unique_sources", "mean"))
    )

    # Build rolling sums over weeks for each symbol
    # Create week index ordering by date
    weekly_counts["week_dt"] = pd.to_datetime(weekly_counts["week_ending_date"])
    weekly_counts = weekly_counts.sort_values(["symbol", "week_dt"])

    # Rolling windows in weeks:
    # 5d ~ 1 week, 20d ~ 4 weeks, 60d ~ 12 weeks
    debug_dir = paths.derived / "debug_ci" / f"week_ending={week_end}"
    
    print("[DEBUG] pre add_roll cols:", list(weekly_counts.columns), "index.names:", weekly_counts.index.names)
    dump_df(weekly_counts.head(200), debug_dir, "weekly_counts_pre_add_roll_head")

    print("[DEBUG] post add_roll cols:", list(weekly_counts.columns), "index.names:", weekly_counts.index.names)
    if "symbol" in weekly_counts.columns:
        print("[DEBUG] head identifiers:", weekly_counts[["symbol"]].head())
    else:
        print("[DEBUG] head (no symbol column):", weekly_counts.head())
    
    # Sanity check: ensure no "index" column was accidentally created
    if "index" in weekly_counts.columns:
        raise ValueError(
            f"Unexpected 'index' column found after add_roll. This suggests unnamed index materialization. "
            f"Columns: {weekly_counts.columns.tolist()}"
        )
    
    # Now guarantee symbol is a column
    if "symbol" not in weekly_counts.columns:
        raise ValueError(
            f"'symbol' missing after add_roll. Columns={weekly_counts.columns.tolist()} "
            f"IndexNames={weekly_counts.index.names}"
        )
    
    # Extract current week rows with explicit datetime handling
    week_end_ts = pd.to_datetime(week_end)
    weekly_counts["week_ending_date"] = pd.to_datetime(weekly_counts["week_ending_date"])
    cur_counts = weekly_counts.loc[weekly_counts["week_ending_date"] == week_end_ts].copy()
    
    if cur_counts.empty:
        raise ValueError(
            f"cur_counts empty for week_end={week_end_ts.date()} after filtering. "
            f"Check week_ending_date types and coverage. Available weeks: "
            f"{sorted(weekly_counts['week_ending_date'].dt.date.unique().tolist())}"
        )
    
    # Verify per-symbol data
    if "symbol" not in cur_counts.columns:
        raise ValueError(f"'symbol' not found in cur_counts. Columns: {cur_counts.columns.tolist()}")
    
    symbol_count = cur_counts["symbol"].nunique()
    if symbol_count == 0:
        raise ValueError(f"cur_counts has no symbols for week_end={week_end_ts.date()}")

    # NV_raw and NA_raw (as previously specified with scaling)
    cur_counts["NV_raw"] = np.log1p(cur_counts["count_5d_dedup"]) - np.log1p(cur_counts["count_60d_dedup"] / 12.0)
    cur_counts["NA_raw"] = np.log1p(cur_counts["count_5d_dedup"]) - np.log1p(cur_counts["count_20d_dedup"] / 4.0)

    # --- Novelty (NS) ---
    # Current week embeddings per symbol, compare to historical embeddings prior weeks (exclude current week)
    # We compute NS_raw = 0.7*median(1 - topKmedian(sim)) + 0.3*p75(...)
    df["week_ending_date"] = pd.to_datetime(df["week_ending_date"])
    week_end_ts = pd.to_datetime(week_end)
    df["week_dt"] = df["week_ending_date"]
    cur_df = df.loc[df["week_ending_date"] == week_end_ts].copy()
    hist_df = df.loc[df["week_ending_date"] != week_end_ts].copy()

    # Prepare embedding arrays
    # embeddings stored as list[float] in parquet -> object; convert to np arrays
    def to_vec(x) -> Optional[np.ndarray]:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        if isinstance(x, (list, tuple, np.ndarray)):
            return np.array(x, dtype=np.float32)
        return None

    cur_df["emb_vec"] = cur_df["embedding"].apply(to_vec)
    hist_df["emb_vec"] = hist_df["embedding"].apply(to_vec)

    # Keep only 12-week lookback history (already loaded), and exclude missing vectors
    cur_df = cur_df[cur_df["emb_vec"].notna()].copy()
    hist_df = hist_df[hist_df["emb_vec"].notna()].copy()

    ns_rows = []
    # Use canonical symbol set (no duplicates, consistent case)
    symbols = pd.Series(universe["symbol"]).dropna().astype(str).unique().tolist()
    for sym in symbols:
        cur_sym = cur_df[cur_df["symbol"] == sym]
        if cur_sym.empty:
            ns_rows.append((sym, np.nan, 0, 0))
            continue
        hist_sym = hist_df[hist_df["symbol"] == sym]
        # history count
        hcount = len(hist_sym)
        if hcount == 0:
            ns_rows.append((sym, np.nan, 0, 0))
            continue

        cur_mat = np.vstack(cur_sym["emb_vec"].to_list())
        hist_mat = np.vstack(hist_sym["emb_vec"].to_list())

        sims = cosine_sim_matrix(cur_mat, hist_mat)  # (m,n)
        # novelty per current rep: 1 - median(topK similarities)
        novs = []
        topk = min(10, hist_mat.shape[0])
        for i in range(sims.shape[0]):
            row = sims[i]
            # top-k median
            idx = np.argpartition(-row, kth=topk - 1)[:topk]
            top_vals = row[idx]
            sim_topk_median = float(np.median(top_vals))
            nov = 1.0 - sim_topk_median
            novs.append(nov)

        novs = np.array(novs, dtype=float)
        ns_med = float(np.median(novs))
        ns_p75 = float(np.percentile(novs, 75))
        ns_raw = 0.7 * ns_med + 0.3 * ns_p75

        ns_rows.append((sym, ns_raw, hcount, len(cur_sym)))

    ns = pd.DataFrame(ns_rows, columns=["symbol", "NS_raw", "nov_hist_count", "nov_cur_reps"])

    # Cold start shrinkage
    # If history < novelty_min_history, blend toward cross-sectional median
    ns_array = ns["NS_raw"].to_numpy()
    total_symbols = len(ns)
    nan_count = np.sum(np.isnan(ns_array))
    valid_count = total_symbols - nan_count
    nan_pct = (nan_count / total_symbols * 100) if total_symbols > 0 else 0.0
    
    # Log novelty score coverage
    print(f"  [Novelty] Total symbols: {total_symbols} | Valid NS_raw: {valid_count} ({100-nan_pct:.1f}%) | NaN: {nan_count} ({nan_pct:.1f}%)")
    
    # Detect complete failure vs partial degradation
    if np.all(np.isnan(ns_array)):
        # All NaN - likely embeddings pipeline broke or insufficient history
        avg_history = ns["nov_hist_count"].mean() if len(ns) > 0 else 0
        
        error_msg = (
            f"⚠️  WARNING: All NS_raw values are NaN ({total_symbols} symbols)!\n"
            f"   Average history count: {avg_history:.1f}\n"
        )
        
        # Decide: hard fail if embeddings broke, soft degrade if insufficient history
        if avg_history < 1.0:
            # Insufficient historical data - this is expected in early weeks
            print(error_msg + "   Reason: Insufficient historical data (expected for early weeks)")
            print("   → Setting median_ns = 0.0 and continuing with degraded novelty scores")
            median_ns = 0.0
        else:
            # Have history but no embeddings - embeddings pipeline likely broke
            print(error_msg + "   Reason: Historical data exists but no embeddings found")
            print("   → This suggests the embeddings pipeline is broken!")
            raise RuntimeError(
                f"Embeddings pipeline failure: {total_symbols} symbols have history (avg={avg_history:.1f}) "
                f"but all NS_raw are NaN. Check embedding generation in enrichment step."
            )
    elif nan_pct > 50.0:
        # Partial degradation - warn but continue
        print(f"  ⚠️  WARNING: High NaN rate ({nan_pct:.1f}%) in NS_raw - novelty scores may be degraded")
        median_ns = float(np.nanmedian(ns_array))
    else:
        median_ns = float(np.nanmedian(ns_array))
    def shrink(row):
        val = row["NS_raw"]
        h = int(row["nov_hist_count"])
        if not np.isfinite(val):
            return median_ns
        if h >= novelty_min_history:
            return float(val)
        w = max(0.0, min(1.0, h / float(novelty_min_history)))
        return float(w * val + (1.0 - w) * median_ns)

    ns["NS_raw_shrunk"] = ns.apply(shrink, axis=1)

    # --- Sentiment shift (SS_raw) ---
    # Weekly: sent_5d = weighted mean current week; baseline sent_60d = weighted mean prior 12 weeks excluding current.
    # Weight: w = 1 + 0.15*log(1+echo_index) ; v1 proxy echo_index via cluster_size & unique_sources
    # We'll compute w using cluster_size and unique_sources.
    def rep_weight(r) -> float:
        cs = float(r.get("cluster_size", 1.0) or 1.0)
        us = float(r.get("unique_sources", 1.0) or 1.0)
        echo_index = math.log(1 + cs) * math.log(1 + us)
        return 1.0 + 0.15 * math.log(1.0 + echo_index)

    df["w_rep"] = df.apply(lambda r: rep_weight(r), axis=1)

    cur_sent = df[df["week_ending_date"] == week_end_ts].copy()
    hist_sent = df[df["week_ending_date"] != week_end_ts].copy()

    # Aggregate weighted sentiment
    def wmean(x: np.ndarray, w: np.ndarray) -> float:
        wsum = float(np.nansum(w))
        if wsum <= 0:
            return 0.0
        return float(np.nansum(x * w) / wsum)

    sent_rows = []
    # Use canonical symbol set (no duplicates, consistent case)
    symbols = pd.Series(universe["symbol"]).dropna().astype(str).unique().tolist()
    for sym in symbols:
        c = cur_sent[cur_sent["symbol"] == sym]
        h = hist_sent[hist_sent["symbol"] == sym]
        sent_5d = wmean(c["sent_score_final"].to_numpy(dtype=float), c["w_rep"].to_numpy(dtype=float)) if not c.empty else 0.0
        sent_60d = wmean(h["sent_score_final"].to_numpy(dtype=float), h["w_rep"].to_numpy(dtype=float)) if not h.empty else 0.0
        ss_raw = sent_5d - sent_60d

        price_action_rate = float(np.mean((c["sent_driver"] == "Market/Price-action").to_numpy())) if not c.empty else 0.0
        mixed_rate = float(np.mean((c["sentiment"].apply(lambda d: d.get("sent_label")) == "Mixed/Unclear").to_numpy())) if not c.empty else 0.0
        sent_intensity = float(wmean(np.abs(c["sent_score_final"].to_numpy(dtype=float)),
                                     c["w_rep"].to_numpy(dtype=float))) if not c.empty else 0.0

        sent_rows.append((sym, sent_5d, sent_60d, ss_raw, sent_intensity, mixed_rate, price_action_rate))

    ss = pd.DataFrame(
        sent_rows,
        columns=["symbol", "sent_5d", "sent_60d", "SS_raw", "sent_intensity", "mixed_rate", "price_action_rate"],
    )

    # --- Event intensity (EI_raw) ---
    # EI_raw = sum(min(sev_final,2.5)) over current week clusters
    ei_rows = []
    # Use canonical symbol set (no duplicates, consistent case)
    symbols = pd.Series(universe["symbol"]).dropna().astype(str).unique().tolist()
    for sym in symbols:
        c = cur_sent[cur_sent["symbol"] == sym]
        if c.empty:
            ei_rows.append((sym, 0.0, 0, 0.0))
            continue
        sev = c["sev_final"].astype(float).to_numpy()
        ei_raw = float(np.sum(np.minimum(sev, 2.5)))
        cnt2 = int(np.sum(sev >= 1.5))
        ei_max = float(np.max(sev)) if sev.size else 0.0
        ei_rows.append((sym, ei_raw, cnt2, ei_max))

    ei = pd.DataFrame(ei_rows, columns=["symbol", "EI_raw", "EI_cnt_sev2plus", "EI_max"])

    # Assemble per-symbol panel for current week
    # Sanity checks before merge - detailed diagnostics for debugging
    if "symbol" not in cur_counts.columns:
        raise ValueError(
            f"cur_counts missing 'symbol' column.\n"
            f"  Columns: {cur_counts.columns.tolist()}\n"
            f"  Index names: {cur_counts.index.names}\n"
            f"  Shape: {cur_counts.shape}\n"
            f"  Sample index: {cur_counts.index[:3].tolist() if len(cur_counts) > 0 else 'empty'}"
        )
    if "symbol" not in ns.columns:
        raise ValueError(
            f"ns missing 'symbol' column.\n"
            f"  Columns: {ns.columns.tolist()}\n"
            f"  Index names: {ns.index.names}\n"
            f"  Shape: {ns.shape}\n"
            f"  Sample symbols: {ns.index[:3].tolist() if len(ns) > 0 else 'empty'}"
        )
    
    # Sample join keys for verification
    print(f"  [DEBUG] Merging on 'symbol': cur_counts has {len(cur_counts)} rows, ns has {len(ns)} rows")
    if len(cur_counts) > 0:
        print(f"  [DEBUG] cur_counts sample symbols: {cur_counts['symbol'].head(3).tolist()}")
    if len(ns) > 0:
        print(f"  [DEBUG] ns sample symbols: {ns['symbol'].head(3).tolist()}")
    
    out = cur_counts.merge(ns[["symbol", "NS_raw_shrunk", "nov_hist_count", "nov_cur_reps"]], on="symbol", how="left")
    out = out.merge(ss, on="symbol", how="left")
    out = out.merge(ei, on="symbol", how="left")

    # Rename NS field
    out["NS_raw"] = out["NS_raw_shrunk"]
    out.drop(columns=["NS_raw_shrunk"], inplace=True)

    # Fill NaNs
    for col in ["NV_raw", "NA_raw", "NS_raw", "SS_raw", "EI_raw", "EC_raw"]:
        if col in out.columns:
            out[col] = out[col].astype(float).fillna(0.0)

    return out

# ----------------------------
# UPS scoring
# ----------------------------

def validate_feature_panel_contract(panel: pd.DataFrame, context: str = "feature_panel") -> None:
    """
    Validates that the feature panel meets the contract requirements before scoring.
    Raises ValueError with detailed diagnostics if any invariant is violated.
    """
    errors = []
    
    # 1. symbol must be a column, not index
    if "symbol" not in panel.columns:
        errors.append(f"❌ 'symbol' is not a column. Columns: {list(panel.columns)}")
        errors.append(f"   Index names: {panel.index.names}")
    
    # 2. week_ending_date must be present
    if "week_ending_date" not in panel.columns:
        errors.append(f"❌ 'week_ending_date' missing. Columns: {list(panel.columns)}")
    
    # 3. No unexpected 'index' column (indicates unnamed index materialization)
    if "index" in panel.columns:
        errors.append(f"❌ Unexpected 'index' column found (unnamed index materialization)")
        errors.append(f"   Columns: {list(panel.columns)}")
    
    # 4. Required feature columns for scoring
    required_features = ["NV_raw", "NA_raw", "NS_raw", "SS_raw", "EI_raw", "EC_raw"]
    missing_features = [f for f in required_features if f not in panel.columns]
    if missing_features:
        errors.append(f"❌ Missing required feature columns: {missing_features}")
    
    # If symbol exists, do additional checks
    if "symbol" in panel.columns and "week_ending_date" in panel.columns:
        # 5. Check for duplicate (symbol, week_ending_date) pairs
        dup_count = panel.duplicated(["symbol", "week_ending_date"]).sum()
        if dup_count > 0:
            errors.append(f"❌ Found {dup_count} duplicate (symbol, week_ending_date) pairs")
            # Show sample duplicates
            dups = panel[panel.duplicated(["symbol", "week_ending_date"], keep=False)]
            errors.append(f"   Sample duplicates:\n{dups[['symbol', 'week_ending_date']].head(10)}")
    
    # Raise if any errors
    if errors:
        error_msg = f"\n{context} contract validation failed:\n" + "\n".join(errors)
        error_msg += f"\n\nPanel shape: {panel.shape}"
        error_msg += f"\nPanel dtypes:\n{panel.dtypes}"
        raise ValueError(error_msg)
    
    # Success - print diagnostics
    print(f"✓ {context} contract validated:")
    print(f"  - Shape: {panel.shape}")
    print(f"  - Columns: {list(panel.columns)}")
    print(f"  - Index: {panel.index.names}")
    print(f"  - Has 'symbol': True (column)")
    print(f"  - Has 'week_ending_date': True")
    print(f"  - No 'index' column: True")
    print(f"  - Required features present: {len(required_features)}/{len(required_features)}")
    if "symbol" in panel.columns and "week_ending_date" in panel.columns:
        print(f"  - Unique (symbol, week) pairs: {len(panel)} (no duplicates)")


def compute_scores(
    panel: pd.DataFrame,
    mkt: pd.DataFrame,
    universe: pd.DataFrame,
):
    features = df[features_cols].copy()

    scores_cols = [
        "symbol","sector","asof_date",
        "NV","NA","NS","SS","EVS",
        "IFS","PD_up_raw","PD_down_raw","MCS_up","MCS_down",
        "UPS_raw","DPS_raw","UPS_adj","DPS_adj",
        "rank_UPS","rank_DPS",
        "VR_pct",
    ]
    scores = df[[c for c in scores_cols if c in df.columns]].copy()

    return features, scores

# ----------------------------
# Main
# ----------------------------

from typing import Optional

def run(
    universe_csv: Path,
    week_end: str,
    lookback_weeks: int = 12,
    regime_id: str = "news-novelty-v1",
    schema_id: str = "news-novelty-v1b",
    paths: Optional[Paths] = None,
    force: bool = False,
):

    _debug_env_stamp()
    if paths is None:
        paths = default_paths(regime_id=regime_id, schema_id=schema_id)

    # Output path for skip logic
    out_score_dir = paths.out_scores_dir / f"week_ending={week_end}"
    score_path = out_score_dir / "scores_weekly.parquet"
    from .reuse import should_skip
    if should_skip(score_path, force):
        print(f"SKIP: {score_path} exists and --force not set.")
        return
    # Schema selection logic: args.schema > regime config default_schema > fallback
    import yaml
    regime_cfg_path = Path("config/regimes") / f"{regime_id}.yaml"
    regime_cfg = {}
    if regime_cfg_path.exists():
        with open(regime_cfg_path, "r") as f:
            regime_cfg = yaml.safe_load(f)
    default_schema = regime_cfg.get("default_schema") if regime_cfg else None
    if schema_id is None:
        if default_schema:
            schema_id = default_schema
        else:
            schema_id = "news-novelty-v1"

    # Load scoring schema
    schema = load_schema(schema_id)
    print(f"📋 Using schema: {schema.schema_id} (hash: {schema.content_hash})")
    print(f"   Weights: {schema.get_weights()}")
    print(f"   Skip rules: {schema.get_skip_rules()}")


    week_end_et = parse_week_end(week_end)

    universe = load_universe(universe_csv)
    universe_syms = universe["symbol"].tolist()

    # Load market candles
    if not paths.market_daily_path.exists():
        raise FileNotFoundError(
            f"Missing daily candles file: {paths.market_daily_path}\n"
            "Expected columns: symbol,date,o,h,l,c,v and must include SPY."
        )
    candles_daily = pd.read_parquet(paths.market_daily_path)

    mkt = market_features_for_week(candles_daily, universe_syms, week_end)

    panel = build_news_feature_panel(
        paths=paths,
        universe=universe,
        week_end_et=week_end_et,
        week_end=week_end,
        lookback_weeks=lookback_weeks,
    )

    # Validate panel contract before scoring
    validate_feature_panel_contract(panel, context="build_news_feature_panel output")

    print("▶ Computing scores from validated feature panel...")
    features, scores = compute_scores(panel, mkt, universe)

    # Scoring self-check diagnostics
    print("\n" + "="*60)
    print("📊 SCORING SELF-CHECK")
    print("="*60)


    # Check for NaN in key score columns
    scoring_features = ["NV", "NA", "NS", "SS", "EVS", "UPS_adj", "DPS_adj", "VR_pct"]
    nan_pcts = {}
    for feat in scoring_features:
        if feat in scores.columns:
            nan_pct = scores[feat].isna().sum() / len(scores) * 100
            nan_pcts[feat] = nan_pct

    # Show top 10 worst NaN offenders
    if nan_pcts:
        print(f"\nNaN% per feature (top 10 worst):")
        sorted_nans = sorted(nan_pcts.items(), key=lambda x: x[1], reverse=True)[:10]
        for feat, pct in sorted_nans:
            print(f"  {feat:15s}: {pct:5.1f}%")

    # Score distribution
    print(f"\nScores DataFrame:")
    print(f"  Rows: {len(scores)}")
    print(f"  Symbols eligible for scoring: {scores['symbol'].nunique() if 'symbol' in scores.columns else 'N/A'}")

    if "UPS_adj" in scores.columns:
        ups_valid = scores["UPS_adj"].notna() & np.isfinite(scores["UPS_adj"])
        print(f"\nUPS_adj distribution:")
        print(f"  Valid (finite, non-NaN): {ups_valid.sum()} / {len(scores)} ({ups_valid.sum()/len(scores)*100:.1f}%)")
        if ups_valid.any():
            valid_ups = scores.loc[ups_valid, "UPS_adj"]
            print(f"  Min:    {valid_ups.min():.4f}")
            print(f"  Median: {valid_ups.median():.4f}")
            print(f"  Max:    {valid_ups.max():.4f}")

            # Top 5 and bottom 5
            top5 = scores.nlargest(5, "UPS_adj")[["symbol", "UPS_adj"]]
            bottom5 = scores.nsmallest(5, "UPS_adj")[["symbol", "UPS_adj"]]
            print(f"\nTop 5 symbols by UPS_adj:")
            for idx, row in top5.iterrows():
                print(f"  {row['symbol']:6s}: {row['UPS_adj']:7.4f}")
            print(f"\nBottom 5 symbols by UPS_adj:")
            for idx, row in bottom5.iterrows():
                print(f"  {row['symbol']:6s}: {row['UPS_adj']:7.4f}")

    print("="*60 + "\n")

    # Write outputs with regime-namespaced paths
    out_feat_dir = paths.out_features_dir / f"week_ending={week_end}"
    out_score_dir = paths.out_scores_dir / f"week_ending={week_end}"
    out_feat_dir.mkdir(parents=True, exist_ok=True)
    out_score_dir.mkdir(parents=True, exist_ok=True)

    feat_path = out_feat_dir / "features_weekly.parquet"
    score_path = out_score_dir / "scores_weekly.parquet"

    # Use atomic writes to prevent corruption
    write_parquet_atomic(features, feat_path)
    write_parquet_atomic(scores, score_path)

    print(f"Wrote: {feat_path}")
    print(f"Wrote: {score_path}")
    print(f"Rows: features={len(features):,}, scores={len(scores):,}")

    # Write schema provenance
    schema_prov_path = write_schema_provenance(schema, out_score_dir)
    print(f"Wrote schema provenance: {schema_prov_path}")

    # Write report_meta.json with full provenance
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paths.root, text=True).strip()
    except Exception:
        git_sha = "unknown"

    # Count clusters and symbols for diagnostics
    cluster_count = None
    symbol_count = None
    try:
        # Try to count clusters from the features DataFrame if available
        if "count_5d_dedup" in features.columns:
            cluster_count = int(features["count_5d_dedup"].sum())
        symbol_count = int(features["symbol"].nunique())
    except Exception:
        pass

    report_meta = {
        "week_end": week_end,
        "regime": regime_id,
        "schema_id": schema.schema_id,
        "schema_hash": schema.content_hash,
        "git_sha": git_sha,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_features": len(features),
        "n_scores": len(scores),
        "cluster_count": cluster_count,
        "symbol_count": symbol_count,
    }

    meta_path = out_score_dir / "report_meta.json"
    meta_path.write_text(json.dumps(report_meta, indent=2))
    print(f"Wrote provenance: {meta_path}")
