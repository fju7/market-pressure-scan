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

from .io_atomic import write_parquet_atomic
from .reuse import should_skip
from .scoring_schema import load_schema, write_schema_provenance

NY = tz.gettz("America/New_York")


# =============================================================================
# Debug / diagnostics
# =============================================================================
def _debug_env_stamp() -> None:
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


# =============================================================================
# Paths
# =============================================================================
@dataclass(frozen=True)
class Paths:
    root: Path
    derived: Path

    market_daily_path: Path
    company_news_dir: Path
    news_clusters_dir: Path
    rep_enriched_dir: Path

    out_features_dir: Path
    out_scores_dir: Path

    regime_id: str
    schema_id: str


def default_paths(regime_id: str, schema_id: str) -> Paths:
    root = Path(__file__).resolve().parents[1]
    derived = root / "data" / "derived"

    market_daily_path = derived / "market_daily" / "candles_daily.parquet"
    company_news_dir = derived / "company_news"
    news_clusters_dir = derived / "news_clusters"
    rep_enriched_dir = derived / "rep_enriched"

    out_features_dir = derived / "features_weekly" / f"regime={regime_id}" / f"schema={schema_id}"
    out_scores_dir = derived / "scores_weekly" / f"regime={regime_id}" / f"schema={schema_id}"

    return Paths(
        root=root,
        derived=derived,
        market_daily_path=market_daily_path,
        company_news_dir=company_news_dir,
        news_clusters_dir=news_clusters_dir,
        rep_enriched_dir=rep_enriched_dir,
        out_features_dir=out_features_dir,
        out_scores_dir=out_scores_dir,
        regime_id=regime_id,
        schema_id=schema_id,
    )


# =============================================================================
# Time helpers
# =============================================================================
def parse_week_end(s: str) -> datetime:
    d = datetime.fromisoformat(s)
    return datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=NY)


def list_prior_weeks(week_end_et: datetime, n_weeks: int) -> List[str]:
    out: List[str] = []
    for k in range(n_weeks):
        dd = week_end_et - timedelta(days=7 * k)
        out.append(dd.date().isoformat())
    return out


# =============================================================================
# IO helpers
# =============================================================================
def load_week_parquet_optional(base_dir: Path, week_end: str, filename: str) -> Optional[pd.DataFrame]:
    p = base_dir / f"week_ending={week_end}" / filename
    if not p.exists():
        return None
    return pd.read_parquet(p)


def load_universe(universe_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(universe_csv)
    if "symbol" not in df.columns:
        raise ValueError(f"Universe CSV missing 'symbol' column: {universe_csv}")
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    return df[["symbol", "sector"]].drop_duplicates()


# =============================================================================
# Math helpers
# =============================================================================
def winsorize(x: np.ndarray, zcap: float = 4.0) -> np.ndarray:
    return np.clip(x, -zcap, zcap)


def zscore_series(s: pd.Series, zcap: float = 4.0) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").astype(float).to_numpy()
    mu = np.nanmean(vals)
    sd = np.nanstd(vals)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    z = (vals - mu) / sd
    z = winsorize(z, zcap=zcap)
    return pd.Series(z, index=s.index)


def to_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


# =============================================================================
# Market features
# =============================================================================
def market_features_for_week(
    candles_daily: pd.DataFrame,
    universe_syms: List[str],
    week_end: str,
) -> pd.DataFrame:
    """
    Minimal market features:
      - RV60 and VR_pct (ranked vol)
      - AR5 and z_AR5 (5-trading-day return ending at asof_date)

    Expected columns:
      - symbol, date, close, volume (or Finnhub shorthand c/v)
    """
    cd = candles_daily.copy()
    if "symbol" not in cd.columns or "date" not in cd.columns:
        raise ValueError("candles_daily.parquet must include columns: symbol,date")

    cd["symbol"] = cd["symbol"].astype(str).str.upper().str.strip()
    cd["date"] = pd.to_datetime(cd["date"]).dt.date

    if "close" not in cd.columns and "c" in cd.columns:
        cd = cd.rename(columns={"c": "close"})
    if "volume" not in cd.columns and "v" in cd.columns:
        cd = cd.rename(columns={"v": "volume"})

    target = datetime.fromisoformat(week_end).date()
    syms = set(universe_syms) | {"SPY"}
    cd = cd[cd["symbol"].isin(syms)].copy()

    # last available date <= target per symbol
    last_dates = (
        cd[cd["date"] <= target]
        .groupby("symbol", as_index=False)["date"]
        .max()
        .rename(columns={"date": "asof_date"})
    )
    if last_dates.empty:
        raise RuntimeError("No trading data found for universe symbols at or before week_end.")

    def realized_vol_60(g: pd.DataFrame, end_date, lookback: int = 60) -> Optional[float]:
        gg = g[g["date"] <= end_date].sort_values("date")
        if len(gg) < lookback + 1:
            return None
        c = pd.to_numeric(gg["close"], errors="coerce").astype(float).to_numpy()
        c = c[-(lookback + 1) :]
        if len(c) < lookback + 1 or not np.isfinite(c).all():
            return None
        rets = np.diff(c) / (c[:-1] + 1e-12)
        if rets.size < 2:
            return None
        return float(np.nanstd(rets, ddof=1))

    def ar5_return(g: pd.DataFrame, end_date) -> Optional[float]:
        gg = g[g["date"] <= end_date].sort_values("date")
        if len(gg) < 6:
            return None
        c = pd.to_numeric(gg["close"], errors="coerce").astype(float).to_numpy()
        c = c[-6:]
        if len(c) < 6 or not np.isfinite(c).all():
            return None
        return float((c[-1] / (c[0] + 1e-12)) - 1.0)

    rows: List[Tuple[str, str, float, float]] = []
    for sym in universe_syms:
        g = cd[cd["symbol"] == sym]
        if g.empty:
            continue
        asof_row = last_dates[last_dates["symbol"] == sym]
        if asof_row.empty:
            continue
        asof_date = asof_row["asof_date"].iloc[0]

        rv60 = realized_vol_60(g, asof_date, lookback=60)
        ar5 = ar5_return(g, asof_date)
        if rv60 is None:
            continue
        if ar5 is None:
            ar5 = 0.0
        rows.append((sym, str(asof_date), float(rv60), float(ar5)))

    mkt = pd.DataFrame(rows, columns=["symbol", "asof_date", "RV60", "AR5"])
    if mkt.empty:
        return pd.DataFrame(columns=["symbol", "asof_date", "RV60", "VR_pct", "AR5", "z_AR5"])

    mkt["VR_pct"] = mkt["RV60"].rank(pct=True)
    mkt["z_AR5"] = zscore_series(mkt["AR5"])
    return mkt


# =============================================================================
# News feature panel (rep_enriched)
# =============================================================================
def compute_rep_sent_score_final(sent: dict) -> Tuple[float, float, str]:
    score = float(sent.get("sent_score", 0.0) or 0.0)
    conf = float(sent.get("confidence", 0.0) or 0.0)
    driver = str(sent.get("sent_driver", ""))

    if driver == "Market/Price-action":
        score *= 0.35
        conf *= 0.70
    elif driver == "Speculation/Opinion":
        score *= 0.65
        conf *= 0.85

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

    w_echo = 1.0 + 0.10 * math.log(1.0 + max(unique_sources, 0.0))
    w_echo = min(w_echo, 1.3)
    return float(sev_adj * w_echo)


def build_news_feature_panel(
    paths: Paths,
    universe: pd.DataFrame,
    week_end_et: datetime,
    week_end: str,
    lookback_weeks: int = 12,
) -> pd.DataFrame:
    weeks = list_prior_weeks(week_end_et, lookback_weeks)
    frames: List[pd.DataFrame] = []

    for w in weeks:
        dfw = load_week_parquet_optional(paths.rep_enriched_dir, w, "rep_enriched.parquet")
        if dfw is None or dfw.empty:
            continue
        frames.append(dfw)

    if not frames:
        raise FileNotFoundError(
            "No rep_enriched.parquet found in lookback. "
            f"Expected under {paths.rep_enriched_dir}/week_ending=<week>/rep_enriched.parquet"
        )

    df = pd.concat(frames, ignore_index=True)

    if "symbol" not in df.columns:
        raise ValueError("rep_enriched missing required column: symbol")

    if "week_ending_date" not in df.columns:
        if "week_end" in df.columns:
            df = df.rename(columns={"week_end": "week_ending_date"})
        else:
            raise ValueError("rep_enriched missing required column: week_ending_date")

    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["week_ending_date"] = pd.to_datetime(df["week_ending_date"]).dt.normalize()

    # cluster_id fallback
    if "cluster_id" not in df.columns:
        if "cluster" in df.columns:
            df = df.rename(columns={"cluster": "cluster_id"})
        else:
            df["cluster_id"] = np.arange(len(df))

    # numeric defaults
    if "cluster_size" not in df.columns:
        df["cluster_size"] = 1.0
    if "unique_sources" not in df.columns:
        df["unique_sources"] = 1.0

    # -------------------------------------------------------------------------
    # Parse sentiment_json / event_json into dicts (your rep_enriched format)
    # -------------------------------------------------------------------------
    def _parse_json_cell(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return {}
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return {}
            try:
                return json.loads(s)
            except Exception:
                return {}
        return {}

    if "sentiment_json" in df.columns:
        df["sentiment"] = df["sentiment_json"].apply(_parse_json_cell)
    elif "sentiment" not in df.columns:
        df["sentiment"] = [{} for _ in range(len(df))]

    if "event_json" in df.columns:
        df["event"] = df["event_json"].apply(_parse_json_cell)
    elif "event" not in df.columns:
        df["event"] = [{} for _ in range(len(df))]

    # compute final per-rep fields
    sent_final = df["sentiment"].apply(lambda d: compute_rep_sent_score_final(d if isinstance(d, dict) else {}))
    df["sent_score_final"] = sent_final.apply(lambda t: t[0])
    df["sent_driver"] = sent_final.apply(lambda t: t[2])

    df["sev_final"] = df.apply(
        lambda r: compute_event_sev_final(
            r["event"] if isinstance(r["event"], dict) else {},
            unique_sources=to_float(r.get("unique_sources", 1.0), 1.0),
        ),
        axis=1,
    )

    # split current week vs history
    week_end_ts = pd.to_datetime(week_end).normalize()
    df_cur = df[df["week_ending_date"] == week_end_ts].copy()
    df_hist = df[df["week_ending_date"] != week_end_ts].copy()

    if df_cur.empty:
        raise ValueError(
            f"rep_enriched has no rows for week_end={week_end_ts.date()}. "
            f"Available weeks: {sorted(df['week_ending_date'].dt.date.unique().tolist())}"
        )
    # current counts per symbol (dedup clusters)
    cur_counts = (
        df_cur.groupby("symbol", as_index=False)
        .agg(
            # "dedup clusters" = unique clusters, not sum of cluster sizes
            count_5d_dedup=("cluster_id", "nunique"),
            # keep EC_raw as "average cluster size" (even if placeholder right now)
            EC_raw=("cluster_size", "mean"),
            unique_sources_mean=("unique_sources", "mean"),
            # optional extra diagnostics (helpful while debugging)
            rep_rows_5d=("cluster_id", "size"),
        )
    )

    hist_by_week = (
        df_hist.groupby(["week_ending_date", "symbol"], as_index=False)
        .agg(
            # weekly dedup clusters = unique cluster_ids that week
            count_week=("cluster_id", "nunique")
        )
        .sort_values(["symbol", "week_ending_date"])
    )

    def add_roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("week_ending_date")
        g["count_20d_dedup"] = g["count_week"].rolling(window=4, min_periods=1).sum()
        g["count_60d_dedup"] = g["count_week"].rolling(window=12, min_periods=1).sum()
        return g

    if not hist_by_week.empty:
        hist_by_week = hist_by_week.groupby("symbol", group_keys=False).apply(add_roll)

        # If pandas suggestion or version puts symbol into the index, recover it
        if "symbol" not in hist_by_week.columns:
            hist_by_week = hist_by_week.reset_index()

        hist_latest = (
            hist_by_week.groupby("symbol", as_index=False)
            .tail(1)[["symbol", "count_20d_dedup", "count_60d_dedup"]]
        )
    else:
        hist_latest = pd.DataFrame(columns=["symbol", "count_20d_dedup", "count_60d_dedup"])

    cur = cur_counts.merge(hist_latest, on="symbol", how="left")

    cur["count_20d_dedup"] = pd.to_numeric(cur["count_20d_dedup"], errors="coerce").fillna(0.0)
    cur["count_60d_dedup"] = pd.to_numeric(cur["count_60d_dedup"], errors="coerce").fillna(0.0)

    # novelty proxies
    # Novelty proxy: compare recent vs baseline *rates* (prevents degeneracy when counts are small/quantized)
    recent_rate = (cur["count_5d_dedup"] / 5.0).astype(float)
    baseline_rate = (cur["count_60d_dedup"] / 60.0).astype(float)
    cur["NV_raw"] = np.log1p((recent_rate + 1e-6) / (baseline_rate + 1e-6))
    cur["NA_raw"] = np.log1p((recent_rate + 1e-6) / ((cur["count_20d_dedup"] / 20.0).astype(float) + 1e-6))


    # sentiment signal: current week mean minus history mean
    sent_cur = df_cur.groupby("symbol", as_index=False).agg(sent_5d=("sent_score_final", "mean"))
    if not df_hist.empty:
        sent_hist = df_hist.groupby("symbol", as_index=False).agg(sent_60d=("sent_score_final", "mean"))
    else:
        sent_hist = pd.DataFrame({"symbol": [], "sent_60d": []})

    ss = sent_cur.merge(sent_hist, on="symbol", how="left")
    ss["sent_60d"] = pd.to_numeric(ss["sent_60d"], errors="coerce").fillna(0.0)
    ss["SS_raw"] = ss["sent_5d"] - ss["sent_60d"]

    # event intensity: sum of sev_final this week
    ei = df_cur.groupby("symbol", as_index=False).agg(EI_raw=("sev_final", "sum"))

    out = cur.merge(ss[["symbol", "SS_raw"]], on="symbol", how="left").merge(ei, on="symbol", how="left")
    out["EI_raw"] = pd.to_numeric(out["EI_raw"], errors="coerce").fillna(0.0)
    out["SS_raw"] = pd.to_numeric(out["SS_raw"], errors="coerce").fillna(0.0)

    # placeholder kept for compatibility
    out["NS_raw"] = 0.0
    out["week_ending_date"] = week_end_ts

    # ensure all universe symbols exist
    out = universe[["symbol"]].merge(out, on="symbol", how="left")

    for col in ["NV_raw", "NA_raw", "NS_raw", "SS_raw", "EI_raw", "EC_raw"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out


# =============================================================================
# Validation + scoring
# =============================================================================
def validate_feature_panel_contract(panel: pd.DataFrame, context: str = "feature_panel") -> None:
    errors: List[str] = []

    if "symbol" not in panel.columns:
        errors.append(f"'symbol' is not a column. Columns={list(panel.columns)}")

    if "week_ending_date" not in panel.columns:
        errors.append(f"'week_ending_date' missing. Columns={list(panel.columns)}")

    if "index" in panel.columns:
        errors.append("Unexpected 'index' column found (index materialization).")

    required = ["NV_raw", "NA_raw", "NS_raw", "SS_raw", "EI_raw", "EC_raw"]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        errors.append(f"Missing required feature columns: {missing}")

    if errors:
        raise ValueError(f"{context} contract validation failed:\n- " + "\n- ".join(errors))


def compute_scores(
    panel: pd.DataFrame,
    mkt: pd.DataFrame,
    universe: pd.DataFrame,
    schema_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = panel.merge(universe, on="symbol", how="left")

    # zscore raw features cross-sectionally
    for c in ["NV_raw", "NA_raw", "NS_raw", "SS_raw", "EI_raw", "EC_raw"]:
        df[f"z_{c}"] = zscore_series(df[c])

    # market merge
    if not mkt.empty:
        df = df.merge(mkt[["symbol", "asof_date", "VR_pct", "AR5", "z_AR5"]], on="symbol", how="left")
    else:
        df["asof_date"] = None
        df["VR_pct"] = np.nan
        df["AR5"] = np.nan
        df["z_AR5"] = 0.0

    df["VR_pct"] = pd.to_numeric(df["VR_pct"], errors="coerce").fillna(0.0)
    df["z_AR5"] = pd.to_numeric(df["z_AR5"], errors="coerce").fillna(0.0)

    schema = load_schema(schema_id)
    weights: Dict[str, float] = schema.get_weights()

    def pick_weight(keys: List[str]) -> float:
        for k in keys:
            if k in weights:
                return float(weights[k])
        return 0.0

    # Branch 1: factor schema
    has_factor_schema = any(k in weights for k in ["novelty", "event_intensity", "sentiment", "divergence"])
    if has_factor_schema:
        w_nov = pick_weight(["novelty"])
        w_ei = pick_weight(["event_intensity"])
        w_sent = pick_weight(["sentiment"])
        w_div = pick_weight(["divergence"])

        if (w_nov + w_ei + w_sent + w_div) == 0.0:
            raise ValueError(
                f"Schema {schema_id} has factor keys but resolved all factor weights to 0. "
                f"weights keys={list(weights.keys())}"
            )

        df["novelty_factor"] = 0.5 * df["z_NV_raw"] + 0.5 * df["z_NA_raw"]
        df["event_intensity_factor"] = 0.5 * df["z_EI_raw"] + 0.5 * df["z_EC_raw"]
        df["sentiment_factor"] = df["z_SS_raw"]
        df["divergence_factor"] = (df["z_AR5"] - df["z_SS_raw"]).abs()
        df["divergence"] = df["divergence_factor"]

        df["UPS_raw"] = (
            w_nov * df["novelty_factor"]
            + w_ei * df["event_intensity_factor"]
            + w_sent * df["sentiment_factor"]
            + w_div * df["divergence_factor"]
        )
    else:
        # Branch 2: legacy weights
        w_nv = pick_weight(["NV", "NV_raw"])
        w_na = pick_weight(["NA", "NA_raw"])
        w_ns = pick_weight(["NS", "NS_raw"])
        w_ss = pick_weight(["SS", "SS_raw"])
        w_ei = pick_weight(["EI", "EVS", "EI_raw"])
        w_ec = pick_weight(["EC", "EC_raw"])

        if (w_nv + w_na + w_ns + w_ss + w_ei + w_ec) == 0.0:
            raise ValueError(
                f"Schema {schema_id} resolved all legacy weights to 0. "
                f"weights keys={list(weights.keys())}"
            )

        df["UPS_raw"] = (
            w_nv * df["z_NV_raw"]
            + w_na * df["z_NA_raw"]
            + w_ns * df["z_NS_raw"]
            + w_ss * df["z_SS_raw"]
            + w_ei * df["z_EI_raw"]
            + w_ec * df["z_EC_raw"]
        )

    df["UPS_adj"] = df["UPS_raw"] * (1.0 - 0.25 * df["VR_pct"])
    df["rank_UPS"] = df["UPS_adj"].rank(ascending=False, method="first")

    features_cols = [
        "symbol",
        "week_ending_date",
        "count_5d_dedup",
        "count_20d_dedup",
        "count_60d_dedup",
        "unique_sources_mean",
        "NV_raw",
        "NA_raw",
        "NS_raw",
        "SS_raw",
        "EI_raw",
        "EC_raw",
        "z_NV_raw",
        "z_NA_raw",
        "z_NS_raw",
        "z_SS_raw",
        "z_EI_raw",
        "z_EC_raw",
        "VR_pct",
        "AR5",
        "z_AR5",
        "UPS_raw",
        "UPS_adj",
    ]
    if "novelty_factor" in df.columns:
        features_cols += ["novelty_factor", "event_intensity_factor", "sentiment_factor", "divergence_factor", "divergence"]

    features = df[[c for c in features_cols if c in df.columns]].copy()

    scores_cols = ["symbol", "sector", "asof_date", "UPS_raw", "UPS_adj", "rank_UPS", "VR_pct"]
    scores = df[[c for c in scores_cols if c in df.columns]].copy()

    return features, scores


# =============================================================================
# Main runner
# =============================================================================
def run(
    universe_csv: Path,
    week_end: str,
    lookback_weeks: int = 12,
    regime_id: str = "news-novelty-v1",
    schema_id: str = "news-novelty-v1b",
    force: bool = False,
) -> None:
    _debug_env_stamp()

    paths = default_paths(regime_id=regime_id, schema_id=schema_id)

    out_score_dir = paths.out_scores_dir / f"week_ending={week_end}"
    score_path = out_score_dir / "scores_weekly.parquet"

    if should_skip(score_path, force):
        print(f"SKIP: {score_path} exists and --force not set.")
        return

    week_end_et = parse_week_end(week_end)

    universe = load_universe(universe_csv)
    universe_syms = universe["symbol"].tolist()

    if not paths.market_daily_path.exists():
        raise FileNotFoundError(f"Missing daily candles file: {paths.market_daily_path}")

    candles_daily = pd.read_parquet(paths.market_daily_path)
    mkt = market_features_for_week(candles_daily, universe_syms, week_end)

    panel = build_news_feature_panel(
        paths=paths,
        universe=universe,
        week_end_et=week_end_et,
        week_end=week_end,
        lookback_weeks=lookback_weeks,
    )
    validate_feature_panel_contract(panel, context="build_news_feature_panel output")

    features, scores = compute_scores(panel, mkt, universe, schema_id=schema_id)

    out_feat_dir = paths.out_features_dir / f"week_ending={week_end}"
    out_feat_dir.mkdir(parents=True, exist_ok=True)
    out_score_dir.mkdir(parents=True, exist_ok=True)

    feat_path = out_feat_dir / "features_weekly.parquet"
    score_path = out_score_dir / "scores_weekly.parquet"

    write_parquet_atomic(features, feat_path)
    write_parquet_atomic(scores, score_path)

    print(f"Wrote: {feat_path}")
    print(f"Wrote: {score_path}")
    print(f"Rows: features={len(features):,}, scores={len(scores):,}")

    schema = load_schema(schema_id)
    schema_prov_path = write_schema_provenance(schema, out_score_dir)
    print(f"Wrote schema provenance: {schema_prov_path}")

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paths.root, text=True).strip()
    except Exception:
        git_sha = "unknown"

    report_meta = {
        "week_end": week_end,
        "regime": regime_id,
        "schema_id": schema.schema_id,
        "schema_hash": schema.content_hash,
        "git_sha": git_sha,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_features": int(len(features)),
        "n_scores": int(len(scores)),
    }

    meta_path = out_score_dir / "report_meta.json"
    meta_path.write_text(json.dumps(report_meta, indent=2))
    print(f"Wrote provenance: {meta_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True, type=Path)
    ap.add_argument("--week_end", required=True)
    ap.add_argument("--lookback_weeks", type=int, default=12)
    ap.add_argument("--regime", default="news-novelty-v1")
    ap.add_argument("--schema", default="news-novelty-v1b")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run(
        universe_csv=args.universe,
        week_end=args.week_end,
        lookback_weeks=args.lookback_weeks,
        regime_id=args.regime,
        schema_id=args.schema,
        force=args.force,
    )


if __name__ == "__main__":
    main()
