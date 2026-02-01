# src/backtest_weekly.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def _pick_score_col(df: pd.DataFrame) -> str:
    """Pick the best available score column."""
    for c in ["UPS_adj", "score", "UPS", "UPS_raw"]:
        if c in df.columns:
            return c
    raise KeyError("No score column found. Expected one of: UPS_adj, score, UPS, UPS_raw")


def parse_date(s: str) -> date:
    return datetime.fromisoformat(s).date()


def iter_fridays(from_week_end: date, to_week_end: date) -> List[date]:
    """Inclusive list of week_end dates stepping by 7 days (we still call them 'Fridays' historically)."""
    out = []
    d = from_week_end
    while d <= to_week_end:
        out.append(d)
        d = d + timedelta(days=7)
    return out

def list_score_weeks(scores_dir: Path) -> List[date]:
    """
    Return sorted list of week_end dates that actually exist on disk
    as week_ending=YYYY-MM-DD/scores_weekly.parquet under scores_dir.
    """
    out: List[date] = []
    if not scores_dir.exists():
        return out

    for d in sorted(scores_dir.glob("week_ending=*")):
        if not d.is_dir():
            continue
        f = d / "scores_weekly.parquet"
        if not f.exists():
            continue

        week_str = d.name.split("week_ending=")[-1]
        try:
            out.append(parse_date(week_str))
        except Exception:
            continue

    return sorted(out)

def load_universe(universe_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(universe_csv)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    return df[["symbol", "sector"]].drop_duplicates()


def select_top_with_sector_cap(
    scores: pd.DataFrame,
    top_n: int = 20,
    sector_cap: int = 5,
    score_col: str = "UPS_adj",
) -> pd.DataFrame:
    """
    scores must have: symbol, sector, score_col
    Returns selected rows with equal weights.
    """
    if score_col not in scores.columns:
        score_col = _pick_score_col(scores)

    s = scores.sort_values(score_col, ascending=False).copy()

    counts: Dict[str, int] = {}
    picks = []
    for _, r in s.iterrows():
        sec = str(r.get("sector", "Unknown"))
        counts.setdefault(sec, 0)
        # If sector is unknown/blank, do NOT enforce the sector cap.
        # This prevents accidental 5-position portfolios when universe sectors are missing.
        if sec not in ("Unknown", "", "nan", "None"):
            if counts[sec] >= sector_cap:
                continue
        picks.append(r)
        counts[sec] += 1
        if len(picks) >= top_n:
            break

    if not picks:
        return pd.DataFrame(columns=list(scores.columns) + ["weight"])

    out = pd.DataFrame(picks).copy()
    out["weight"] = 1.0 / len(out)
    return out


def trading_week_dates(week_end: date) -> Tuple[date, date]:
    """
    Defines the holding week for a given signal week_end (typically Friday, but can be Thu on holiday weeks):
      Entry: next Monday (or next trading day open via candle alignment)
      Exit: that Friday (or prior trading day close via candle alignment)

    IMPORTANT: Do NOT assume week_end is Friday; compute next Monday robustly.
    """
    # weekday: Mon=0 ... Sun=6
    wd = week_end.weekday()

    # days until next Monday strictly AFTER week_end
    days_until_monday = (7 - wd) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    entry_target = week_end + timedelta(days=days_until_monday)

    # exit target is that same week's Friday (Mon + 4)
    exit_target = entry_target + timedelta(days=4)
    return entry_target, exit_target


def nearest_trading_date_on_or_after(trading_dates: List[date], target: date) -> Optional[date]:
    for d in trading_dates:
        if d >= target:
            return d
    return None


def nearest_trading_date_on_or_before(trading_dates: List[date], target: date) -> Optional[date]:
    for d in reversed(trading_dates):
        if d <= target:
            return d
    return None


def compute_symbol_return_mon_open_fri_close(
    candles: pd.DataFrame,
    symbol: str,
    entry_target: date,
    exit_target: date,
) -> Optional[float]:
    """
    Return = close(exit_day) / open(entry_day) - 1
    Handles holidays via nearest trading day on/after for entry, on/before for exit.
    """
    g = candles[candles["symbol"] == symbol].copy()
    if g.empty:
        return None
    g = g.sort_values("date")
    trading_dates = g["date"].tolist()

    entry_day = nearest_trading_date_on_or_after(trading_dates, entry_target)
    exit_day = nearest_trading_date_on_or_before(trading_dates, exit_target)

    if entry_day is None or exit_day is None or exit_day <= entry_day:
        return None

    row_entry = g[g["date"] == entry_day].iloc[0]
    row_exit = g[g["date"] == exit_day].iloc[0]

    o = float(row_entry["open"])
    c = float(row_exit["close"])
    if o <= 0:
        return None
    return c / o - 1.0


def portfolio_turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    """turnover = sum(|new - old|)/2"""
    keys = set(prev_w.keys()) | set(new_w.keys())
    s = 0.0
    for k in keys:
        s += abs(new_w.get(k, 0.0) - prev_w.get(k, 0.0))
    return 0.5 * s


# ----------------------------
# Paths / IO
# ----------------------------

@dataclass
class Paths:
    root: Path
    derived: Path
    scores_dir: Path
    market_daily: Path
    out_dir: Path


def default_paths(regime: str, schema: str, out_dir: Optional[Path] = None) -> Paths:
    root = Path(__file__).resolve().parents[1]
    derived = root / "data" / "derived"
    scores_dir = derived / "scores_weekly" / f"regime={regime}" / f"schema={schema}"
    return Paths(
        root=root,
        derived=derived,
        scores_dir=scores_dir,
        market_daily=derived / "market_daily" / "candles_daily.parquet",
        out_dir=out_dir if out_dir is not None else (derived / "backtest"),
    )


def load_scores_for_week(paths: Paths, week_end: str) -> pd.DataFrame:
    p = paths.scores_dir / f"week_ending={week_end}" / "scores_weekly.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing scores file: {p}")
    return pd.read_parquet(p)


# ----------------------------
# Main backtest
# ----------------------------

def run_backtest(
    universe_csv: Path,
    from_week_end: str,
    to_week_end: str,
    regime: str,
    schema: str,
    top_n: int = 20,
    sector_cap: int = 5,
    tc_roundtrip_bps: float = 30.0,
    out_dir: Optional[Path] = None,
    skip_missing_scores: bool = False,
):
    paths = default_paths(regime=regime, schema=schema, out_dir=out_dir)
    uni = load_universe(universe_csv)

    if not paths.market_daily.exists():
        raise FileNotFoundError(f"Missing candles file: {paths.market_daily}")
    candles = pd.read_parquet(paths.market_daily)
    candles["date"] = pd.to_datetime(candles["date"]).dt.date
    candles = candles.sort_values(["symbol", "date"])

    if "SPY" not in set(candles["symbol"].unique()):
        raise RuntimeError("candles_daily.parquet must include SPY for benchmark return")

    from_d = parse_date(from_week_end)
    to_d = parse_date(to_week_end)
    all_weeks = list_score_weeks(paths.scores_dir)
    weeks = [w for w in all_weeks if (w >= from_d and w <= to_d)]
    print(f"Backtest weeks (on disk) in range: {len(weeks)}")

    bt_rows = []
    pos_rows = []
    missing_score_weeks: List[str] = []

    prev_weights: Dict[str, float] = {}
    tc_rate = tc_roundtrip_bps / 10000.0  # bps -> fraction

    for w in weeks:
        w_str = w.isoformat()
        try:
            scores = load_scores_for_week(paths, w_str)
        except FileNotFoundError:
            if skip_missing_scores:
                missing_score_weeks.append(w_str)
                continue
            raise

        # attach sector from universe if missing/unknown
        if "sector" not in scores.columns or scores["sector"].isna().all():
            scores = scores.merge(uni, on="symbol", how="left")
        else:
            scores = scores.merge(uni, on="symbol", how="left", suffixes=("", "_u"))
            scores["sector"] = scores["sector"].fillna(scores["sector_u"])
            if "sector_u" in scores.columns:
                scores.drop(columns=["sector_u"], inplace=True)

        score_col = _pick_score_col(scores)
        selected = select_top_with_sector_cap(scores, top_n=top_n, sector_cap=sector_cap, score_col=score_col)
        if selected.empty:
            continue

        new_weights = {r["symbol"]: float(r["weight"]) for _, r in selected.iterrows()}

        entry_target, exit_target = trading_week_dates(w)

        pret = 0.0
        missing = 0
        for _, r in selected.iterrows():
            sym = r["symbol"]
            wgt = float(r["weight"])
            ret = compute_symbol_return_mon_open_fri_close(candles, sym, entry_target, exit_target)
            if ret is None or not np.isfinite(ret):
                missing += 1
                continue
            pret += wgt * float(ret)

            pos_rows.append({
                "signal_week_end": w_str,
                "hold_entry_target": entry_target.isoformat(),
                "hold_exit_target": exit_target.isoformat(),
                "symbol": sym,
                "sector": r.get("sector", "Unknown"),
                "weight": wgt,
                "score_col": score_col,
                "score": float(r.get(score_col, np.nan)),
                "ret_mon_open_fri_close": float(ret),
            })

        spy_ret = compute_symbol_return_mon_open_fri_close(candles, "SPY", entry_target, exit_target)
        spy_ret = float(spy_ret) if spy_ret is not None else np.nan

        to = portfolio_turnover(prev_weights, new_weights)
        tcost = to * tc_rate

        pret_net = pret - tcost
        active_net = pret_net - (spy_ret if np.isfinite(spy_ret) else 0.0)

        bt_rows.append({
            "signal_week_end": w_str,
            "hold_entry_target": entry_target.isoformat(),
            "hold_exit_target": exit_target.isoformat(),
            "n_positions": int(len(selected)),
            "missing_returns": int(missing),
            "score_col": score_col,
            "gross_return": float(pret),
            "turnover": float(to),
            "tcost": float(tcost),
            "net_return": float(pret_net),
            "spy_return": float(spy_ret) if np.isfinite(spy_ret) else np.nan,
            "active_net_return": float(active_net),
        })

        prev_weights = new_weights

    bt = pd.DataFrame(bt_rows)
    pos = pd.DataFrame(pos_rows)

    paths.out_dir.mkdir(parents=True, exist_ok=True)
    bt_path = paths.out_dir / "bt_weekly.parquet"
    pos_path = paths.out_dir / "bt_positions.parquet"
    bt.to_parquet(bt_path, index=False)
    pos.to_parquet(pos_path, index=False)

    print(f"Wrote: {bt_path} ({len(bt):,} rows)")
    print(f"Wrote: {pos_path} ({len(pos):,} rows)")
    if missing_score_weeks:
        print(f"NOTE: skipped {len(missing_score_weeks)} week(s) with missing scores: {missing_score_weeks}")
    if not bt.empty:
        print(bt.tail(10).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True, help="Path to sp500_universe.csv")
    ap.add_argument("--from_week_end", required=True, help="YYYY-MM-DD (week_end stepping by 7 days)")
    ap.add_argument("--to_week_end", required=True, help="YYYY-MM-DD (week_end stepping by 7 days)")
    ap.add_argument("--regime", required=True, help="Regime name (e.g. news-novelty-v1)")
    ap.add_argument("--schema", required=True, help="Schema name (e.g. news-novelty-v1b)")
    ap.add_argument("--top_n", type=int, default=20)
    ap.add_argument("--sector_cap", type=int, default=5)
    ap.add_argument("--tc_bps", type=float, default=30.0, help="Round-trip transaction cost in bps, applied on turnover")
    ap.add_argument("--out_dir", default=None, help="Output directory for backtest artifacts (default: data/derived/backtest)")
    ap.add_argument("--skip_missing_scores", action="store_true", help="Skip weeks that have no scores_weekly.parquet instead of failing")
    args = ap.parse_args()

    run_backtest(
        universe_csv=Path(args.universe),
        from_week_end=args.from_week_end,
        to_week_end=args.to_week_end,
        regime=args.regime,
        schema=args.schema,
        top_n=args.top_n,
        sector_cap=args.sector_cap,
        tc_roundtrip_bps=args.tc_bps,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        skip_missing_scores=bool(args.skip_missing_scores),
    )
