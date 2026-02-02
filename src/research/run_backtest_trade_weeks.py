from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_date(s: str) -> date:
    return datetime.fromisoformat(s).date()


def trading_week_dates(signal_week_end: date) -> Tuple[date, date]:
    # Signal is Friday close. Hold is next Monday open to next Friday close.
    entry_target = signal_week_end + timedelta(days=3)
    exit_target = signal_week_end + timedelta(days=7)
    return entry_target, exit_target


def nearest_on_or_after(dates: List[date], target: date) -> Optional[date]:
    for d in dates:
        if d >= target:
            return d
    return None


def nearest_on_or_before(dates: List[date], target: date) -> Optional[date]:
    for d in reversed(dates):
        if d <= target:
            return d
    return None


def ret_mon_open_fri_close(candles: pd.DataFrame, symbol: str, entry_target: date, exit_target: date) -> Optional[float]:
    g = candles[candles["symbol"] == symbol].copy()
    if g.empty:
        return None
    g = g.sort_values("date")
    dts = g["date"].tolist()
    entry_day = nearest_on_or_after(dts, entry_target)
    exit_day = nearest_on_or_before(dts, exit_target)
    if entry_day is None or exit_day is None or exit_day <= entry_day:
        return None
    row_entry = g[g["date"] == entry_day].iloc[0]
    row_exit = g[g["date"] == exit_day].iloc[0]
    o = float(row_entry["open"])
    c = float(row_exit["close"])
    if o <= 0:
        return None
    return c / o - 1.0


def load_universe(universe_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(universe_csv)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    return df[["symbol", "sector"]].drop_duplicates()


def pick_score_col(cols: List[str]) -> str:
    for c in ["UPS_adj", "score", "UPS", "UPS_raw"]:
        if c in cols:
            return c
    raise KeyError("No score column found. Expected one of: UPS_adj, score, UPS, UPS_raw")


def select_top_with_sector_cap(scores: pd.DataFrame, top_n: int, sector_cap: int, score_col: str) -> pd.DataFrame:
    s = scores.sort_values(score_col, ascending=False).copy()

    counts: Dict[str, int] = {}
    picks = []
    for _, r in s.iterrows():
        sec = str(r.get("sector", "Unknown"))
        counts.setdefault(sec, 0)
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


def turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    keys = set(prev_w.keys()) | set(new_w.keys())
    return 0.5 * sum(abs(new_w.get(k, 0.0) - prev_w.get(k, 0.0)) for k in keys)


def spearman_no_scipy(x: pd.Series, y: pd.Series) -> float:
    # Spearman = Pearson corr of ranks
    rx = x.rank(method="average")
    ry = y.rank(method="average")
    return float(rx.corr(ry, method="pearson"))


@dataclass
class Paths:
    scores_root: Path
    reports_root: Path
    candles_path: Path
    out_dir: Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--universe", required=True)
    ap.add_argument("--trade_weeks_file", default="data/derived/analysis/trade_weeks.txt")
    ap.add_argument("--top_n", type=int, default=20)
    ap.add_argument("--sector_cap", type=int, default=5)
    ap.add_argument("--tc_bps", type=float, default=30.0)
    ap.add_argument("--out_csv", default="data/derived/analysis/perf_trade_weeks.csv")
    ap.add_argument("--out_ic_csv", default="data/derived/analysis/ic_trade_weeks.csv")
    args = ap.parse_args()

    paths = Paths(
        scores_root=Path("data/derived/scores_weekly") / f"regime={args.regime}" / f"schema={args.schema}",
        reports_root=Path("data/derived/reports"),
        candles_path=Path("data/derived/market_daily/candles_daily.parquet"),
        out_dir=Path("data/derived/analysis"),
    )

    if not paths.candles_path.exists():
        raise FileNotFoundError(f"Missing candles file: {paths.candles_path}")

    trade_weeks = [w.strip() for w in Path(args.trade_weeks_file).read_text().splitlines() if w.strip()]
    if not trade_weeks:
        raise RuntimeError(f"No trade weeks found in {args.trade_weeks_file}")

    uni = load_universe(Path(args.universe))

    candles = pd.read_parquet(paths.candles_path, columns=["symbol", "date", "open", "close"])
    candles["symbol"] = candles["symbol"].astype(str).str.upper().str.strip()
    candles["date"] = pd.to_datetime(candles["date"]).dt.date
    candles = candles.sort_values(["symbol", "date"])

    if "SPY" not in set(candles["symbol"].unique()):
        raise RuntimeError("candles_daily.parquet must include SPY for benchmark return")

    prev_weights: Dict[str, float] = {}
    tc_rate = args.tc_bps / 10000.0

    bt_rows = []
    pos_rows = []
    ic_rows = []

    for w in trade_weeks:
        sp = paths.scores_root / f"week_ending={w}" / "scores_weekly.parquet"
        if not sp.exists():
            raise FileNotFoundError(f"Missing scores parquet for trade week {w}: {sp}")

        scores = pd.read_parquet(sp)
        scores["symbol"] = scores["symbol"].astype(str).str.upper().str.strip()

        # attach sector
        scores = scores.merge(uni, on="symbol", how="left", suffixes=("", "_u"))
        if "sector" not in scores.columns or scores["sector"].isna().all():
            scores["sector"] = scores.get("sector_u", "Unknown")
        else:
            scores["sector"] = scores["sector"].fillna(scores["sector_u"])
        if "sector_u" in scores.columns:
            scores.drop(columns=["sector_u"], inplace=True)

        score_col = pick_score_col(list(scores.columns))

        selected = select_top_with_sector_cap(scores, top_n=args.top_n, sector_cap=args.sector_cap, score_col=score_col)
        if selected.empty:
            continue

        new_weights = {r["symbol"]: float(r["weight"]) for _, r in selected.iterrows()}

        week_end = parse_date(w)
        entry_target, exit_target = trading_week_dates(week_end)

        # portfolio return + positions
        pret = 0.0
        missing = 0

        for _, r in selected.iterrows():
            sym = r["symbol"]
            wgt = float(r["weight"])
            ret = ret_mon_open_fri_close(candles, sym, entry_target, exit_target)
            if ret is None or not np.isfinite(ret):
                missing += 1
                continue
            pret += wgt * float(ret)
            pos_rows.append({
                "signal_week_end": w,
                "hold_entry_target": entry_target.isoformat(),
                "hold_exit_target": exit_target.isoformat(),
                "symbol": sym,
                "sector": r.get("sector", "Unknown"),
                "weight": wgt,
                "score_col": score_col,
                "score_value": float(r.get(score_col, np.nan)),
                "ret_mon_open_fri_close": float(ret),
            })

        spy_ret = ret_mon_open_fri_close(candles, "SPY", entry_target, exit_target)
        spy_ret = float(spy_ret) if spy_ret is not None else np.nan

        to = turnover(prev_weights, new_weights)
        tcost = to * tc_rate
        pret_net = pret - tcost
        active_net = pret_net - (spy_ret if np.isfinite(spy_ret) else 0.0)

        bt_rows.append({
            "signal_week_end": w,
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

        # IC (Spearman) across all available symbols this week
        # compute forward returns for all symbols in scores
        rets = []
        for sym in scores["symbol"].tolist():
            r = ret_mon_open_fri_close(candles, sym, entry_target, exit_target)
            rets.append(np.nan if r is None else float(r))
        tmp = scores[[score_col]].copy()
        tmp["fwd_ret"] = rets
        tmp = tmp.dropna()
        if len(tmp) >= 20:
            ic = spearman_no_scipy(tmp[score_col], tmp["fwd_ret"])
        else:
            ic = np.nan
        ic_rows.append({"week_end": w, "score_col": score_col, "n": int(len(tmp)), "ic_spearman": ic})

        prev_weights = new_weights

    bt = pd.DataFrame(bt_rows).sort_values("signal_week_end")
    pos = pd.DataFrame(pos_rows).sort_values(["signal_week_end", "symbol"])
    ic = pd.DataFrame(ic_rows).sort_values("week_end")

    paths.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv)
    out_ic = Path(args.out_ic_csv)

    bt.to_csv(out_csv, index=False)
    ic.to_csv(out_ic, index=False)

    print("Wrote:", out_csv, f"({len(bt)} rows)")
    print("Wrote:", out_ic, f"({len(ic)} rows)")

    print("\nPERF (trade weeks):")
    cols = ["signal_week_end","n_positions","missing_returns","score_col","turnover","tcost","net_return","spy_return","active_net_return"]
    if len(bt):
        print(bt[cols].to_string(index=False))
    else:
        print("(no rows)")

    print("\nIC (trade weeks):")
    if len(ic):
        print(ic.to_string(index=False))
        v = ic["ic_spearman"].dropna()
        if len(v):
            print("\nIC summary:")
            print("  weeks:", len(v))
            print("  mean:", float(v.mean()))
            print("  median:", float(v.median()))
            print("  frac > 0:", float((v > 0).mean()))
    else:
        print("(no rows)")

    # Extra: show top missing reason if all missing
    if len(bt) and bt["missing_returns"].max() >= 1:
        worst = bt.sort_values("missing_returns", ascending=False).head(1).iloc[0]
        if int(worst["missing_returns"]) == int(worst["n_positions"]):
            print("\nWARNING: At least one week had missing_returns == n_positions.")
            print("This usually means candles_daily.parquet lacks symbol/date coverage for that holding window.")


if __name__ == "__main__":
    main()
