from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Small utilities
# ----------------------------

def _to_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).normalize()


def _equity_curve(returns: pd.Series) -> pd.Series:
    # returns are simple weekly returns
    returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    return (1.0 + returns).cumprod()


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def _summary_stats(returns: pd.Series, ann_factor: float = 52.0) -> Dict[str, float]:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) == 0:
        return {
            "n": 0,
            "mean_weekly": float("nan"),
            "std_weekly": float("nan"),
            "sharpe_annualized": float("nan"),
            "hit_rate": float("nan"),
            "cum_return": float("nan"),
            "max_drawdown": float("nan"),
        }

    mean_w = float(r.mean())
    std_w = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    sharpe = (
        float((mean_w / std_w) * np.sqrt(ann_factor))
        if (std_w and np.isfinite(std_w) and std_w > 0)
        else float("nan")
    )
    hit = float((r > 0).mean())
    eq = _equity_curve(r)
    cum = float(eq.iloc[-1] - 1.0) if len(eq) else float("nan")
    mdd = _max_drawdown(eq)

    return {
        "n": int(len(r)),
        "mean_weekly": mean_w,
        "std_weekly": std_w,
        "sharpe_annualized": sharpe,
        "hit_rate": hit,
        "cum_return": cum,
        "max_drawdown": mdd,
    }


def _markdown_table(df: pd.DataFrame, float_fmt: str = "{:.6f}") -> str:
    """Minimal markdown table generator with no optional dependencies."""
    if df is None or df.empty:
        return "_(no rows)_"

    d = df.copy()

    def fmt(x):
        if isinstance(x, (float, np.floating)):
            if np.isfinite(x):
                return float_fmt.format(float(x))
            return ""
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if pd.isna(x):
            return ""
        return str(x)

    cols = list(d.columns)
    rows = [[fmt(v) for v in d.iloc[i].tolist()] for i in range(len(d))]

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([header, sep, body])


def _compute_regime_buckets(weeks: List[str], scores_root: Path) -> pd.DataFrame:
    """
    Compute regime-strength metadata for each signal_week_end.
    - Loads per-week scores_weekly.parquet from scores_root/week_ending=YYYY-MM-DD/
    - Uses base col: UPS_adj if present else score
    - rs_top10_mean: mean of top decile of base col
    - regime_pct: percentile rank across provided weeks
    - regime_bucket: LOW/MID/HIGH terciles
    This is evaluation-only metadata (no trading changes).
    """
    rows = []
    for week_end in weeks:
        f = scores_root / f"week_ending={week_end}" / "scores_weekly.parquet"
        if not f.exists():
            rows.append(
                {
                    "signal_week_end": str(week_end),
                    "regime_missing": True,
                    "regime_base_col": "",
                    "rs_top10_mean": float("nan"),
                }
            )
            continue

        df = pd.read_parquet(f)
        base = "UPS_adj" if "UPS_adj" in df.columns else ("score" if "score" in df.columns else "")
        if not base:
            rows.append(
                {
                    "signal_week_end": str(week_end),
                    "regime_missing": True,
                    "regime_base_col": "",
                    "rs_top10_mean": float("nan"),
                }
            )
            continue

        x = pd.to_numeric(df[base], errors="coerce").dropna().to_numpy()
        if x.size == 0:
            rows.append(
                {
                    "signal_week_end": str(week_end),
                    "regime_missing": True,
                    "regime_base_col": base,
                    "rs_top10_mean": float("nan"),
                }
            )
            continue

        x_sorted = np.sort(x)
        top_n = max(1, int(0.10 * x_sorted.size))
        rs_top10_mean = float(np.mean(x_sorted[-top_n:]))

        rows.append(
            {
                "signal_week_end": str(week_end),
                "regime_missing": False,
                "regime_base_col": base,
                "rs_top10_mean": rs_top10_mean,
            }
        )

    reg = pd.DataFrame(rows)
    ok = reg[~reg["regime_missing"]].copy()
    if not ok.empty:
        ok["regime_pct"] = ok["rs_top10_mean"].rank(pct=True)
        ok["regime_bucket"] = pd.cut(
            ok["regime_pct"],
            bins=[0.0, 1 / 3, 2 / 3, 1.0],
            labels=["LOW", "MID", "HIGH"],
            include_lowest=True,
        )
    else:
        ok["regime_pct"] = []
        ok["regime_bucket"] = []

    reg = reg.merge(
        ok[["signal_week_end", "regime_pct", "regime_bucket"]],
        on="signal_week_end",
        how="left",
    )
    return reg


@dataclass(frozen=True)
class Paths:
    bt_weekly: Path
    candles_daily: Path
    out_dir: Path


def default_paths() -> Paths:
    return Paths(
        bt_weekly=Path("data/derived/backtest/bt_weekly.parquet"),
        candles_daily=Path("data/derived/market_daily/candles_daily.parquet"),
        out_dir=Path("data/derived/analysis"),
    )


def _as_ts(x) -> pd.Timestamp:
    return pd.to_datetime(x).normalize()


def _expected_week_end_from_candles(week_end: pd.Timestamp, candle_dates: pd.Series) -> pd.Timestamp:
    """
    Expected week_end = last trading day in the Mon–Fri window containing `week_end`.
    Candle dates are the authoritative trading calendar.
    """
    we = _as_ts(week_end)
    mon = we - pd.Timedelta(days=int(we.dayofweek))
    fri = mon + pd.Timedelta(days=4)

    mask = (candle_dates >= mon) & (candle_dates <= fri)
    wdays = candle_dates[mask]
    if wdays.empty:
        raise ValueError(
            f"No candle dates found for week window {mon.date()}..{fri.date()} (week_end={we.date()})"
        )
    return wdays.max().normalize()


def _apply_calendar_policy(
    bt: pd.DataFrame,
    candle_dates: pd.Series,
    mode: str = "warn",
    normalize: bool = True,
) -> pd.DataFrame:
    """
    mode:
      - ignore: do nothing
      - warn: print mismatches; optionally normalize signal_week_end
      - strict: raise on mismatches
    normalize:
      - if True, rewrite signal_week_end to expected week_end (string form)
    """
    if mode not in {"ignore", "warn", "strict"}:
        raise ValueError(f"Invalid calendar mode: {mode}")

    if mode == "ignore":
        return bt

    b = bt.copy()
    if "signal_week_end" not in b.columns:
        raise ValueError("bt missing required column: signal_week_end")

    sig_ts = pd.to_datetime(b["signal_week_end"]).dt.normalize()
    expected = sig_ts.apply(lambda x: _expected_week_end_from_candles(x, candle_dates))
    b["_expected_week_end"] = expected.dt.strftime("%Y-%m-%d")
    b["_signal_week_end_norm"] = sig_ts.dt.strftime("%Y-%m-%d")

    mism = b[b["_signal_week_end_norm"] != b["_expected_week_end"]].copy()
    if mism.empty:
        return b

    cols = ["signal_week_end", "_expected_week_end", "hold_entry_target", "hold_exit_target", "missing_returns"]
    cols = [c for c in cols if c in mism.columns]
    show = mism[cols].copy()

    msg = (
        "\n[CALENDAR MISMATCH] signal_week_end does not match last trading day of that Mon–Fri window\n"
        + _markdown_table(show, float_fmt="{:.6f}")
        + "\n"
    )

    if mode == "strict":
        raise SystemExit(msg + "Refusing to continue in --calendar_mode strict.\n")

    print(msg)
    if normalize:
        b["signal_week_end"] = b["_expected_week_end"]

    return b


def compute_week_completeness(bt: pd.DataFrame, candles: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    A week is "complete" if:
      - hold_exit_target <= max candle date
      - missing_returns == 0
      - spy_return is finite (or at least non-null)
    """
    if "date" not in candles.columns:
        raise ValueError("candles must include 'date' column")

    candle_max = pd.to_datetime(candles["date"]).max().normalize()
    if not np.isfinite(pd.Timestamp(candle_max).value):
        raise RuntimeError("Could not determine candles max date")

    b = bt.copy()

    for c in ["signal_week_end", "hold_entry_target", "hold_exit_target"]:
        if c not in b.columns:
            raise ValueError(f"bt missing required column: {c}")

    b["signal_week_end"] = b["signal_week_end"].astype(str)
    b["hold_exit_target"] = pd.to_datetime(b["hold_exit_target"]).dt.normalize()
    b["missing_returns"] = pd.to_numeric(b.get("missing_returns", 0), errors="coerce").fillna(0).astype(int)

    spy = pd.to_numeric(b.get("spy_return", np.nan), errors="coerce")
    b["_spy_ok"] = np.isfinite(spy)

    b["_exit_in_candles"] = b["hold_exit_target"] <= candle_max
    b["_missing_ok"] = b["missing_returns"] == 0

    b["_complete"] = b["_exit_in_candles"] & b["_missing_ok"] & b["_spy_ok"]
    return b, candle_max


def main() -> None:
    ap = argparse.ArgumentParser("Evaluate complete backtest weeks (one-command)")
    ap.add_argument("--bt", default=None, help="Path to bt_weekly.parquet (Friday-only source recommended)")
    ap.add_argument("--candles", default=None, help="Path to candles_daily.parquet")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: data/derived/analysis)")
    ap.add_argument("--ann_factor", type=float, default=52.0, help="Annualization factor (weekly=52)")
    ap.add_argument(
        "--scores_root",
        default=None,
        help="Root dir for per-week scores (default: data/derived/scores_weekly/regime=news-novelty-v1/schema=news-novelty-v1b)",
    )
    ap.add_argument(
        "--calendar_mode",
        default="warn",
        choices=["ignore", "warn", "strict"],
        help="Calendar validation for signal_week_end using candles (default: warn)",
    )
    ap.add_argument(
        "--normalize_week_end",
        action="store_true",
        help="If set, rewrite signal_week_end to expected last-trading-day week_end when mismatches are found",
    )
    ap.add_argument(
        "--no_normalize_week_end",
        action="store_true",
        help="Disable normalization even if calendar_mode is warn",
    )
    args = ap.parse_args()

    normalize = True
    if args.no_normalize_week_end:
        normalize = False
    elif args.normalize_week_end:
        normalize = True

    p = default_paths()
    bt_path = Path(args.bt) if args.bt else p.bt_weekly
    candles_path = Path(args.candles) if args.candles else p.candles_daily
    out_dir = Path(args.out_dir) if args.out_dir else p.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bt_path.exists():
        raise SystemExit(f"Missing backtest file: {bt_path}")
    if not candles_path.exists():
        raise SystemExit(f"Missing candles file: {candles_path}")

    bt = pd.read_parquet(bt_path).copy()
    candles = pd.read_parquet(candles_path, columns=["date"]).copy()

    candle_dates = pd.to_datetime(candles["date"]).dt.normalize()
    bt = _apply_calendar_policy(bt, candle_dates, mode=args.calendar_mode, normalize=normalize)

    bt = bt.sort_values("signal_week_end").reset_index(drop=True)

    bt2, candle_max = compute_week_completeness(bt, candles)

    complete = bt2[bt2["_complete"]].copy()
    incomplete = bt2[~bt2["_complete"]].copy()

    complete_weeks = complete["signal_week_end"].astype(str).tolist()
    incomplete_weeks = incomplete["signal_week_end"].astype(str).tolist()

    complete_txt = out_dir / "weeks_eval_complete.txt"
    incomplete_txt = out_dir / "weeks_eval_incomplete.txt"
    complete_txt.write_text("\n".join(complete_weeks) + ("\n" if complete_weeks else ""))
    incomplete_txt.write_text("\n".join(incomplete_weeks) + ("\n" if incomplete_weeks else ""))

    print(f"candles max date: {candle_max.date().isoformat()}")
    print(f"complete weeks: {len(complete_weeks)}")
    print(f"incomplete weeks: {len(incomplete_weeks)}")
    print(f"Wrote: {complete_txt}")
    print(f"Wrote: {incomplete_txt}")

    if len(incomplete):
        cols = ["signal_week_end", "hold_entry_target", "hold_exit_target", "missing_returns", "spy_return"]
        cols = [c for c in cols if c in incomplete.columns]
        print("\nIncomplete detail:")
        print(incomplete[cols].to_string(index=False))

    if complete.empty:
        raise SystemExit("No complete weeks found. (Need more candles to cover hold windows.)")

    c = complete.copy()
    for col in ["net_return", "spy_return", "active_net_return", "turnover", "tcost"]:
        if col in c.columns:
            c[col] = pd.to_numeric(c[col], errors="coerce")

    S = _summary_stats(c["net_return"], ann_factor=args.ann_factor)
    B = _summary_stats(c["spy_return"], ann_factor=args.ann_factor)
    A = _summary_stats(c["active_net_return"], ann_factor=args.ann_factor)

    summary = {
        "inputs": {
            "bt_path": str(bt_path),
            "candles_path": str(candles_path),
            "candles_max_date": candle_max.date().isoformat(),
            "n_complete_weeks": int(len(c)),
            "n_incomplete_weeks": int(len(incomplete)),
            "calendar_mode": args.calendar_mode,
            "normalize_week_end": bool(normalize),
        },
        "strategy_net": S,
        "spy": B,
        "active_net": A,
    }

    summary_path = out_dir / "perf_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote: {summary_path}")

    keep_cols = [
        "signal_week_end",
        "hold_entry_target",
        "hold_exit_target",
        "n_positions",
        "missing_returns",
        "score_col",
        "gross_return",
        "turnover",
        "tcost",
        "net_return",
        "spy_return",
        "active_net_return",
    ]
    keep_cols = [k for k in keep_cols if k in c.columns]
    perf_csv = out_dir / "perf_complete_weeks.csv"
    c[keep_cols].sort_values("signal_week_end").to_csv(perf_csv, index=False)
    print(f"Wrote: {perf_csv}")

    scores_root = Path(args.scores_root) if args.scores_root else Path(
        "data/derived/scores_weekly/regime=news-novelty-v1/schema=news-novelty-v1b"
    )
    regime = _compute_regime_buckets(complete_weeks, scores_root)
    regime_join = c[["signal_week_end", "active_net_return", "score_col"]].merge(regime, on="signal_week_end", how="left")

    regime_csv = out_dir / "regime_buckets_complete_weeks.csv"
    regime_join.sort_values("signal_week_end").to_csv(regime_csv, index=False)
    print(f"Wrote: {regime_csv}")

    strength_cols = ["signal_week_end", "score_col", "regime_base_col", "rs_top10_mean", "active_net_return"]
    strength_df = (
        regime_join[strength_cols].copy()
        if all(col in regime_join.columns for col in strength_cols)
        else pd.DataFrame()
    )
    strength_path = out_dir / "regime_strength_vs_active.csv"
    if not strength_df.empty:
        strength_df = strength_df.sort_values("signal_week_end")
        strength_df.to_csv(strength_path, index=False)
        print(f"Wrote: {strength_path}")
    else:
        print(f"NOTE: strength_df empty; not writing {strength_path}")

    corr_summary: Dict[str, float] = {}
    if not strength_df.empty:
        x = pd.to_numeric(strength_df["rs_top10_mean"], errors="coerce")
        y = pd.to_numeric(strength_df["active_net_return"], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() >= 2:
            corr_summary["n"] = int(ok.sum())
            corr_summary["pearson"] = float(x[ok].corr(y[ok]))
            xr = x[ok].rank(method="average")
            yr = y[ok].rank(method="average")
            corr_summary["spearman_like"] = float(xr.corr(yr))
        else:
            corr_summary["n"] = int(ok.sum())
            corr_summary["pearson"] = float("nan")
            corr_summary["spearman_like"] = float("nan")

    corr_path = out_dir / "regime_strength_summary.json"
    corr_path.write_text(json.dumps(corr_summary, indent=2) + "\n")
    print(f"Wrote: {corr_path}")

    regime_ok = regime_join[regime_join["regime_missing"] == False].copy()  # noqa: E712
    if not regime_ok.empty and "regime_bucket" in regime_ok.columns:
        bucket_summary = (
            regime_ok.groupby("regime_bucket", observed=True)
            .agg(
                weeks=("active_net_return", "count"),
                mean_return=("active_net_return", "mean"),
                median_return=("active_net_return", "median"),
                hit_rate=("active_net_return", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
                max_return=("active_net_return", "max"),
                min_return=("active_net_return", "min"),
            )
            .reset_index()
        )
    else:
        bucket_summary = pd.DataFrame()

    md_path = out_dir / "evaluation_complete_weeks.md"
    lines: List[str] = []
    lines.append("# Evaluation: complete weeks\n")
    lines.append(f"- candles max date: **{candle_max.date().isoformat()}**")
    lines.append(f"- complete weeks: **{len(complete_weeks)}**")
    lines.append(f"- incomplete weeks: **{len(incomplete_weeks)}**")
    lines.append(f"- calendar_mode: **{args.calendar_mode}**")
    lines.append(f"- normalize_week_end: **{normalize}**\n")

    lines.append("## Summary stats\n")
    lines.append("### Strategy (net)\n")
    lines.append("```json")
    lines.append(json.dumps(S, indent=2))
    lines.append("```\n")

    lines.append("### SPY\n")
    lines.append("```json")
    lines.append(json.dumps(B, indent=2))
    lines.append("```\n")

    lines.append("### Active (net - SPY)\n")
    lines.append("```json")
    lines.append(json.dumps(A, indent=2))
    lines.append("```\n")

    lines.append("## Per-week (complete only)\n")
    table_df = c[keep_cols].sort_values("signal_week_end")
    lines.append(_markdown_table(table_df, float_fmt="{:.6f}"))
    lines.append("")

    lines.append("## Regime buckets (rs_top10_mean)\n")
    lines.append("- Base column per week: `UPS_adj` if present else `score`")
    lines.append("- `rs_top10_mean`: mean of the top decile of the base column across symbols")
    lines.append("- `regime_pct`: percentile rank across **complete weeks only**")
    lines.append("- `regime_bucket`: terciles of `regime_pct` (LOW/MID/HIGH)\n")
    lines.append(f"- scores_root: `{scores_root}`\n")

    if not bucket_summary.empty:
        lines.append("### Bucket summary (active_net_return)\n")
        lines.append(_markdown_table(bucket_summary, float_fmt="{:.6f}"))
        lines.append("")
    else:
        lines.append("_(No regime bucket summary available.)\n")

    if "regime_bucket" in regime_ok.columns and not regime_ok.empty:
        lines.append("### Weeks by bucket\n")
        for b in ["LOW", "MID", "HIGH"]:
            w = regime_ok.loc[regime_ok["regime_bucket"] == b, "signal_week_end"].astype(str).tolist()
            lines.append(f"- **{b}**: " + (", ".join(w) if w else "_(none)_"))
        lines.append("")

    if len(incomplete):
        lines.append("## Incomplete weeks detail\n")
        cols = ["signal_week_end", "hold_entry_target", "hold_exit_target", "missing_returns", "spy_return"]
        cols = [cc for cc in cols if cc in incomplete.columns]
        lines.append(_markdown_table(incomplete[cols].sort_values("signal_week_end"), float_fmt="{:.6f}"))
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {md_path}")

    print("\nSTRATEGY (net):", S)
    print("SPY:", B)
    print("ACTIVE (net - SPY):", A)


if __name__ == "__main__":
    main()
