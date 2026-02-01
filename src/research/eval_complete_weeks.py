from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Small utilities
# ----------------------------

def _to_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).normalize()


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


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
    sharpe = float((mean_w / std_w) * np.sqrt(ann_factor)) if (std_w and np.isfinite(std_w) and std_w > 0) else float("nan")
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
    """
    Minimal markdown table generator with no optional dependencies.
    """
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


@dataclass(frozen=True)
class Paths:
    bt_weekly_all: Path
    candles_daily: Path
    out_dir: Path


def default_paths() -> Paths:
    return Paths(
        bt_weekly_all=Path("data/derived/analysis/bt_weekly_all.parquet"),
        candles_daily=Path("data/derived/market_daily/candles_daily.parquet"),
        out_dir=Path("data/derived/analysis"),
    )


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

    # normalize required cols
    for c in ["signal_week_end", "hold_entry_target", "hold_exit_target"]:
        if c not in b.columns:
            raise ValueError(f"bt missing required column: {c}")

    b["signal_week_end"] = b["signal_week_end"].astype(str)
    b["hold_exit_target"] = pd.to_datetime(b["hold_exit_target"]).dt.normalize()
    b["missing_returns"] = pd.to_numeric(b.get("missing_returns", 0), errors="coerce").fillna(0).astype(int)

    # allow spy_return to be missing => incomplete
    spy = pd.to_numeric(b.get("spy_return", np.nan), errors="coerce")
    b["_spy_ok"] = np.isfinite(spy)

    b["_exit_in_candles"] = b["hold_exit_target"] <= candle_max
    b["_missing_ok"] = b["missing_returns"] == 0

    b["_complete"] = b["_exit_in_candles"] & b["_missing_ok"] & b["_spy_ok"]

    return b, candle_max


def main() -> None:
    ap = argparse.ArgumentParser("Evaluate complete backtest weeks (one-command)")
    ap.add_argument("--bt", default=None, help="Path to bt_weekly_all.parquet")
    ap.add_argument("--candles", default=None, help="Path to candles_daily.parquet")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: data/derived/analysis)")
    ap.add_argument("--ann_factor", type=float, default=52.0, help="Annualization factor (weekly=52)")
    args = ap.parse_args()

    p = default_paths()
    bt_path = Path(args.bt) if args.bt else p.bt_weekly_all
    candles_path = Path(args.candles) if args.candles else p.candles_daily
    out_dir = Path(args.out_dir) if args.out_dir else p.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bt_path.exists():
        raise SystemExit(f"Missing backtest file: {bt_path}")
    if not candles_path.exists():
        raise SystemExit(f"Missing candles file: {candles_path}")

    bt = pd.read_parquet(bt_path).copy()
    candles = pd.read_parquet(candles_path, columns=["date"]).copy()

    bt = bt.sort_values("signal_week_end").reset_index(drop=True)

    bt2, candle_max = compute_week_completeness(bt, candles)

    complete = bt2[bt2["_complete"]].copy()
    incomplete = bt2[~bt2["_complete"]].copy()

    # Write week lists
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

    # Metrics on complete weeks only
    c = complete.copy()

    # ensure numeric
    for col in ["net_return", "spy_return", "active_net_return", "turnover", "tcost"]:
        if col in c.columns:
            c[col] = pd.to_numeric(c[col], errors="coerce")

    # Summary stats
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
        },
        "strategy_net": S,
        "spy": B,
        "active_net": A,
    }

    summary_path = out_dir / "perf_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote: {summary_path}")

    # Per-week CSV
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

    # Markdown report
    md_path = out_dir / "evaluation_complete_weeks.md"
    lines: List[str] = []
    lines.append("# Evaluation: complete weeks\n")
    lines.append(f"- candles max date: **{candle_max.date().isoformat()}**")
    lines.append(f"- complete weeks: **{len(complete_weeks)}**")
    lines.append(f"- incomplete weeks: **{len(incomplete_weeks)}**\n")

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

    if len(incomplete):
        lines.append("## Incomplete weeks detail\n")
        cols = ["signal_week_end", "hold_entry_target", "hold_exit_target", "missing_returns", "spy_return"]
        cols = [cc for cc in cols if cc in incomplete.columns]
        lines.append(_markdown_table(incomplete[cols].sort_values("signal_week_end"), float_fmt="{:.6f}"))
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {md_path}")

    # Console summary
    print("\nSTRATEGY (net):", S)
    print("SPY:", B)
    print("ACTIVE (net - SPY):", A)


if __name__ == "__main__":
    main()
