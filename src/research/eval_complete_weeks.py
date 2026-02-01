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
            rows.append({
                "signal_week_end": str(week_end),
                "regime_missing": True,
                "regime_base_col": "",
                "rs_top10_mean": float("nan"),
            })
            continue

        df = pd.read_parquet(f)
        base = "UPS_adj" if "UPS_adj" in df.columns else ("score" if "score" in df.columns else "")
        if not base:
            rows.append({
                "signal_week_end": str(week_end),
                "regime_missing": True,
                "regime_base_col": "",
                "rs_top10_mean": float("nan"),
            })
            continue

        x = pd.to_numeric(df[base], errors="coerce").dropna().to_numpy()
        if x.size == 0:
            rows.append({
                "signal_week_end": str(week_end),
                "regime_missing": True,
                "regime_base_col": base,
                "rs_top10_mean": float("nan"),
            })
            continue

        x_sorted = np.sort(x)
        top_n = max(1, int(0.10 * x_sorted.size))
        rs_top10_mean = float(np.mean(x_sorted[-top_n:]))

        rows.append({
            "signal_week_end": str(week_end),
            "regime_missing": False,
            "regime_base_col": base,
            "rs_top10_mean": rs_top10_mean,
        })

    reg = pd.DataFrame(rows)
    ok = reg[~reg["regime_missing"]].copy()
    if not ok.empty:
        ok["regime_pct"] = ok["rs_top10_mean"].rank(pct=True)
        ok["regime_bucket"] = pd.cut(
            ok["regime_pct"],
            bins=[0.0, 1/3, 2/3, 1.0],
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
    ap.add_argument(
        "--scores_root",
        default=None,
        help="Root dir for per-week scores (default: data/derived/scores_weekly/regime=news-novelty-v1/schema=news-novelty-v1b)",
    )
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

    # Regime buckets (evaluation-only)
    scores_root = Path(args.scores_root) if args.scores_root else Path(
        "data/derived/scores_weekly/regime=news-novelty-v1/schema=news-novelty-v1b"
    )
    regime = _compute_regime_buckets(complete_weeks, scores_root)
    regime_join = c[["signal_week_end", "active_net_return", "score_col"]].merge(
        regime,
        on="signal_week_end",
        how="left",
    )
    regime_csv = out_dir / "regime_buckets_complete_weeks.csv"
    regime_join.sort_values("signal_week_end").to_csv(regime_csv, index=False)
    print(f"Wrote: {regime_csv}")

    # Bucket summary (active_net_return)
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

    # Regime buckets section (evaluation-only)
    lines.append("## Regime buckets (rs_top10_mean)\n")
    lines.append("- Base column per week: `UPS_adj` if present else `score`")
    lines.append("- `rs_top10_mean`: mean of the top decile of the base column across symbols")
    lines.append("- `regime_pct`: percentile rank across **complete weeks only**")
    lines.append("- `regime_bucket`: terciles of `regime_pct` (LOW/MID/HIGH)\n")
    lines.append(f"- scores_root: `{scores_root}`\n")
    if bucket_summary is not None and not bucket_summary.empty:
        lines.append("### Bucket summary (active_net_return)\n")
        lines.append(_markdown_table(bucket_summary, float_fmt="{:.6f}"))
        lines.append("")
    else:
        lines.append("_(No regime bucket summary available.)\n")
    # List weeks by bucket
    try:
        if "regime_bucket" in regime_ok.columns and not regime_ok.empty:
            lines.append("### Weeks by bucket\n")
            for b in ["LOW", "MID", "HIGH"]:
                w = regime_ok.loc[regime_ok["regime_bucket"] == b, "signal_week_end"].astype(str).tolist()
                lines.append(f"- **{b}**: " + (", ".join(w) if w else "_(none)_"))
            lines.append("")
    except Exception:
        pass

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
