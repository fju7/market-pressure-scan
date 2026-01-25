from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def compute_drivers(row: pd.Series) -> str:
    """Compute driver summary for position (for attribution analysis)"""
    pieces = []
    if np.isfinite(row.get("EVS", np.nan)) and row["EVS"] > 0.6:
        pieces.append("Event intensity")
    if np.isfinite(row.get("SS", np.nan)) and row["SS"] > 0.6:
        pieces.append("Sentiment inflection")
    if np.isfinite(row.get("NS", np.nan)) and row["NS"] > 0.6:
        pieces.append("Novelty")
    if np.isfinite(row.get("MCS_up", np.nan)) and row["MCS_up"] > 0.6:
        pieces.append("Market follow-through")
    if not pieces:
        pieces.append("Mixed / low signal")
    return ", ".join(pieces[:2])


def compute_signal_state(row: pd.Series) -> str:
    """Compute signal state for position"""
    ifs = float(row.get("IFS", np.nan))
    ar5 = float(row.get("AR5", np.nan))
    
    if not np.isfinite(ifs) or not np.isfinite(ar5):
        return "Early"
    if ifs > 0.35 and ar5 < 0.0:
        return "Divergent"
    if ifs < 0.0 and ar5 > 0.35:
        return "Overheated"
    if ifs > 0.2 and ar5 > 0.2:
        return "Confirmed"
    return "Early"


def compute_conviction(ups_adj: float) -> str:
    """Compute conviction band for position"""
    if ups_adj is None or not np.isfinite(ups_adj):
        return "—"
    a = abs(float(ups_adj))
    if a < 0.25:
        return "Weak"
    if a < 0.75:
        return "Moderate"
    return "Strong"


def run(week_end: str, top_n: int, skip_low_info: bool, regime: str = "news-novelty-v1", equal_weight: bool = True) -> Path:
    meta_path = Path(f"data/derived/reports/week_ending={week_end}/report_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing report meta: {meta_path}. Run report_weekly first.")

    meta = json.loads(meta_path.read_text())
    is_low = bool(meta.get("is_low_information_week", False))

    out_dir = Path(f"data/derived/baskets/week_ending={week_end}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "basket.csv"

    if skip_low_info and is_low:
        # Write a "do nothing" basket
        df = pd.DataFrame([{
            "week_ending_date": week_end,
            "action": "SKIP",
            "reason": "; ".join(meta.get("low_info_reasons", []))[:500],
        }])
        df.to_csv(out_csv, index=False, quoting=1)  # QUOTE_MINIMAL
        print(f"Wrote: {out_csv} (SKIP)")
        return out_csv

    # Load scores output (try regime-specific path first, fall back to legacy)
    scores_path = Path(f"data/derived/scores_weekly/regime={regime}/week_ending={week_end}/scores_weekly.parquet")
    if not scores_path.exists():
        scores_path_legacy = Path(f"data/derived/scores_weekly/week_ending={week_end}/scores_weekly.parquet")
        if scores_path_legacy.exists():
            scores_path = scores_path_legacy
        else:
            raise FileNotFoundError(
                f"Missing scores parquet for week {week_end}. "
                f"Expected: {scores_path} or {scores_path_legacy}"
            )

    df = pd.read_parquet(scores_path).copy()
    if "symbol" not in df.columns or "UPS_adj" not in df.columns:
        raise RuntimeError("Scores file must include at least: symbol, UPS_adj")

    # Exclude SPY/ETFs if they slip through
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = df[df["symbol"] != "SPY"].copy()
    if "sector" in df.columns:
        df = df[df["sector"].astype(str).str.upper().str.strip() != "ETF"].copy()

    # Top N by UPS_adj (descending)
    df = df.sort_values("UPS_adj", ascending=False).head(top_n).copy()

    if equal_weight:
        df["weight"] = 1.0 / len(df) if len(df) else 0.0
    else:
        # placeholder for later (risk parity / vol targeting)
        df["weight"] = 1.0 / len(df) if len(df) else 0.0

    # Add reason codes for attribution analysis
    df["drivers"] = df.apply(compute_drivers, axis=1)
    df["signal_state"] = df.apply(compute_signal_state, axis=1)
    df["conviction"] = df["UPS_adj"].apply(compute_conviction)

    out = pd.DataFrame({
        "week_ending_date": week_end,
        "action": "TRADE",
        "symbol": df["symbol"],
        "sector": df["sector"] if "sector" in df.columns else "",
        "UPS_adj": df["UPS_adj"],
        "conviction": df["conviction"],
        "drivers": df["drivers"],
        "signal_state": df["signal_state"],
        "weight": df["weight"],
    })

    out.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} ({len(out)} rows)")
    return out_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--week_end", required=True)
    ap.add_argument("--regime", default="news-novelty-v1", help="Regime ID (e.g., news-novelty-v1, news-novelty-v1b)")
    ap.add_argument("--top_n", type=int, default=None, help="Basket size (default: from CONFIG.yaml)")
    ap.add_argument("--skip_low_info", action="store_true")
    args = ap.parse_args()

    # Use config default if not specified
    top_n = args.top_n if args.top_n is not None else config.get_basket_size()
    
    run(args.week_end, top_n=top_n, skip_low_info=args.skip_low_info, regime=args.regime)
