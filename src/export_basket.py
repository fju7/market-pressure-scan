from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import config


def run(week_end: str, top_n: int, skip_low_info: bool, equal_weight: bool = True) -> Path:
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

    # Load scores output
    scores_path = Path(f"data/derived/scores_weekly/week_ending={week_end}/scores_weekly.parquet")
    if not scores_path.exists():
        raise FileNotFoundError(
            f"Missing scores parquet for week {week_end}. "
            f"Expected: {scores_path}"
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

    out = pd.DataFrame({
        "week_ending_date": week_end,
        "action": "TRADE",
        "symbol": df["symbol"],
        "sector": df["sector"] if "sector" in df.columns else "",
        "UPS_adj": df["UPS_adj"],
        "conviction": df.get("conviction", ""),
        "weight": df["weight"],
    })

    out.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} ({len(out)} rows)")
    return out_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--week_end", required=True)
    ap.add_argument("--top_n", type=int, default=None, help="Basket size (default: from CONFIG.yaml)")
    ap.add_argument("--skip_low_info", action="store_true")
    args = ap.parse_args()

    # Use config default if not specified
    top_n = args.top_n if args.top_n is not None else config.get_basket_size()
    
    run(args.week_end, top_n=top_n, skip_low_info=args.skip_low_info)
