"""
Log weekly decision (TRADE or SKIP) to weeks_log.csv

This script reads the basket.csv and report_meta.json for a given week
and logs the decision to weeks_log.csv for tracking purposes.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


CANON_COLS = [
    "week_ending_date",
    "action",
    "basket_size",
    "overlap_pct",
    "turnover_pct",
    "num_clusters",
    "avg_novelty_z",
    "avg_event_intensity_z",
    "recap_pct",
    "is_low_info",
    "num_positions",
    "logged_at",
    "skip_reason",
]


def clean_text(x: str) -> str:
    """Sanitize text fields to prevent CSV corruption from newlines/commas"""
    if not isinstance(x, str):
        return ""
    return " ".join(x.replace("\r", " ").replace("\n", " ").split())


def load_and_normalize_weeks_log(path: Path) -> pd.DataFrame:
    """
    Load weeks_log.csv with schema normalization and error recovery.
    
    This function:
    - Creates an empty DataFrame with canonical columns if file doesn't exist
    - Handles parse errors gracefully by falling back to python engine
    - Normalizes schema by adding any missing columns
    - Rewrites the file in canonical format with proper quoting to repair corruption
    
    Parameters
    ----------
    path : Path
        Path to weeks_log.csv
    
    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with all canonical columns
    """
    if not path.exists():
        return pd.DataFrame(columns=CANON_COLS)

    # Count original lines for repair diagnostics
    with open(path, 'r') as f:
        original_line_count = sum(1 for _ in f)
    
    had_parse_error = False
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError as e:
        print(f"⚠️  weeks_log.csv parse error: {e}")
        print("    Attempting recovery with python engine...")
        had_parse_error = True
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        
        # Report dropped lines (if any)
        recovered_rows = len(df)
        expected_rows = original_line_count - 1  # -1 for header
        if recovered_rows < expected_rows:
            dropped = expected_rows - recovered_rows
            print(f"⚠️  weeks_log.csv repair: recovered_rows={recovered_rows} dropped_lines≈{dropped}")

    # Normalize schema (upgrade older logs that might be missing columns)
    schema_upgraded = False
    for c in CANON_COLS:
        if c not in df.columns:
            df[c] = ""
            schema_upgraded = True

    # Reorder to canonical column order
    df = df[CANON_COLS]

    # Rewrite in canonical format w/ safe quoting (repairs old headers + corruption)
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
    
    # Detailed diagnostics
    status_flags = []
    if had_parse_error:
        status_flags.append("repaired")
    if schema_upgraded:
        status_flags.append("upgraded")
    
    status_str = f" [{', '.join(status_flags)}]" if status_flags else ""
    print(f"✅ weeks_log.csv normalized: cols={len(df.columns)} rows={len(df)} quoting=QUOTE_ALL{status_str}")
    
    return df


def log_week_decision(week_end: str) -> str:
    """
    Log the week's decision to weeks_log.csv
    
    Parameters
    ----------
    week_end : str
        Week ending date (YYYY-MM-DD)
    
    Returns
    -------
    str
        Action taken: TRADE or SKIP
    """
    # Read basket
    basket_path = Path(f"data/derived/baskets/week_ending={week_end}/basket.csv")
    if not basket_path.exists():
        raise FileNotFoundError(f"Basket not found: {basket_path}")
    
    basket = pd.read_csv(basket_path)
    action = basket["action"].iloc[0]
    
    # Calculate turnover metrics by comparing with prior week
    basket_size = len(basket) if action == "TRADE" else 0
    overlap_pct = 0.0
    turnover_pct = 0.0
    
    if action == "TRADE" and basket_size > 0:
        # Find most recent prior TRADE week
        log_path = Path("data/live/weeks_log.csv")
        if log_path.exists():
            weeks_log = load_and_normalize_weeks_log(log_path)
            prior_trades = weeks_log[
                (weeks_log["action"] == "TRADE") & 
                (weeks_log["week_ending_date"] < week_end)
            ]
            
            if not prior_trades.empty:
                prior_week_end = prior_trades.iloc[-1]["week_ending_date"]
                prior_basket_path = Path(f"data/derived/baskets/week_ending={prior_week_end}/basket.csv")
                
                if prior_basket_path.exists():
                    prior_basket = pd.read_csv(prior_basket_path)
                    
                    # Get ticker lists (handle potential column name variations)
                    current_tickers = set(basket["ticker"].values if "ticker" in basket.columns else basket.iloc[:, 0].values)
                    prior_tickers = set(prior_basket["ticker"].values if "ticker" in prior_basket.columns else prior_basket.iloc[:, 0].values)
                    
                    # Calculate overlap and turnover
                    overlap_count = len(current_tickers & prior_tickers)
                    overlap_pct = (overlap_count / basket_size * 100) if basket_size > 0 else 0.0
                    turnover_pct = ((basket_size - overlap_count) / basket_size * 100) if basket_size > 0 else 0.0
    
    # Read report metadata
    meta_path = Path(f"data/derived/reports/week_ending={week_end}/report_meta.json")
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    
    # Prepare log entry
    # Use cluster_count from report_meta.json (single source of truth)
    skip_reason = ""
    if action == "SKIP":
        skip_reason = basket["reason"].iloc[0] if "reason" in basket.columns else ""
    
    log_entry = {
        "week_ending_date": week_end,
        "action": action,
        "basket_size": basket_size,
        "overlap_pct": round(overlap_pct, 1),
        "turnover_pct": round(turnover_pct, 1),
        "num_clusters": meta.get("cluster_count", 0),  # Changed from num_clusters
        "avg_novelty_z": meta.get("avg_novelty_z", 0.0),
        "avg_event_intensity_z": meta.get("avg_event_intensity_z", 0.0),
        "recap_pct": meta.get("recap_pct", 0.0),
        "is_low_info": meta.get("is_low_information_week", False),
        "num_positions": len(basket) if action == "TRADE" else 0,
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "skip_reason": clean_text(skip_reason),  # Clean to prevent CSV corruption
    }
    
    # Append to weeks_log.csv
    log_path = Path("data/live/weeks_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load and normalize existing log (handles schema drift and corruption)
    existing = load_and_normalize_weeks_log(log_path)
    
    if week_end in existing["week_ending_date"].astype(str).values:
        print(f"⚠️  Week {week_end} already logged. Skipping.")
        return action
    
    # Create row with canonical column order
    row = pd.DataFrame([{c: log_entry.get(c, "") for c in CANON_COLS}])
    
    # Append with safe quoting to prevent future corruption
    row.to_csv(log_path, mode="a", header=False, index=False, quoting=csv.QUOTE_ALL)
    print(f"✅ Appended week {week_end} to {log_path}")
    
    # Display summary
    print(f"\n📊 Week {week_end} summary:")
    print(f"   Action: {action}")
    print(f"   Basket size: {log_entry['basket_size']}")
    print(f"   Clusters: {log_entry['num_clusters']}")
    print(f"   Recap %: {log_entry['recap_pct']:.0f}%")
    print(f"   Low info: {log_entry['is_low_info']}")
    
    if action == "SKIP":
        print(f"   ⏭️  Reason: {log_entry['skip_reason'][:80]}")
    else:
        print(f"   📈 Positions: {log_entry['num_positions']}")
        if overlap_pct > 0:
            print(f"   🔄 Overlap: {log_entry['overlap_pct']:.1f}% | Turnover: {log_entry['turnover_pct']:.1f}%")
    
    return action


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Log weekly decision (TRADE or SKIP) to weeks_log.csv"
    )
    ap.add_argument("--week_end", required=True, help="Week ending date YYYY-MM-DD")
    
    args = ap.parse_args()
    
    action = log_week_decision(args.week_end)
    
    # Print next steps
    print("\n📋 Next steps:")
    if action == "SKIP":
        print("   ✓ Week logged as SKIP")
        print("   → No further action needed this week")
    else:
        print("   ✓ Week logged as TRADE")
        print(f"   → Monday: Execute trades and log with:")
        print(f"     python -m src.log_trades --week_end {args.week_end} --execution_date YYYY-MM-DD --account_value XXXXX")
        print(f"   → Friday: Log P&L with:")
        print(f"     python -m src.log_weekly_pnl --week_end {args.week_end} --exit_date YYYY-MM-DD")
