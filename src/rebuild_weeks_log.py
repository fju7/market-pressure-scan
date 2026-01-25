"""
Rebuild weeks_log.csv from scratch by discovering all weeks in data/derived/reports

This script:
- Discovers all weeks from data/derived/reports/week_ending=*
- Computes each row using the same logic as log_week_decision
- Writes a fresh, sorted, deduplicated weeks_log.csv
- Is idempotent (can be run multiple times safely)

Usage:
    python -m src.rebuild_weeks_log
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.log_week_decision import CANON_COLS, clean_text


def compute_week_row(week_end: str, all_weeks: list[str]) -> dict:
    """
    Compute a single row for weeks_log.csv using the same logic as log_week_decision
    
    Parameters
    ----------
    week_end : str
        Week ending date (YYYY-MM-DD)
    all_weeks : list[str]
        All available weeks (sorted) for turnover calculation
    
    Returns
    -------
    dict
        Row data with canonical columns
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
        # Find most recent prior TRADE week from all_weeks
        prior_weeks = [w for w in all_weeks if w < week_end]
        
        for prior_week_end in reversed(prior_weeks):
            prior_basket_path = Path(f"data/derived/baskets/week_ending={prior_week_end}/basket.csv")
            
            if prior_basket_path.exists():
                prior_basket = pd.read_csv(prior_basket_path)
                
                # Only compare to prior TRADE weeks
                if prior_basket["action"].iloc[0] != "TRADE":
                    continue
                
                # Get ticker lists (handle potential column name variations)
                current_tickers = set(basket["ticker"].values if "ticker" in basket.columns else basket.iloc[:, 0].values)
                prior_tickers = set(prior_basket["ticker"].values if "ticker" in prior_basket.columns else prior_basket.iloc[:, 0].values)
                
                # Calculate overlap and turnover
                overlap_count = len(current_tickers & prior_tickers)
                overlap_pct = (overlap_count / basket_size * 100) if basket_size > 0 else 0.0
                turnover_pct = ((basket_size - overlap_count) / basket_size * 100) if basket_size > 0 else 0.0
                break  # Found most recent TRADE week
    
    # Read report metadata
    meta_path = Path(f"data/derived/reports/week_ending={week_end}/report_meta.json")
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    
    # Prepare log entry
    skip_reason = ""
    if action == "SKIP":
        skip_reason = basket["reason"].iloc[0] if "reason" in basket.columns else ""
    
    return {
        "week_ending_date": week_end,
        "action": action,
        "basket_size": basket_size,
        "overlap_pct": round(overlap_pct, 1),
        "turnover_pct": round(turnover_pct, 1),
        "num_clusters": meta.get("cluster_count", 0),
        "avg_novelty_z": meta.get("avg_novelty_z", 0.0),
        "avg_event_intensity_z": meta.get("avg_event_intensity_z", 0.0),
        "recap_pct": meta.get("recap_pct", 0.0),
        "is_low_info": meta.get("is_low_information_week", False),
        "num_positions": len(basket) if action == "TRADE" else 0,
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "skip_reason": clean_text(skip_reason),
    }


def rebuild_weeks_log():
    """
    Rebuild weeks_log.csv from scratch by discovering all weeks
    """
    print("🔍 Discovering weeks from data/derived/reports/...")
    
    # Discover all weeks
    reports_dir = Path("data/derived/reports")
    if not reports_dir.exists():
        raise FileNotFoundError(f"Reports directory not found: {reports_dir}")
    
    week_dirs = sorted([
        d.name.replace("week_ending=", "")
        for d in reports_dir.iterdir()
        if d.is_dir() and d.name.startswith("week_ending=")
    ])
    
    if not week_dirs:
        raise RuntimeError("No weeks found in data/derived/reports/")
    
    print(f"   Found {len(week_dirs)} weeks: {week_dirs[0]} to {week_dirs[-1]}")
    
    # Compute rows for all weeks
    print("\n📊 Computing rows...")
    rows = []
    for i, week_end in enumerate(week_dirs, 1):
        print(f"   [{i}/{len(week_dirs)}] {week_end}...", end=" ")
        
        try:
            row = compute_week_row(week_end, week_dirs)
            rows.append(row)
            print(f"✓ {row['action']}")
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    if not rows:
        raise RuntimeError("No rows computed")
    
    # Create DataFrame with canonical column order
    df = pd.DataFrame([{c: row.get(c, "") for c in CANON_COLS} for row in rows])
    
    # Sort by week_ending_date
    df = df.sort_values("week_ending_date").reset_index(drop=True)
    
    # Deduplicate (keep last if duplicates exist)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["week_ending_date"], keep="last")
    after_dedup = len(df)
    
    if before_dedup > after_dedup:
        print(f"\n⚠️  Removed {before_dedup - after_dedup} duplicate week(s)")
    
    # Write to weeks_log.csv with safe quoting
    log_path = Path("data/live/weeks_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(log_path, index=False, quoting=csv.QUOTE_ALL)
    
    print(f"\n✅ Rebuilt {log_path}")
    print(f"   Total weeks: {len(df)}")
    print(f"   Date range: {df['week_ending_date'].min()} to {df['week_ending_date'].max()}")
    print(f"   TRADE weeks: {(df['action'] == 'TRADE').sum()}")
    print(f"   SKIP weeks: {(df['action'] == 'SKIP').sum()}")
    
    # Summary stats
    if (df["action"] == "TRADE").any():
        trade_df = df[df["action"] == "TRADE"]
        print(f"\n📈 TRADE weeks summary:")
        print(f"   Avg basket size: {trade_df['basket_size'].astype(float).mean():.1f}")
        print(f"   Avg turnover: {trade_df['turnover_pct'].astype(float).mean():.1f}%")
    
    print(f"\n💡 File is now canonical, sorted, and deduplicated")


if __name__ == "__main__":
    rebuild_weeks_log()
