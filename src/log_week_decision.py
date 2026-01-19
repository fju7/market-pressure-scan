"""
Log weekly decision (TRADE or SKIP) to weeks_log.csv

This script reads the basket.csv and report_meta.json for a given week
and logs the decision to weeks_log.csv for tracking purposes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


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
    
    # Read report metadata
    meta_path = Path(f"data/derived/reports/week_ending={week_end}/report_meta.json")
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    
    # Prepare log entry
    # Use cluster_count from report_meta.json (single source of truth)
    log_entry = {
        "week_ending_date": week_end,
        "action": action,
        "num_clusters": meta.get("cluster_count", 0),  # Changed from num_clusters
        "avg_novelty_z": meta.get("avg_novelty_z", 0.0),
        "avg_event_intensity_z": meta.get("avg_event_intensity_z", 0.0),
        "recap_pct": meta.get("recap_pct", 0.0),
        "is_low_info": meta.get("is_low_information_week", False),
        "num_positions": len(basket) if action == "TRADE" else 0,
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    if action == "SKIP":
        log_entry["skip_reason"] = basket["reason"].iloc[0] if "reason" in basket.columns else ""
    else:
        log_entry["skip_reason"] = ""
    
    # Append to weeks_log.csv
    log_path = Path("data/live/weeks_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([log_entry])
    
    if log_path.exists():
        # Check if this week is already logged
        existing = pd.read_csv(log_path)
        if week_end in existing["week_ending_date"].values:
            print(f"⚠️  Week {week_end} already logged. Skipping.")
            return action
        
        # Append mode
        df.to_csv(log_path, mode="a", header=False, index=False)
        print(f"✅ Appended week {week_end} to {log_path}")
    else:
        # Create new file with header
        df.to_csv(log_path, index=False)
        print(f"✅ Created {log_path} with week {week_end}")
    
    # Display summary
    print(f"\n📊 Week {week_end} summary:")
    print(f"   Action: {action}")
    print(f"   Clusters: {log_entry['num_clusters']}")
    print(f"   Recap %: {log_entry['recap_pct']:.0f}%")
    print(f"   Low info: {log_entry['is_low_info']}")
    
    if action == "SKIP":
        print(f"   ⏭️  Reason: {log_entry['skip_reason'][:80]}")
    else:
        print(f"   📈 Positions: {log_entry['num_positions']}")
    
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
