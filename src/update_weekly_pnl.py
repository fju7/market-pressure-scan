"""
Auto-compute weekly P&L from candles data (Friday close)

For TRADE weeks:
- Reads basket.csv to get positions
- Enriches with signal metadata (drivers, conviction, event types)
- Pulls Monday open (entry) and Friday close (exit) from candles
- Computes equal-weight basket return
- Computes SPY benchmark return
- Appends to weekly_pnl.csv with reason codes

For SKIP weeks:
- Appends SKIP row with zeros and skip reason

For ERROR weeks:
- Appends ERROR row with zeros and error reason
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def get_monday_friday_dates(week_end: str) -> tuple[str, str]:
    """
    Get Monday and Friday dates for a given week ending.
    
    Parameters
    ----------
    week_end : str
        Week ending date (YYYY-MM-DD), typically a Friday
    
    Returns
    -------
    tuple[str, str]
        Monday date, Friday date (YYYY-MM-DD)
    """
    friday = pd.Timestamp(week_end)
    monday = friday - pd.Timedelta(days=4)  # Friday - 4 days = Monday
    return monday.strftime("%Y-%m-%d"), friday.strftime("%Y-%m-%d")


def get_price_from_candles(symbol: str, date: str, price_type: str = "open") -> float | None:
    """
    Get price from candles_daily.parquet
    
    Parameters
    ----------
    symbol : str
        Stock symbol
    date : str
        Date (YYYY-MM-DD)
    price_type : str
        "open" or "close"
    
    Returns
    -------
    float | None
        Price or None if not found
    """
    candles_path = Path("data/derived/market_daily/candles_daily.parquet")
    if not candles_path.exists():
        return None
    
    df = pd.read_parquet(candles_path)
    df = df[(df["symbol"] == symbol) & (df["date"] == date)]
    
    if df.empty:
        return None
    
    if price_type == "open":
        return df["open"].iloc[0]
    elif price_type == "close":
        return df["close"].iloc[0]
    else:
        return None


def enrich_position_metadata(symbol: str, week_end: str) -> dict:
    """
    Get signal metadata for a position (reason codes for attribution)
    
    Parameters
    ----------
    symbol : str
        Stock symbol
    week_end : str
        Week ending date (YYYY-MM-DD)
    
    Returns
    -------
    dict
        Metadata: drivers, event_type_primary, conviction, signal_state
    """
    metadata = {
        "drivers": None,
        "event_type_primary": None,
        "conviction": None,
        "signal_state": None,
    }
    
    # Try to load from weekly report (has driver/conviction/signal_state summary)
    report_path = Path(f"data/derived/reports/week_ending={week_end}/weekly_report.md")
    if report_path.exists():
        report_text = report_path.read_text()
        
        # Extract from stock card section for this symbol
        symbol_section = f"### {symbol} —"
        if symbol_section in report_text:
            section_start = report_text.index(symbol_section)
            # Find next ### or end
            next_section = report_text.find("###", section_start + 1)
            if next_section == -1:
                section = report_text[section_start:]
            else:
                section = report_text[section_start:next_section]
            
            # Extract conviction
            if "**Conviction:**" in section:
                conv_start = section.index("**Conviction:**") + len("**Conviction:**")
                conv_end = section.index("|", conv_start)
                metadata["conviction"] = section[conv_start:conv_end].strip()
            
            # Extract signal_state
            if "**Signal state:**" in section:
                state_start = section.index("**Signal state:**") + len("**Signal state:**")
                state_end = section.find("\n", state_start)
                metadata["signal_state"] = section[state_start:state_end].strip()
            
            # Extract drivers
            if "**Drivers:**" in section or "| Drivers |" in section:
                # From table row
                lines = section.split("\n")
                for line in lines:
                    if "| Drivers |" in line:
                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) > 5:
                            metadata["drivers"] = parts[5]
                        break
    
    # Get event types from enriched clusters
    enriched_path = Path(f"data/derived/rep_enriched/week_ending={week_end}/rep_enriched.parquet")
    if enriched_path.exists():
        enriched = pd.read_parquet(enriched_path)
        enriched = enriched[enriched["symbol"] == symbol]
        
        if not enriched.empty:
            # Parse event_json to get event types
            event_types = []
            for _, row in enriched.iterrows():
                if pd.notna(row.get("event_json")):
                    try:
                        event_data = json.loads(row["event_json"]) if isinstance(row["event_json"], str) else row["event_json"]
                        if isinstance(event_data, dict) and "event_type" in event_data:
                            event_types.append(event_data["event_type"])
                    except:
                        pass
            
            if event_types:
                # Use most common or first
                metadata["event_type_primary"] = event_types[0]
    
    return metadata


def update_weekly_pnl(week_end: str) -> None:
    """
    Update weekly_pnl.csv with P&L calculation
    
    Parameters
    ----------
    week_end : str
        Week ending date (YYYY-MM-DD)
    """
    # Read basket
    basket_path = Path(f"data/derived/baskets/week_ending={week_end}/basket.csv")
    if not basket_path.exists():
        raise FileNotFoundError(f"Basket not found: {basket_path}")
    
    basket = pd.read_csv(basket_path)
    action = basket["action"].iloc[0]
    
    # Get Monday and Friday dates
    monday_date, friday_date = get_monday_friday_dates(week_end)
    
    # Initialize weekly_pnl.csv if needed
    pnl_path = Path("data/live/weekly_pnl.csv")
    pnl_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not pnl_path.exists():
        pnl_df = pd.DataFrame(columns=[
            "week_ending_date", "symbol", "entry_price", "exit_price",
            "return_pct", "benchmark_return_pct", "active_return_pct",
            "drivers", "event_type_primary", "conviction", "signal_state", "notes"
        ])
        pnl_df.to_csv(pnl_path, index=False)
    
    # Read existing weekly_pnl
    pnl_df = pd.read_csv(pnl_path)
    
    # Check if week already logged
    if week_end in pnl_df["week_ending_date"].values:
        print(f"Week {week_end} already logged in weekly_pnl.csv. Skipping.")
        return
    
    if action == "SKIP":
        # Log SKIP week
        skip_reason = basket["notes"].iloc[0] if "notes" in basket.columns else "Low information week"
        skip_row = {
            "week_ending_date": week_end,
            "symbol": "SKIP",
            "entry_price": 0.0,
            "exit_price": 0.0,
            "return_pct": 0.0,
            "benchmark_return_pct": 0.0,
            "active_return_pct": 0.0,
            "drivers": None,
            "event_type_primary": None,
            "conviction": None,
            "signal_state": None,
            "notes": skip_reason
        }
        pnl_df = pd.concat([pnl_df, pd.DataFrame([skip_row])], ignore_index=True)
        pnl_df.to_csv(pnl_path, index=False)
        print(f"✓ Week {week_end}: SKIP logged (reason: {skip_reason})")
        return
    
    if action == "ERROR":
        # Log ERROR week (pipeline failure)
        error_reason = basket["notes"].iloc[0] if "notes" in basket.columns else "Pipeline error"
        error_row = {
            "week_ending_date": week_end,
            "symbol": "ERROR",
            "entry_price": 0.0,
            "exit_price": 0.0,
            "return_pct": 0.0,
            "benchmark_return_pct": 0.0,
            "active_return_pct": 0.0,
            "drivers": None,
            "event_type_primary": None,
            "conviction": None,
            "signal_state": None,
            "notes": error_reason
        }
        pnl_df = pd.concat([pnl_df, pd.DataFrame([error_row])], ignore_index=True)
        pnl_df.to_csv(pnl_path, index=False)
        print(f"⚠️ Week {week_end}: ERROR logged (reason: {error_reason})")
        return
    
    # TRADE week - calculate P&L
    positions = basket[basket["action"] == "TRADE"].copy()
    
    # Filter out rows with no symbol (header rows)
    positions = positions[positions["symbol"].notna()].copy()
    
    if positions.empty:
        print(f"No TRADE positions in basket for week {week_end}")
        return
    
    # Calculate returns for each position
    returns = []
    for _, row in positions.iterrows():
        symbol = row["symbol"]
        entry_price = get_price_from_candles(symbol, monday_date, "open")
        exit_price = get_price_from_candles(symbol, friday_date, "close")
        
        if entry_price is None or exit_price is None:
            print(f"⚠️  Missing candle data for {symbol}: entry={entry_price}, exit={exit_price}")
            continue
        
        ret_pct = ((exit_price - entry_price) / entry_price) * 100
        
        # Get signal metadata - try basket first, then fall back to report
        metadata = {
            "drivers": row.get("drivers") if "drivers" in row and pd.notna(row.get("drivers")) else None,
            "event_type_primary": None,
            "conviction": row.get("conviction") if "conviction" in row and pd.notna(row.get("conviction")) else None,
            "signal_state": row.get("signal_state") if "signal_state" in row and pd.notna(row.get("signal_state")) else None,
        }
        
        # If not in basket, try to get from report
        if metadata["conviction"] is None or metadata["drivers"] is None:
            report_metadata = enrich_position_metadata(symbol, week_end)
            if metadata["conviction"] is None:
                metadata["conviction"] = report_metadata["conviction"]
            if metadata["drivers"] is None:
                metadata["drivers"] = report_metadata["drivers"]
            if metadata["signal_state"] is None:
                metadata["signal_state"] = report_metadata["signal_state"]
            if metadata["event_type_primary"] is None:
                metadata["event_type_primary"] = report_metadata["event_type_primary"]
        
        returns.append({
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return_pct": ret_pct,
            "metadata": metadata
        })
    
    if not returns:
        print(f"No valid returns calculated for week {week_end}")
        return
    
    # Calculate benchmark return (SPY)
    spy_entry = get_price_from_candles("SPY", monday_date, "open")
    spy_exit = get_price_from_candles("SPY", friday_date, "close")
    
    if spy_entry is None or spy_exit is None:
        print(f"⚠️  Missing SPY candle data: entry={spy_entry}, exit={spy_exit}")
        benchmark_return = 0.0
    else:
        benchmark_return = ((spy_exit - spy_entry) / spy_entry) * 100
    
    # Log each position
    for ret in returns:
        active_return = ret["return_pct"] - benchmark_return
        log_row = {
            "week_ending_date": week_end,
            "symbol": ret["symbol"],
            "entry_price": ret["entry_price"],
            "exit_price": ret["exit_price"],
            "return_pct": ret["return_pct"],
            "benchmark_return_pct": benchmark_return,
            "active_return_pct": active_return,
            "drivers": ret["metadata"]["drivers"],
            "event_type_primary": ret["metadata"]["event_type_primary"],
            "conviction": ret["metadata"]["conviction"],
            "signal_state": ret["metadata"]["signal_state"],
            "notes": f"Auto-calculated from candles (Mon open → Fri close)"
        }
        pnl_df = pd.concat([pnl_df, pd.DataFrame([log_row])], ignore_index=True)
    
    # Save
    pnl_df.to_csv(pnl_path, index=False)
    
    # Calculate and display summary
    basket_return = sum(r["return_pct"] for r in returns) / len(returns)
    avg_active = basket_return - benchmark_return
    
    print(f"✓ Week {week_end}: TRADE P&L logged")
    print(f"  Positions: {len(returns)}")
    print(f"  Basket return: {basket_return:.2f}%")
    print(f"  SPY return: {benchmark_return:.2f}%")
    print(f"  Active return: {avg_active:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Update weekly P&L from candles")
    parser.add_argument(
        "--week_end",
        type=str,
        required=True,
        help="Week ending date (YYYY-MM-DD)"
    )
    args = parser.parse_args()
    
    update_weekly_pnl(args.week_end)


if __name__ == "__main__":
    main()
