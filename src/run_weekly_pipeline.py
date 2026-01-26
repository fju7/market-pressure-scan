import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

from . import config


def sh(cmd):
    print("\n▶ Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def log_error_week(week_end: str, error_type: str, regime: str, scores_path: Path, exception: Exception):
    """
    Log an error week to weeks_log.csv to prevent silent week loss.
    
    This is a fallback logger for when scores exist but downstream steps fail.
    
    Parameters
    ----------
    week_end : str
        Week ending date
    error_type : str
        Error classification (e.g., ERROR_POST_SCORES)
    regime : str
        Regime ID that was being processed
    scores_path : Path
        Path to the scores file that exists
    exception : Exception
        The exception that was caught
    """
    log_path = Path("data/live/weeks_log.csv")
    
    # Extract concise error info
    error_class = type(exception).__name__
    error_msg = str(exception).split('\n')[0][:100]  # First line, truncated
    
    # Build structured error reason
    error_reason = (
        f"{error_type} | "
        f"regime={regime} | "
        f"scores={scores_path} | "
        f"{error_class}: {error_msg}"
    )
    
    # Prepare error row
    error_row = {
        "week_ending_date": week_end,
        "action": "ERROR",
        "basket_size": 0,
        "overlap_pct": 0.0,
        "turnover_pct": 0.0,
        "num_clusters": 0,
        "avg_novelty_z": 0.0,
        "avg_event_intensity_z": 0.0,
        "recap_pct": 0.0,
        "is_low_info": "",
        "num_positions": 0,
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "skip_reason": error_reason[:500],  # Truncate to prevent CSV issues
    }
    
    # Append to weeks_log
    if log_path.exists():
        df = pd.read_csv(log_path)
        # Don't duplicate if already logged
        if week_end not in df["week_ending_date"].values:
            df = pd.concat([df, pd.DataFrame([error_row])], ignore_index=True)
            df.to_csv(log_path, index=False)
            print(f"✓ Logged error week to {log_path}")
    else:
        # Create new log file
        df = pd.DataFrame([error_row])
        df.to_csv(log_path, index=False)
        print(f"✓ Created {log_path} with error week")


def main():
    ap = argparse.ArgumentParser("Run full weekly market pressure pipeline")
    ap.add_argument("--week_end", required=True, help="Week ending date YYYY-MM-DD")
    ap.add_argument("--universe", default="sp500_universe.csv")
    ap.add_argument("--regime", default="news-novelty-v1", help="Regime ID (e.g., news-novelty-v1, news-novelty-v1b)")
    ap.add_argument(
        "--schema",
        default="news-novelty-v1b",
        help="Scoring schema ID (e.g., news-novelty-v1, news-novelty-v1b)"
    )
    ap.add_argument("--max_clusters_per_symbol", type=int, default=None, 
                    help="Max clusters per symbol (default: from CONFIG.yaml)")
    ap.add_argument("--skip_backtest", action="store_true")
    args = ap.parse_args()

    # CRITICAL: Validate against week_end.txt (single source of truth)
    week_end_file = Path("week_end.txt")
    if week_end_file.exists():
        week_end_from_file = week_end_file.read_text().strip()
        if week_end_from_file != args.week_end:
            print(f"❌ CRITICAL ERROR: week_end mismatch!")
            print(f"   week_end.txt:   {week_end_from_file}")
            print(f"   --week_end arg: {args.week_end}")
            print(f"   These must match to prevent artifact contamination.")
            sys.exit(1)
        print(f"✓ week_end validation passed: {args.week_end}")
    else:
        print(f"⚠️  Warning: week_end.txt not found (local run?)")
        print(f"   Proceeding with --week_end={args.week_end}")

    # Validate config
    config.validate_config()
    
    # Validate universe file
    universe_path = Path(args.universe)
    if not universe_path.exists():
        print(f"❌ ERROR: Universe file not found: {args.universe}")
        sys.exit(1)
    
    universe_df = pd.read_csv(universe_path)
    num_symbols = len(universe_df)
    print(f"✓ Loaded universe: {args.universe} ({num_symbols} symbols)")
    
    # CRITICAL GUARD: Fail if production mode and universe too small
    MIN_PRODUCTION_SYMBOLS = 350
    if num_symbols < MIN_PRODUCTION_SYMBOLS:
        print(f"❌ CRITICAL ERROR: Universe has only {num_symbols} symbols (minimum: {MIN_PRODUCTION_SYMBOLS})")
        print(f"   This looks like a test universe file. Expected ~500 S&P 500 symbols.")
        print(f"   Refusing to run pipeline to prevent invalid experiment data.")
        sys.exit(1)
    
    # Use config defaults if not specified
    max_clusters = args.max_clusters_per_symbol if args.max_clusters_per_symbol is not None else config.get_max_clusters_per_symbol()
    
    print(f"\n🔒 Experiment: {config.get_experiment_name()} v{config.get_experiment_version()}")
    print(f"   Regime: {args.regime}")
    print(f"   Schema: {args.schema}")
    print(f"   Max clusters/symbol: {max_clusters}")
    print(f"   Basket size: {config.get_basket_size()}")
    print(f"   Skip rules: {'ENABLED' if config.get_skip_rules_enabled() else 'DISABLED'}\n")

    py = sys.executable

    # 1) Market candles
    sh([
        py, "-m", "src.ingest_market_candles",
        "--universe", args.universe,
        "--week_end", args.week_end
    ])

    # 2) Company news
    sh([
        py, "-m", "src.ingest_company_news",
        "--universe", args.universe,
        "--week_end", args.week_end
    ])

    # 3) Cluster news
    sh([
        py, "-m", "src.cluster_news",
        "--week_end", args.week_end,
        "--max_clusters_per_symbol", str(max_clusters)
    ])

    # 4) Enrich clusters (OpenAI)
    sh([
        py, "-m", "src.enrich_reps_openai",
        "--week_end", args.week_end
    ])

    # 5) Features + scores
    sh([
        py, "-m", "src.features_scores",
        "--universe", args.universe,
        "--week_end", args.week_end,
        "--regime", args.regime,
        "--schema", args.schema
    ])

    # 6) Weekly report (with error fallback)
    try:
        sh([
            py, "-m", "src.report_weekly",
            "--week_end", args.week_end,
            "--regime", args.regime,
            "--schema", args.schema
        ])
    except subprocess.CalledProcessError as e:
        # If scores exist but report fails, log it to prevent silent week loss
        scores_path = Path(
            f"data/derived/scores_weekly/regime={args.regime}/schema={args.schema}/"
            f"week_ending={args.week_end}/scores_weekly.parquet"
        )
        if scores_path.exists():
            print(f"⚠️  WARNING: Scores exist but report_weekly failed. Logging ERROR_POST_SCORES to weeks_log.")
            log_error_week(
                week_end=args.week_end,
                error_type="ERROR_POST_SCORES",
                regime=args.regime,
                scores_path=scores_path,
                exception=e
            )
        raise  # Re-raise to still fail the pipeline

    # 7) Export basket
    sh([
        py, "-m", "src.export_basket",
        "--week_end", args.week_end,
        "--regime", args.regime,
        "--schema", args.schema,
        "--skip_low_info"
        # top_n now comes from CONFIG.yaml
    ])

    # 8) Trader sheet
    sh([
        py, "-m", "src.trader_sheet",
        "--week_end", args.week_end,
        "--regime", args.regime,
        "--schema", args.schema
    ])

    # 9) Log week decision
    sh([
        py, "-m", "src.log_week_decision",
        "--week_end", args.week_end
    ])

    # 10) Backtest (optional, skipped for single week runs)
    # The backtest requires --from_week_end and --to_week_end for multi-week backtests
    # For single week pipeline runs, it doesn't make sense to run backtest
    if not args.skip_backtest:
        print("\n⚠️  Backtest skipped for single-week runs.")
        print("   To run backtest, use: python -m src.backtest_weekly --universe ... --from_week_end ... --to_week_end ...")

    print("\n✅ Weekly pipeline completed successfully.")


if __name__ == "__main__":
    main()
