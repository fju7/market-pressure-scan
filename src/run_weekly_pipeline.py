import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd

from . import config


def sh(cmd):
    print("\n▶ Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser("Run full weekly market pressure pipeline")
    ap.add_argument("--week_end", required=True, help="Week ending date YYYY-MM-DD")
    ap.add_argument("--universe", default="sp500_universe.csv")
    ap.add_argument("--regime", default="news-novelty-v1", help="Regime ID (e.g., news-novelty-v1, news-novelty-v1b)")
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
        "--regime", args.regime
    ])

    # 6) Weekly report
    sh([
        py, "-m", "src.report_weekly",
        "--week_end", args.week_end
    ])

    # 7) Export basket
    sh([
        py, "-m", "src.export_basket",
        "--week_end", args.week_end,
        "--skip_low_info"
        # top_n now comes from CONFIG.yaml
    ])

    # 8) Trader sheet
    sh([
        py, "-m", "src.trader_sheet",
        "--week_end", args.week_end
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
