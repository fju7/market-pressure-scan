import argparse
import subprocess
import sys
from pathlib import Path


def sh(cmd):
    print("\n▶ Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser("Run full weekly market pressure pipeline")
    ap.add_argument("--week_end", required=True, help="Week ending date YYYY-MM-DD")
    ap.add_argument("--universe", default="sp500_universe.csv")
    ap.add_argument("--max_clusters_per_symbol", type=int, default=1)
    ap.add_argument("--skip_backtest", action="store_true")
    args = ap.parse_args()

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
        "--max_clusters_per_symbol", str(args.max_clusters_per_symbol)
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
        "--week_end", args.week_end
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
        "--top_n", "20",
        "--skip_low_info"
    ])

    # 8) Trader sheet
    sh([
        py, "-m", "src.trader_sheet",
        "--week_end", args.week_end
    ])

    # 9) Backtest (optional, skipped for single week runs)
    # The backtest requires --from_week_end and --to_week_end for multi-week backtests
    # For single week pipeline runs, it doesn't make sense to run backtest
    if not args.skip_backtest:
        print("\n⚠️  Backtest skipped for single-week runs.")
        print("   To run backtest, use: python -m src.backtest_weekly --universe ... --from_week_end ... --to_week_end ...")

    print("\n✅ Weekly pipeline completed successfully.")


if __name__ == "__main__":
    main()
