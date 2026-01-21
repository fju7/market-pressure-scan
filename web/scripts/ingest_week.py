import os
import json
import csv
from datetime import datetime
from pathlib import Path

from supabase import create_client

# pip install supabase
# In Codespaces: python -m pip install supabase

REQUIRED = [
    "report_meta.json",
    "basket.csv",
    "scores_weekly.parquet",
    "ops_compact_friday.log",
]


def supabase_admin():
    url = os.environ["NEXT_PUBLIC_SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def load_report_meta(p: Path):
    with p.open("r") as f:
        return json.load(f)


def load_basket_symbols(p: Path):
    symbols = []
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        # Expect a column like symbol/ticker; be flexible.
        cols = [c.lower() for c in reader.fieldnames or []]
        sym_col = (
            "symbol"
            if "symbol" in cols
            else ("ticker" if "ticker" in cols else None)
        )
        if not sym_col:
            raise ValueError(
                f"basket.csv missing symbol/ticker column; found {reader.fieldnames}"
            )
        for row in reader:
            symbols.append(row[sym_col].strip().upper())
    return symbols


def ingest_week(
    artifact_dir: Path, experiment_name: str, baseline_version: str, run_type: str, repo_root: Path = None
):
    if repo_root is None:
        repo_root = artifact_dir.parent.parent.parent  # default fallback
    missing = [f for f in REQUIRED if not (artifact_dir / f).exists()]

    meta = None
    if (artifact_dir / "report_meta.json").exists():
        meta = load_report_meta(artifact_dir / "report_meta.json")

    # week_ending: prefer metadata; fallback to folder name if it's yyyy-mm-dd
    folder_date = artifact_dir.name
    week_ending = meta.get("week_ending") if meta else None
    week_ending = week_ending or folder_date
    # normalize date
    week_ending = datetime.strptime(week_ending, "%Y-%m-%d").date().isoformat()

    # Determine week type
    week_type = "DATA_IMPAIRED"
    completeness_pass = len(missing) == 0

    # If you have explicit classification in report_meta, use it:
    if meta and "week_type" in meta:
        week_type = meta["week_type"]
    else:
        # fallback: if completeness passes, assume CLEAN_TRADE if basket exists else CLEAN_SKIP
        if completeness_pass:
            week_type = (
                "CLEAN_TRADE"
                if (artifact_dir / "basket.csv").exists()
                else "CLEAN_SKIP"
            )

    # Optional: capture skip reasons if report_meta includes them
    skip_reasons = meta.get("skip_reasons", {}) if meta else {}

    sb = supabase_admin()

    # Upsert strategy_version
    sb.table("strategy_versions").upsert(
        {
            "experiment_name": experiment_name,
            "baseline_version": baseline_version,
            "locked_date": "2026-01-21",
            "benchmark_symbol": "SPY",
        },
        on_conflict="experiment_name,baseline_version",
    ).execute()
    # fetch id
    sv_row = (
        sb.table("strategy_versions")
        .select("id")
        .eq("experiment_name", experiment_name)
        .eq("baseline_version", baseline_version)
        .single()
        .execute()
    )
    sv_id = sv_row.data["id"]

    # Upsert week
    week_payload = {
        "strategy_version_id": sv_id,
        "run_type": run_type,
        "week_ending": week_ending,
        "universe_type": meta.get("universe_type", "test") if meta else "test",
        "num_symbols_covered": meta.get("num_symbols_covered") if meta else None,
        "week_type": week_type,
        "completeness_pass": completeness_pass,
        "missing_artifacts": missing,
        "skip_reasons": skip_reasons,
        "artifact_root": str(artifact_dir),
    }

    sb.table("weeks").upsert(
        week_payload,
        on_conflict="strategy_version_id,run_type,week_ending",
    ).execute()
    week_row = (
        sb.table("weeks")
        .select("id")
        .eq("strategy_version_id", sv_id)
        .eq("run_type", run_type)
        .eq("week_ending", week_ending)
        .single()
        .execute()
    )
    week_id = week_row.data["id"]

    # Holdings
    if (artifact_dir / "basket.csv").exists() and week_type == "CLEAN_TRADE":
        syms = load_basket_symbols(artifact_dir / "basket.csv")
        # equal weight if you don't have weights in file
        w = 1.0 / max(len(syms), 1)
        # clear + insert (simple)
        sb.table("basket_holdings").delete().eq("week_id", week_id).execute()
        rows = [{"week_id": week_id, "symbol": s, "weight": w} for s in syms]
        if rows:
            sb.table("basket_holdings").insert(rows).execute()

    # Performance numbers from bt_weekly.parquet
    # Parse the backtest ledger and upsert weekly_performance
    backtest_ledger = repo_root / "data" / "derived" / "backtest" / "bt_weekly.parquet"
    if backtest_ledger.exists():
        try:
            import pandas as pd
            bt_df = pd.read_parquet(backtest_ledger)
            # Find row matching this week's signal_week_end
            week_rows = bt_df[bt_df['signal_week_end'] == week_ending]
            if not week_rows.empty:
                row = week_rows.iloc[0]
                perf_payload = {
                    "week_id": week_id,
                    "strategy_return": float(row['net_return']),
                    "benchmark_return": float(row['spy_return']),
                    "active_return": float(row['active_net_return']),
                    "transaction_cost_bps": float(row['tcost'] * 10000),  # convert to bps
                    "basket_size": int(row['n_positions']),
                }
                sb.table("weekly_performance").upsert(perf_payload).execute()
                print(f"  → Loaded performance: strategy_return={row['net_return']:.4f}, benchmark={row['spy_return']:.4f}")
        except ImportError:
            print(f"  ⚠ pandas not available; skipping weekly_performance")
        except Exception as e:
            print(f"  ⚠ Error loading weekly_performance: {e}")

    print(
        f"Ingested {artifact_dir} → week_id={week_id}, week_type={week_type}, complete={completeness_pass}, missing={missing}"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Ingest a single week of backtest results into Supabase"
    )
    ap.add_argument(
        "--week-ending",
        required=True,
        help="Week ending date in YYYY-MM-DD format (e.g., 2026-01-02)",
    )
    ap.add_argument("--experiment", default="news-novelty-v1")
    ap.add_argument("--baseline", default="V1")
    ap.add_argument(
        "--run-type", default="shadow", choices=["shadow", "real"]
    )
    args = ap.parse_args()

    # Infer artifact directory from canonical paths
    # Expected structure: data/derived/scores_weekly/week_ending=YYYY-MM-DD/
    repo_root = Path(__file__).parent.parent.parent  # web/ -> . (repo root)
    artifact_dir = repo_root / "data" / "derived" / "scores_weekly" / f"week_ending={args.week_ending}"
    
    if not artifact_dir.exists():
        raise FileNotFoundError(
            f"Week folder not found: {artifact_dir}\n"
            f"Expected artifact structure:\n"
            f"  data/derived/scores_weekly/week_ending={args.week_ending}/\n"
            f"    report_meta.json\n"
            f"    basket.csv\n"
            f"    scores_weekly.parquet\n"
            f"    ops_compact_friday.log"
        )

    ingest_week(artifact_dir, args.experiment, args.baseline, args.run_type, repo_root)
