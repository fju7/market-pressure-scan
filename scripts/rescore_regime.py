# scripts/rescore_regime.py
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True)
    ap.add_argument("--weeks", nargs="+", required=True)  # list of YYYY-MM-DD
    ap.add_argument("--universe", default="sp500_universe.csv")
    ap.add_argument("--lookback_weeks", type=int, default=12)
    args = ap.parse_args()

    for w in args.weeks:
        cmd = [
            "python", "-m", "src.features_scores",
            "--universe", args.universe,
            "--week_end", w,
            "--lookback_weeks", str(args.lookback_weeks),
            "--regime", args.regime,
        ]
        print("RUN:", " ".join(cmd))
        subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
