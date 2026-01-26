from __future__ import annotations

from pathlib import Path
import traceback

import pandas as pd

BASE = Path("data/derived/scores_weekly")
OUT_DIR = Path("data/derived/analysis")

def discover_common_weeks(regime: str, schema_a: str, schema_b: str) -> list[str]:
    def weeks_for(schema: str) -> set[str]:
        root = BASE / f"regime={regime}" / f"schema={schema}"
        if not root.exists():
            return set()
        out = set()
        for p in root.iterdir():
            if p.is_dir() and p.name.startswith("week_ending="):
                out.add(p.name.split("=", 1)[1])
        return out

    return sorted(weeks_for(schema_a) & weeks_for(schema_b))

def load_scores(regime: str, schema: str, week_end: str) -> pd.DataFrame:
    p = BASE / f"regime={regime}" / f"schema={schema}" / f"week_ending={week_end}" / "scores_weekly.parquet"
    return pd.read_parquet(p)

def compare_week(regime: str, week_end: str, a: str, b: str) -> pd.DataFrame:
    da = load_scores(regime, a, week_end).rename(columns={"score": "score_a", "rank": "rank_a"})
    db = load_scores(regime, b, week_end).rename(columns={"score": "score_b", "rank": "rank_b"})

    key = "symbol" if "symbol" in da.columns and "symbol" in db.columns else None
    if key is None:
        raise KeyError(f"Missing 'symbol' column in scores for week {week_end}")

    d = da[[key, "score_a", "rank_a"]].merge(db[[key, "score_b", "rank_b"]], on=key, how="inner")
    d["score_delta"] = d["score_b"] - d["score_a"]
    d["rank_delta"] = d["rank_b"] - d["rank_a"]
    d["week_end"] = week_end
    return d

def summarize(cmp_df: pd.DataFrame, top_n: int = 20) -> dict:
    # top-N overlap based on rank_a vs rank_b
    top_a = set(cmp_df.nsmallest(top_n, "rank_a")["symbol"])
    top_b = set(cmp_df.nsmallest(top_n, "rank_b")["symbol"])
    overlap = len(top_a & top_b) / max(1, top_n)

    return {
        "week_end": cmp_df["week_end"].iloc[0],
        "n_common": int(len(cmp_df)),
        "topN": top_n,
        "topN_overlap_pct": round(100.0 * overlap, 2),
        "median_abs_rank_delta": float(cmp_df["rank_delta"].abs().median()),
        "max_abs_rank_delta": float(cmp_df["rank_delta"].abs().max()),
        "median_abs_score_delta": float(cmp_df["score_delta"].abs().median()),
        "max_abs_score_delta": float(cmp_df["score_delta"].abs().max()),
    }

def main():
    # marker files (no stdout dependency)
    Path("/tmp/compare_imported.txt").write_text("imported\n")
    Path("/tmp/compare_started.txt").write_text("started\n")

    try:
        regime = "news-novelty-v1"
        schema_a = "news-novelty-v1"
        schema_b = "news-novelty-v1b"

        OUT_DIR.mkdir(parents=True, exist_ok=True)

        weeks = discover_common_weeks(regime, schema_a, schema_b)
        Path("/tmp/compare_weeks.txt").write_text("\n".join(weeks) + ("\n" if weeks else ""))

        all_rows = []
        for w in weeks:
            cmp_df = compare_week(regime, w, schema_a, schema_b)

            # write per-week detail
            out_week = OUT_DIR / f"schema_compare_week_end={w}.parquet"
            cmp_df.to_parquet(out_week, index=False)

            all_rows.append(summarize(cmp_df, top_n=20))

        summary = pd.DataFrame(all_rows).sort_values("week_end") if all_rows else pd.DataFrame(
            columns=["week_end","n_common","topN","topN_overlap_pct","median_abs_rank_delta","max_abs_rank_delta","median_abs_score_delta","max_abs_score_delta"]
        )

        out_csv = OUT_DIR / "schema_compare_summary.csv"
        summary.to_csv(out_csv, index=False)

        Path("/tmp/compare_done.txt").write_text(f"done; weeks={len(weeks)}; wrote={out_csv}\n")

    except Exception:
        Path("/tmp/compare_fatal.txt").write_text(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
from pathlib import Path
import pandas as pd

BASE = Path("data/derived/scores_weekly")

WEEKS = [
    "2025-11-13","2025-11-20","2025-11-27","2025-12-04",
    "2025-12-18","2025-12-25","2026-01-09","2026-01-16","2026-01-23",
]

def load_scores(regime: str, schema: str, week_end: str) -> pd.DataFrame:
    p = BASE / f"regime={regime}" / f"schema={schema}" / f"week_ending={week_end}" / "scores_weekly.parquet"
    df = pd.read_parquet(p)
    # rescore_week returns columns including: symbol, score, rank, ... (plus feature cols)

    # --- PATCH: Discovery-driven, skip missing weeks ---
    from pathlib import Path
    import pandas as pd

    BASE = Path("data/derived/scores_weekly")

    def week_dirs(regime: str, schema: str):
        root = BASE / f"regime={regime}" / f"schema={schema}"
        if not root.exists():
            return []
        return sorted(root.glob("week_ending=*"))

    def discover_common_weeks(regime: str, schema_a: str, schema_b: str):
        wa = {p.name.split("=", 1)[1] for p in week_dirs(regime, schema_a)}
        wb = {p.name.split("=", 1)[1] for p in week_dirs(regime, schema_b)}
        return sorted(wa & wb)

    def load_scores(regime: str, schema: str, week_end: str) -> pd.DataFrame:
        p = BASE / f"regime={regime}" / f"schema={schema}" / f"week_ending={week_end}" / "scores_weekly.parquet"
        from pathlib import Path
        import traceback

        # Import-side marker: proves the file is executed at all
        Path("/tmp/compare_imported.txt").write_text("imported\n")

        def _fatal_hook(e: BaseException):
            Path("/tmp/compare_fatal.txt").write_text(traceback.format_exc())
            raise

        try:
            Path("/tmp/compare_boot.txt").write_text("boot\n")
        except Exception:
            pass
        if not p.exists():
            raise FileNotFoundError(str(p))
        df = pd.read_parquet(p)
        if "symbol" not in df.columns or "score" not in df.columns:
            raise KeyError(f"{p} missing required columns. Have={list(df.columns)[:30]}")
        return df[["symbol", "score"]].copy()

    def compare_week(regime: str, week_end: str, a: str, b: str) -> pd.DataFrame:
        da = load_scores(regime, a, week_end).rename(columns={"score": "score_a"})
        db = load_scores(regime, b, week_end).rename(columns={"score": "score_b"})
        out = da.merge(db, on="symbol", how="outer")
        out["delta"] = out["score_b"] - out["score_a"]
        out["rank_a"] = out["score_a"].rank(ascending=False, method="min")
        out["rank_b"] = out["score_b"].rank(ascending=False, method="min")
        out["rank_delta"] = out["rank_b"] - out["rank_a"]
        return out.sort_values("delta", ascending=False)

    def main():
        from pathlib import Path
        Path("/tmp/compare_started.txt").write_text("started\n")
        regime = "news-novelty-v1"
        schema_a = "news-novelty-v1"
        schema_b = "news-novelty-v1b"

        print("DEBUG: BASE =", BASE.resolve())

        root = BASE / f"regime={regime}"
        print("DEBUG: regime dir exists:", root.exists())
        if root.exists():
            print("DEBUG: schemas under regime:", [p.name for p in root.iterdir()])

        weeks = discover_common_weeks(regime, schema_a, schema_b)
        print(f"DEBUG: common weeks = {weeks}")

        if not weeks:
            print("DEBUG: No common weeks found — exiting")
            return

        out_dir = Path("data/derived/analysis")
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for w in weeks:
            try:
                cmp = compare_week(regime, w, schema_a, schema_b)
            except Exception as e:
                print(f"⚠️ Skipping week {w}: {e}")
                continue

            n_syms = int(cmp["symbol"].nunique(dropna=True))
            topN = min(20, n_syms) if n_syms else 0

            top_a = set(cmp.sort_values("score_a", ascending=False).head(topN)["symbol"].dropna())
                Path("/tmp/compare_started.txt").write_text("started\n")
                try:
                    regime = "news-novelty-v1"
                    schema_a = "news-novelty-v1"
                    schema_b = "news-novelty-v1b"

                    print("DEBUG: BASE =", BASE.resolve())

                    root = BASE / f"regime={regime}"
                    print("DEBUG: regime dir exists:", root.exists())
                    if root.exists():
                        print("DEBUG: schemas under regime:", [p.name for p in root.iterdir()])

                    weeks = discover_common_weeks(regime, schema_a, schema_b)
                    print(f"DEBUG: common weeks = {weeks}")

                    if not weeks:
                        print("DEBUG: No common weeks found — exiting")
                        return

                    out_dir = Path("data/derived/analysis")
                    out_dir.mkdir(parents=True, exist_ok=True)

                    rows = []
                    for w in weeks:
                        try:
                            cmp = compare_week(regime, w, schema_a, schema_b)
                        except Exception as e:
                            print(f"⚠️ Skipping week {w}: {e}")
                            continue

                        n_syms = int(cmp["symbol"].nunique(dropna=True))
                        topN = min(20, n_syms) if n_syms else 0

                        top_a = set(cmp.sort_values("score_a", ascending=False).head(topN)["symbol"].dropna())
                        top_b = set(cmp.sort_values("score_b", ascending=False).head(topN)["symbol"].dropna())
                        overlap = len(top_a & top_b) if topN else 0

                        rows.append({
                            "week_end": w,
                            "n_symbols": n_syms,
                            "topN": topN,
                            "topN_overlap": overlap,
                            "topN_overlap_pct": (overlap / topN) if topN else None,
                            "mean_delta": float(cmp["delta"].mean(skipna=True)),
                            "median_abs_rank_delta": float(cmp["rank_delta"].abs().median(skipna=True)),
                        })

                        cmp.to_parquet(out_dir / f"schema_compare_week_end={w}.parquet", index=False)

                    summary = pd.DataFrame(rows).sort_values("week_end")
                    summary.to_csv(out_dir / "schema_compare_summary.csv", index=False)
                    print(summary)
                except Exception as e:
                    _fatal_hook(e)

            rows.append({
                "week_end": w,
                "n_symbols": n_syms,
                "topN": topN,
                "topN_overlap": overlap,
                "topN_overlap_pct": (overlap / topN) if topN else None,
                "mean_delta": float(cmp["delta"].mean(skipna=True)),
                "median_abs_rank_delta": float(cmp["rank_delta"].abs().median(skipna=True)),
            })

            cmp.to_parquet(out_dir / f"schema_compare_week_end={w}.parquet", index=False)

        summary = pd.DataFrame(rows).sort_values("week_end")
        summary.to_csv(out_dir / "schema_compare_summary.csv", index=False)
        print(summary)

    if __name__ == "__main__":
        main()
