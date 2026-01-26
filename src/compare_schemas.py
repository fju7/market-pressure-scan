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
        regime = "news-novelty-v1"
        schema_a = "news-novelty-v1"
        schema_b = "news-novelty-v1b"

        weeks = discover_common_weeks(regime, schema_a, schema_b)
        if not weeks:
            raise SystemExit(f"No common weeks found for {schema_a} vs {schema_b} under {regime}")

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

    if __name__ == "__main__":
        main()
