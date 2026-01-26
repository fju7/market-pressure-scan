from __future__ import annotations

from pathlib import Path
import traceback
import pandas as pd

BASE = Path("data/derived/scores_weekly")
OUT_DIR = Path("data/derived/analysis")


def _schema_root(regime: str, schema: str) -> Path:
    return BASE / f"regime={regime}" / f"schema={schema}"


def discover_common_weeks(regime: str, schema_a: str, schema_b: str) -> list[str]:
    def weeks_for(schema: str) -> set[str]:
        root = _schema_root(regime, schema)
        if not root.exists():
            return set()
        out: set[str] = set()
        for p in root.glob("week_ending=*"):
            if p.is_dir():
                out.add(p.name.split("=", 1)[1])
        return out

    return sorted(weeks_for(schema_a) & weeks_for(schema_b))


def load_scores(regime: str, schema: str, week_end: str) -> pd.DataFrame:
    p = _schema_root(regime, schema) / f"week_ending={week_end}" / "scores_weekly.parquet"
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_parquet(p)

    # Accept either your newer schema (UPS_adj/rank_UPS) or older (score/rank).
    if "symbol" not in df.columns:
        raise KeyError(f"{p} missing 'symbol' column. Have={list(df.columns)[:30]}")

    if "UPS_adj" in df.columns:
        score_col = "UPS_adj"
        rank_col = "rank_UPS" if "rank_UPS" in df.columns else None
    elif "score" in df.columns:
        score_col = "score"
        rank_col = "rank" if "rank" in df.columns else None
    else:
        raise KeyError(f"{p} missing score column (expected 'UPS_adj' or 'score'). Have={list(df.columns)[:30]}")

    out = df[["symbol", score_col]].rename(columns={score_col: "score"}).copy()

    # If rank isn't present, compute it deterministically.
    if rank_col and rank_col in df.columns:
        out["rank"] = df[rank_col]
    else:
        out["rank"] = out["score"].rank(ascending=False, method="min")

    out["rank"] = out["rank"].astype(int)
    return out


def compare_week(regime: str, week_end: str, a: str, b: str) -> pd.DataFrame:
    da = load_scores(regime, a, week_end).rename(columns={"score": "score_a", "rank": "rank_a"})
    db = load_scores(regime, b, week_end).rename(columns={"score": "score_b", "rank": "rank_b"})

    d = da.merge(db, on="symbol", how="inner")
    d["score_delta"] = d["score_b"] - d["score_a"]
    d["rank_delta"] = d["rank_b"] - d["rank_a"]
    d["week_end"] = week_end
    return d


def summarize(cmp_df: pd.DataFrame, top_n: int = 20) -> dict:
    top_a = set(cmp_df.nsmallest(top_n, "rank_a")['symbol'])
    top_b = set(cmp_df.nsmallest(top_n, "rank_b")['symbol'])
    overlap = len(top_a & top_b) / max(1, top_n)

    return {
        "week_end": cmp_df["week_end"].iloc[0],
        "n_common": int(len(cmp_df)),
        "topN": int(top_n),
        "topN_overlap_pct": round(100.0 * overlap, 2),
        "median_abs_rank_delta": float(cmp_df["rank_delta"].abs().median()),
        "max_abs_rank_delta": float(cmp_df["rank_delta"].abs().max()),
        "median_abs_score_delta": float(cmp_df["score_delta"].abs().median()),
        "max_abs_score_delta": float(cmp_df["score_delta"].abs().max()),
    }


def main() -> None:
    # Marker files (useful in CI)
    Path("/tmp/compare_imported.txt").write_text("imported\n")
    Path("/tmp/compare_started.txt").write_text("started\n")

    try:
        regime = "news-novelty-v1"
        schema_a = "news-novelty-v1"
        schema_b = "news-novelty-v1b"

        OUT_DIR.mkdir(parents=True, exist_ok=True)

        weeks = discover_common_weeks(regime, schema_a, schema_b)
        Path("/tmp/compare_weeks.txt").write_text("\n".join(weeks) + ("\n" if weeks else ""))

        all_rows: list[dict] = []
        for w in weeks:
            cmp_df = compare_week(regime, w, schema_a, schema_b)

            out_week = OUT_DIR / f"schema_compare_week_end={w}.parquet"
            cmp_df.to_parquet(out_week, index=False)

            all_rows.append(summarize(cmp_df, top_n=20))

        summary = (
            pd.DataFrame(all_rows).sort_values("week_end")
            if all_rows
            else pd.DataFrame(
                columns=[
                    "week_end",
                    "n_common",
                    "topN",
                    "topN_overlap_pct",
                    "median_abs_rank_delta",
                    "max_abs_rank_delta",
                    "median_abs_score_delta",
                    "max_abs_score_delta",
                ]
            )
        )

        out_csv = OUT_DIR / "schema_compare_summary.csv"
        summary.to_csv(out_csv, index=False)

        Path("/tmp/compare_done.txt").write_text(f"done; weeks={len(weeks)}; wrote={out_csv}\n")

    except Exception:
        Path("/tmp/compare_fatal.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
