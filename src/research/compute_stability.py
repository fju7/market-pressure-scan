from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


ID_CANDIDATES = ["symbol", "ticker", "asset", "secid"]
SCORE_PREF = ["UPS_adj", "score", "UPS", "UPS_raw"]


def pick_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c
    return None


def pick_score_col(df: pd.DataFrame) -> Optional[str]:
    for c in SCORE_PREF:
        if c in df.columns:
            return c
    return None


def load_scores(scores_root: Path, week_end: str) -> Tuple[pd.DataFrame, str, str]:
    p = scores_root / f"week_ending={week_end}" / "scores_weekly.parquet"
    if not p.exists():
        raise FileNotFoundError(str(p))

    df = pd.read_parquet(p)
    id_col = pick_id_col(df)
    score_col = pick_score_col(df)

    if id_col is None:
        raise ValueError(f"No id column found in {p}")
    if score_col is None:
        raise ValueError(f"No score column found in {p}")

    out = df[[id_col, score_col]].copy()
    out[id_col] = out[id_col].astype(str)
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    out = out.dropna()

    return out, id_col, score_col


def spearman_rank_corr(a: pd.Series, b: pd.Series) -> Optional[float]:
    if len(a) < 3:
        return None
    ra = a.rank(method="average")
    rb = b.rank(method="average")
    v = ra.corr(rb)
    return None if pd.isna(v) else float(v)


def topn_jaccard(a_ranked: pd.DataFrame, b_ranked: pd.DataFrame, a_id: str, b_id: str, n: int) -> Optional[float]:
    if len(a_ranked) == 0 or len(b_ranked) == 0:
        return None
    sa = set(a_ranked[a_id].head(n))
    sb = set(b_ranked[b_id].head(n))
    if not sa and not sb:
        return None
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="news-novelty-v1")
    ap.add_argument("--schema", default="news-novelty-v1b")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--out", default="data/derived/analysis/stability_history.csv")
    args = ap.parse_args()

    reports_dir = Path("data/derived/reports")
    scores_root = Path(f"data/derived/scores_weekly/regime={args.regime}/schema={args.schema}")

    weeks = sorted(p.name.split("week_ending=")[-1] for p in reports_dir.glob("week_ending=*"))
    if len(weeks) < 2:
        raise SystemExit("Need at least 2 weeks")

    rows = []

    prev_week = weeks[0]
    prev_df, prev_id, prev_score = load_scores(scores_root, prev_week)

    for w in weeks[1:]:
        cur_df, cur_id, cur_score = load_scores(scores_root, w)

        merged = prev_df.merge(
            cur_df,
            left_on=prev_id,
            right_on=cur_id,
            suffixes=("_prev", "_cur"),
        )

        spearman = None
        if len(merged) >= 3:
            prev_col = f"{prev_score}_prev" if f"{prev_score}_prev" in merged.columns else prev_score
            cur_col  = f"{cur_score}_cur"  if f"{cur_score}_cur"  in merged.columns else cur_score
            spearman = spearman_rank_corr(merged[prev_col], merged[cur_col])

        prev_ranked = prev_df.sort_values(prev_score, ascending=False)
        cur_ranked = cur_df.sort_values(cur_score, ascending=False)
        jacc = topn_jaccard(prev_ranked, cur_ranked, prev_id, cur_id, args.topn)

        rows.append(
            {
                "week_prev": prev_week,
                "week_cur": w,
                "regime": args.regime,
                "schema": args.schema,
                "prev_score_col": prev_score,
                "cur_score_col": cur_score,
                "n_prev": len(prev_df),
                "n_cur": len(cur_df),
                "n_intersection": len(merged),
                "spearman_rank_corr": spearman,
                f"top{args.topn}_jaccard": jacc,
            }
        )

        prev_week = w
        prev_df, prev_id, prev_score = cur_df, cur_id, cur_score

    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Wrote: {out_path} ({len(out)} transitions)")
    s = pd.to_numeric(out["spearman_rank_corr"], errors="coerce")
    j = pd.to_numeric(out[f"top{args.topn}_jaccard"], errors="coerce")

    print("SUMMARY")
    print(f"  transitions: {len(out)}")
    print(f"  spearman: mean={s.mean():.4f} median={s.median():.4f} (non-null={s.notna().sum()})")
    print(f"  top{args.topn} jaccard: mean={j.mean():.4f} median={j.median():.4f} (non-null={j.notna().sum()})")

    small = out[(out["n_prev"] < 50) | (out["n_cur"] < 50)]
    if len(small):
        print("\\nANOMALY: small weeks detected")
        print(small[["week_prev", "week_cur", "n_prev", "n_cur"]].to_string(index=False))


if __name__ == "__main__":
    main()
