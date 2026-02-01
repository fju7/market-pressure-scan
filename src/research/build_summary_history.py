from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _safe_read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _pick_score_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["UPS_adj", "score", "UPS", "UPS_raw"]:
        if c in df.columns:
            return c
    return None


def _first_present(d: Dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return None


def _compute_basket_size(scores: pd.DataFrame) -> int:
    for c in ["in_basket", "selected", "is_selected", "picked"]:
        if c in scores.columns:
            s = pd.to_numeric(scores[c], errors="coerce").fillna(0).astype(int)
            return int(s.sum())
    return int(len(scores))


def _mean_if_present(df: pd.DataFrame, col_candidates: list[str]) -> Optional[float]:
    for c in col_candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            return float(s.mean()) if s.notna().any() else None
    return None


def _spearman(scores_now: pd.DataFrame, scores_prev: pd.DataFrame, score_col: str) -> Optional[float]:
    id_col = None
    for c in ["symbol", "ticker", "asset", "secid"]:
        if c in scores_now.columns and c in scores_prev.columns:
            id_col = c
            break
    if id_col is None:
        return None

    a = scores_now[[id_col, score_col]].copy()
    b = scores_prev[[id_col, score_col]].copy()

    a[score_col] = pd.to_numeric(a[score_col], errors="coerce")
    b[score_col] = pd.to_numeric(b[score_col], errors="coerce")

    a = a.dropna()
    b = b.dropna()
    if len(a) == 0 or len(b) == 0:
        return None

    merged = a.merge(b, on=id_col, suffixes=("_now", "_prev"))
    if len(merged) < 3:
        return None

    # Spearman = Pearson correlation of ranks (no SciPy dependency)
    x = merged[f"{score_col}_now"].rank(method="average")
    y = merged[f"{score_col}_prev"].rank(method="average")
    return float(x.corr(y))
def _topn_jaccard(scores_now: pd.DataFrame, scores_prev: pd.DataFrame, score_col: str, n: int) -> Optional[float]:
    id_col = None
    for c in ["symbol", "ticker", "asset", "secid"]:
        if c in scores_now.columns and c in scores_prev.columns:
            id_col = c
            break
    if id_col is None:
        return None

    now = scores_now[[id_col, score_col]].copy()
    prev = scores_prev[[id_col, score_col]].copy()

    now[score_col] = pd.to_numeric(now[score_col], errors="coerce")
    prev[score_col] = pd.to_numeric(prev[score_col], errors="coerce")

    now = now.dropna().sort_values(score_col, ascending=False).head(n)
    prev = prev.dropna().sort_values(score_col, ascending=False).head(n)

    if len(now) == 0 or len(prev) == 0:
        return None

    set_now = set(now[id_col].astype(str))
    set_prev = set(prev[id_col].astype(str))
    inter = set_now.intersection(set_prev)
    union = set_now.union(set_prev)
    return float(len(inter)) / float(len(union)) if union else None


def build_rows(regime: str, schema: str, topn: int) -> pd.DataFrame:
    reports_dir = Path("data/derived/reports")
    scores_root = Path(f"data/derived/scores_weekly/regime={regime}/schema={schema}")
    feats_root = Path(f"data/derived/features_weekly/regime={regime}/schema={schema}")

    if not reports_dir.exists():
        raise SystemExit(f"Missing: {reports_dir}")
    if not scores_root.exists():
        raise SystemExit(f"Missing: {scores_root}")
    if not feats_root.exists():
        raise SystemExit(f"Missing: {feats_root}")

    weeks = sorted([p.name.split("week_ending=")[-1] for p in reports_dir.glob("week_ending=*")])

    rows = []
    prev_scores_df: Optional[pd.DataFrame] = None
    prev_score_col: Optional[str] = None

    for w in weeks:
        report_dir = reports_dir / f"week_ending={w}"
        report_meta = report_dir / "report_meta.json"
        report_md = report_dir / "weekly_report.md"

        scores_p = scores_root / f"week_ending={w}" / "scores_weekly.parquet"
        feats_p = feats_root / f"week_ending={w}" / "features_weekly.parquet"

        if not (report_meta.exists() and report_md.exists() and scores_p.exists() and feats_p.exists()):
            continue

        meta = _safe_read_json(report_meta)

        scores = pd.read_parquet(scores_p)
        feats = pd.read_parquet(feats_p)

        score_col = _pick_score_col(scores)

        week_type = _first_present(meta, ["week_type", "type", "run_type", "status"])
        is_skip = _first_present(meta, ["is_skip", "skip", "skipped"])
        recap_pct = _first_present(meta, ["recap_pct", "recap_percent", "pct_recap", "price_action_recap_pct"])

        basket_size = _compute_basket_size(scores)
        avg_novelty = _mean_if_present(feats, ["novelty", "novelty_mean", "avg_novelty"])
        avg_event_intensity = _mean_if_present(feats, ["event_intensity", "event_intensity_mean", "avg_event_intensity"])

        score_mean = score_std = score_p90 = None
        if score_col:
            s = pd.to_numeric(scores[score_col], errors="coerce")
            if s.notna().any():
                score_mean = float(s.mean())
                score_std = float(s.std())
                score_p90 = float(s.quantile(0.90))

        spearman_r = None
        topn_j = None
        if prev_scores_df is not None and score_col is not None:
            prev_col = prev_score_col if (prev_score_col in prev_scores_df.columns) else _pick_score_col(prev_scores_df)
            if prev_col and prev_col == score_col:
                spearman_r = _spearman(scores, prev_scores_df, score_col)
                topn_j = _topn_jaccard(scores, prev_scores_df, score_col, topn)

        rows.append(
            {
                "week_end": w,
                "regime": regime,
                "schema": schema,
                "week_type": week_type,
                "is_skip": is_skip,
                "recap_pct": recap_pct,
                "basket_size": basket_size,
                "score_col": score_col,
                "score_mean": score_mean,
                "score_std": score_std,
                "score_p90": score_p90,
                "avg_novelty": avg_novelty,
                "avg_event_intensity": avg_event_intensity,
                "spearman_r_vs_prev": spearman_r,
                f"top{topn}_jaccard_vs_prev": topn_j,
            }
        )

        prev_scores_df = scores
        prev_score_col = score_col

    return pd.DataFrame(rows).sort_values(["week_end", "regime", "schema"])


def upsert_csv(df_new: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    key = ["week_end", "regime", "schema"]

    if out_csv.exists():
        df_old = pd.read_csv(out_csv)

        for c in df_new.columns:
            if c not in df_old.columns:
                df_old[c] = None
        for c in df_old.columns:
            if c not in df_new.columns:
                df_new[c] = None

        df_old = df_old[df_new.columns]

        new_keys = set(tuple(x) for x in df_new[key].astype(str).values.tolist())
        df_old_kept = df_old[~df_old[key].astype(str).apply(tuple, axis=1).isin(new_keys)]

        df_out = pd.concat([df_old_kept, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out = df_out.sort_values(key)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    return df_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="news-novelty-v1")
    ap.add_argument("--schema", default="news-novelty-v1b")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--out", default="data/derived/analysis/summary_history.csv")
    args = ap.parse_args()

    df_new = build_rows(args.regime, args.schema, args.topn)
    out_csv = Path(args.out)
    df_out = upsert_csv(df_new, out_csv)

    print(f"Wrote: {out_csv} ({len(df_out)} rows)")
    print(df_out.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
