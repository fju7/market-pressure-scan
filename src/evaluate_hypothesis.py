# src/evaluate_hypothesis.py
"""
Phase 2: Hypothesis Evaluation (artifact-only)

Reads already-produced Phase 1 artifacts (parquets + weekly outputs) and produces:
- Per-week evaluation markdown
- Aggregate summary CSV across weeks

No network calls. No recomputation of ingestion/enrichment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.scoring_schema import load_schema, ScoringSchema
from src.rescore_week import apply_scoring_schema, _apply_feature_mapping


# -------------------------
# Helpers: locate artifacts
# -------------------------

def _week_dir_name(week_end: str) -> str:
    return f"week_ending={week_end}"


def _find_first(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def locate_scores(output_base: Path, week_end: str, regime: str, schema_id: str) -> Path:
    candidates = [
        output_base / "scores_weekly" / f"regime={regime}" / f"schema={schema_id}" / _week_dir_name(week_end) / "scores_weekly.parquet",
        output_base / "scores_weekly" / f"regime={regime}" / _week_dir_name(week_end) / "scores_weekly.parquet",  # older regime-only
        output_base / "scores_weekly" / _week_dir_name(week_end) / "scores_weekly.parquet",  # legacy
    ]
    p = _find_first(candidates)
    if p is None:
        raise FileNotFoundError("Missing scores_weekly.parquet. Tried:\n" + "\n".join(str(x) for x in candidates))
    return p


def locate_features(output_base: Path, week_end: str, regime: str, schema_id: str) -> Path:
    candidates = [
        output_base / "features_weekly" / f"regime={regime}" / f"schema={schema_id}" / _week_dir_name(week_end) / "features_weekly.parquet",
        output_base / "features_weekly" / f"regime={regime}" / _week_dir_name(week_end) / "features_weekly.parquet",
        output_base / "features_weekly" / _week_dir_name(week_end) / "features_weekly.parquet",
    ]
    p = _find_first(candidates)
    if p is None:
        raise FileNotFoundError("Missing features_weekly.parquet. Tried:\n" + "\n".join(str(x) for x in candidates))
    return p


def locate_rep_enriched(output_base: Path, week_end: str) -> Optional[Path]:
    p = output_base / "rep_enriched" / _week_dir_name(week_end) / "rep_enriched.parquet"
    return p if p.exists() else None


def locate_weekly_report(output_base: Path, week_end: str) -> Optional[Path]:
    p = output_base / "reports" / _week_dir_name(week_end) / "weekly_report.md"
    return p if p.exists() else None


def locate_candles_daily(output_base: Path) -> Path:
    p = output_base / "market_daily" / "candles_daily.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing candles_daily.parquet at {p}")
    return p


# -------------------------
# Metrics: high-info weeks
# -------------------------

def _safe_json_load(x) -> dict:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


@dataclass
class HighInfoMetrics:
    n_clusters: int
    n_symbols: int
    price_action_recap_share: Optional[float]
    high_severity_cluster_count: Optional[int]
    severity_ge2_share: Optional[float]
    score_std: float
    score_iqr: float
    top5_abs_share: float


def compute_high_info_metrics(scores: pd.DataFrame, rep: Optional[pd.DataFrame]) -> HighInfoMetrics:
    # score dispersion
    score_col = next((c for c in ['score','UPS_adj','UPS_raw'] if c in scores.columns), None)
    if score_col is None:
        raise KeyError('scores is missing expected score column. Have columns=' + str(list(scores.columns)))
    s = pd.to_numeric(scores[score_col], errors='coerce')
    score_std = float(np.nanstd(s.to_numpy()))
    q75, q25 = np.nanpercentile(s.to_numpy(), [75, 25])
    score_iqr = float(q75 - q25)

    # concentration: sum(|top5|) / sum(|all|)
    abs_s = np.abs(s.to_numpy())
    denom = float(np.nansum(abs_s)) if np.isfinite(np.nansum(abs_s)) else 0.0
    top5_abs_share = 0.0
    if len(scores) > 0 and denom > 0:
        # Top-5 absolute share computed from the selected score series `s`
        top5 = s.dropna().sort_values(ascending=False).head(5).to_numpy()
        top5_abs_share = float(np.nansum(np.abs(top5)) / denom)

    price_action_recap_share = None
    high_severity_cluster_count = None
    severity_ge2_share = None

    if rep is not None and len(rep) > 0:
        ev = rep["event_json"].apply(_safe_json_load)
        ev_type = ev.apply(lambda d: d.get("event_type_primary"))
        ev_sev = ev.apply(lambda d: d.get("event_severity"))

        n = len(rep)
        price_action_recap_share = float((ev_type == "PRICE_ACTION_RECAP").sum() / n)

        sev_num = pd.to_numeric(ev_sev, errors="coerce")
        high_severity_cluster_count = int((sev_num >= 2).sum())
        severity_ge2_share = float((sev_num >= 2).sum() / n)

    return HighInfoMetrics(
        n_clusters=int(rep["cluster_id"].nunique()) if rep is not None and "cluster_id" in rep.columns else int(len(rep)) if rep is not None else 0,
        n_symbols=int(scores["symbol"].nunique()) if "symbol" in scores.columns else int(len(scores)),
        price_action_recap_share=price_action_recap_share,
        high_severity_cluster_count=high_severity_cluster_count,
        severity_ge2_share=severity_ge2_share,
        score_std=score_std,
        score_iqr=score_iqr,
        top5_abs_share=top5_abs_share,
    )


# -------------------------
# Returns: cross-sectional
# -------------------------

def _compute_forward_return(daily: pd.DataFrame, asof: pd.Timestamp, horizon_td: int) -> pd.Series:
    """
    Compute forward return over horizon_td trading days:
    ret = close[t+h] / close[t] - 1

    daily must contain rows for a SINGLE symbol sorted by date.
    """
    daily = daily.sort_values("date")
    # find first trading day >= asof
    idx = daily["date"].searchsorted(asof)
    if idx >= len(daily):
        return pd.Series({"ret": np.nan, "start_date": pd.NaT, "end_date": pd.NaT})
    start = daily.iloc[idx]
    end_idx = idx + horizon_td
    if end_idx >= len(daily):
        return pd.Series({"ret": np.nan, "start_date": start["date"], "end_date": pd.NaT})
    end = daily.iloc[end_idx]
    if not np.isfinite(start["close"]) or not np.isfinite(end["close"]) or start["close"] == 0:
        return pd.Series({"ret": np.nan, "start_date": start["date"], "end_date": end["date"]})
    return pd.Series({"ret": float(end["close"] / start["close"] - 1.0), "start_date": start["date"], "end_date": end["date"]})


@dataclass
class XSecReturnResult:
    horizon_td: int
    top_n: int
    bottom_n: int
    n_total: int
    n_covered: int
    top_mean: float
    mid_mean: float
    bottom_mean: float
    top_minus_bottom: float


def compute_cross_sectional_returns(
    scores: pd.DataFrame,
    candles_daily: pd.DataFrame,
    asof_date: pd.Timestamp,
    horizon_td: int,
    top_n: int,
    bottom_n: int,
) -> XSecReturnResult:
    # Choose the score column from available candidates; normalize to local column name 'score'
    score_col = next((c for c in ['UPS_adj','UPS_raw','score'] if c in scores.columns), None)
    if score_col is None:
        return XSecReturnResult(
            horizon_td=horizon_td,
            top_mean=None, mid_mean=None, bottom_mean=None, top_minus_bottom=None,
            n_covered=0, n_total=int(len(scores))
        )
    df = scores[['symbol', score_col]].copy()
    df = df.rename(columns={score_col: 'score'})
    df = df.dropna(subset=["symbol"])
    score_col = next((c for c in ["UPS_adj","UPS_raw","score"] if c in df.columns), None)
    if score_col is None:
        raise KeyError("scores missing expected score column for ranking. Have columns=" + str(list(df.columns)))
    df["score"] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.sort_values("score", ascending=False)

    n_total = len(df)
    top = df.head(top_n)
    bottom = df.tail(bottom_n)
    # mid bucket = everything else; if too small, take middle third-ish
    if n_total > (top_n + bottom_n):
        mid = df.iloc[top_n : n_total - bottom_n]
    else:
        mid = df.iloc[int(n_total/3) : int(2*n_total/3)]

    # compute returns per symbol
    daily = candles_daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")

    def sym_ret(sym: str) -> float:
        d = daily[daily["symbol"] == sym]
        if len(d) == 0:
            return np.nan
        ret = _compute_forward_return(d, asof_date, horizon_td).get("ret", np.nan)
        ret_num = pd.to_numeric(ret, errors="coerce")
        return float(ret_num) if pd.notna(ret_num) else np.nan

    df["fwd_ret"] = df["symbol"].map(sym_ret)

    n_covered = int(df["fwd_ret"].notna().sum())

    def mean_bucket(bucket: pd.DataFrame) -> float:
        vals = df[df["symbol"].isin(bucket["symbol"])]["fwd_ret"]
        return float(np.nanmean(vals.to_numpy())) if len(vals) and np.isfinite(vals.to_numpy()).any() else np.nan

    top_mean = mean_bucket(top)
    mid_mean = mean_bucket(mid)
    bottom_mean = mean_bucket(bottom)
    tmb = float(top_mean - bottom_mean) if np.isfinite(top_mean) and np.isfinite(bottom_mean) else np.nan

    return XSecReturnResult(
        horizon_td=horizon_td,
        top_n=top_n,
        bottom_n=bottom_n,
        n_total=n_total,
        n_covered=n_covered,
        top_mean=top_mean,
        mid_mean=mid_mean,
        bottom_mean=bottom_mean,
        top_minus_bottom=tmb,
    )


# -------------------------
# Ablation: rescore weights
# -------------------------

@dataclass
class AblationResult:
    ablation: str
    spearman_rank_corr: float
    top_overlap_frac: float


def _spearman_rank_corr(a: pd.Series, b: pd.Series) -> float:
    """
    Robust Spearman rank correlation.

    Returns np.nan (no warnings) when correlation is undefined:
      - insufficient overlap
      - all-NaN after alignment
      - zero variance (constant ranks) on either side
    """
    # Align on common index and drop NaNs
    df = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(df) < 2:
        return np.nan

    # Spearman = Pearson correlation of ranks
    ra = df["a"].rank(method="average")
    rb = df["b"].rank(method="average")

    # Guard zero-variance cases (constant ranks)
    sda = float(np.nanstd(ra.to_numpy()))
    sdb = float(np.nanstd(rb.to_numpy()))
    if not np.isfinite(sda) or not np.isfinite(sdb) or sda == 0.0 or sdb == 0.0:
        return np.nan

    rho = float(np.corrcoef(ra.to_numpy(), rb.to_numpy())[0, 1])
    return rho if np.isfinite(rho) else np.nan
def run_ablations(
    features: pd.DataFrame,
    schema: ScoringSchema,
    top_n: int,
) -> Tuple[pd.DataFrame, List[AblationResult]]:
    """
    Returns baseline scored df, plus ablation summary.
    """
    # feature mapping adapter (handles legacy columns)
    mapping_res = _apply_feature_mapping(features)
    feat = mapping_res.df

    baseline = apply_scoring_schema(feat, schema).copy()
    baseline = baseline.sort_values("score", ascending=False)
    base_top = set(baseline.head(top_n)["symbol"].tolist())

    ablations: List[AblationResult] = []
    weight_keys = ["novelty", "event_intensity", "sentiment", "divergence"]

    for k in weight_keys:
        raw2 = dict(schema.raw)
        weights2 = dict(schema.get_weights())
        weights2[k] = 0.0
        raw2["weights"] = weights2

        schema2 = ScoringSchema(schema_id=f"{schema.schema_id}__abl_{k}=0", raw=raw2, content_hash="(in-mem)")
        scored2 = apply_scoring_schema(feat, schema2).copy().sort_values("score", ascending=False)

        # align by symbol for rank corr
        base_col = next((c for c in ["UPS_adj","UPS_raw","score"] if c in baseline.columns), None)
        alt_col  = next((c for c in ["UPS_adj","UPS_raw","score"] if c in scored2.columns), None)
        if base_col is None or alt_col is None:
            raise KeyError("Missing expected score column in ablation frames. baseline cols=" + str(list(baseline.columns)) + " scored2 cols=" + str(list(scored2.columns)))
        base_scores = pd.to_numeric(baseline.set_index("symbol")[base_col], errors="coerce")
        alt_scores  = pd.to_numeric(scored2.set_index("symbol")[alt_col], errors="coerce")
        common = base_scores.index.intersection(alt_scores.index)
        if len(common) >= 3:
            rho = _spearman_rank_corr(base_scores.loc[common], alt_scores.loc[common])
        else:
            rho = np.nan

        alt_top = set(scored2.head(top_n)["symbol"].tolist())
        overlap = len(base_top.intersection(alt_top)) / float(top_n) if top_n > 0 else np.nan

        ablations.append(AblationResult(ablation=f"{k}=0", spearman_rank_corr=float(rho), top_overlap_frac=float(overlap)))

    return baseline, ablations


# -------------------------
# Reporting
# -------------------------

def _fmt_pct(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{100.0*x:.1f}%"


def _fmt_float(x: Optional[float], digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def write_week_report(
    out_dir: Path,
    week_end: str,
    regime: str,
    schema_id: str,
    hi: HighInfoMetrics,
    xsecs: List[XSecReturnResult],
    abls: List[AblationResult],
    scores_path: Path,
    features_path: Path,
    rep_path: Optional[Path],
    report_path: Optional[Path],
    returns_note: Optional[str] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "evaluation.md"

    lines: List[str] = []
    lines.append(f"# Hypothesis Evaluation — week ending {week_end}")
    lines.append("")
    lines.append(f"- regime: `{regime}`")
    lines.append(f"- schema: `{schema_id}`")
    lines.append(f"- scores: `{scores_path}`")
    lines.append(f"- features: `{features_path}`")
    lines.append(f"- rep_enriched: `{rep_path}`" if rep_path else "- rep_enriched: (missing)")
    lines.append(f"- weekly_report: `{report_path}`" if report_path else "- weekly_report: (missing)")
    lines.append("")

    lines.append("## High-information diagnostics")
    lines.append("")
    lines.append(f"- clusters: **{hi.n_clusters}**")
    lines.append(f"- symbols scored: **{hi.n_symbols}**")
    lines.append(f"- PRICE_ACTION_RECAP share (clusters): **{_fmt_pct(hi.price_action_recap_share)}**")
    lines.append(f"- severity ≥2 clusters: **{hi.high_severity_cluster_count if hi.high_severity_cluster_count is not None else 'n/a'}**")
    lines.append(f"- severity ≥2 share: **{_fmt_pct(hi.severity_ge2_share)}**")
    lines.append(f"- score std: **{_fmt_float(hi.score_std)}**")
    lines.append(f"- score IQR: **{_fmt_float(hi.score_iqr)}**")
    lines.append(f"- top5 |score| share: **{_fmt_pct(hi.top5_abs_share)}**")
    lines.append("")

    lines.append("## Cross-sectional forward returns (equal-weight)")
    lines.append("")
    if returns_note:
        lines.append(f"> ⚠️ {returns_note}")
        lines.append("")
    # Note when some horizons are not fully covered yet (expected for recent weeks)
    try:
        partial = [
            r for r in xsecs
            if r.n_total and r.n_covered is not None and r.n_covered < r.n_total
        ]
        if partial:
            hs = ", ".join(str(r.horizon_td) for r in partial)
            lines.append(
                f"> ⏳ Some horizons are not fully covered yet (horizons: {hs} TD). "
                "This is expected for recent weeks and will auto-update as new candles are ingested."
            )
            lines.append("")
    except Exception:
        pass

    lines.append("| horizon (trading days) | top mean | mid mean | bottom mean | top-bottom | covered / total |")
    lines.append("|---:|---:|---:|---:|---:|---:|")

    for r in xsecs:
        lines.append(
            f"| {r.horizon_td} | {_fmt_pct(r.top_mean)} | {_fmt_pct(r.mid_mean)} | {_fmt_pct(r.bottom_mean)} | {_fmt_pct(r.top_minus_bottom)} | {r.n_covered}/{r.n_total} |"
        )
    lines.append("")

    lines.append("## Factor ablation (rescore from saved features)")
    lines.append("")
    lines.append("| ablation | Spearman(rank) vs baseline | top-N overlap |")
    lines.append("|---|---:|---:|")
    for a in abls:
        lines.append(f"| {a.ablation} | {_fmt_float(a.spearman_rank_corr)} | {_fmt_pct(a.top_overlap_frac)} |")
    lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


# -------------------------
# Main driver
# -------------------------

def _parse_weeks_from_reports(output_base: Path) -> List[str]:
    reports_dir = output_base / "reports"
    if not reports_dir.exists():
        return []
    weeks = []
    for d in sorted(reports_dir.glob("week_ending=*")):
        if d.is_dir():
            weeks.append(d.name.split("week_ending=")[-1])
    return weeks


def evaluate_week(
    week_end: str,
    *,
    output_base: Path,
    regime: str,
    schema_id: str,
    candles_daily: pd.DataFrame,
    top_n: int,
    bottom_n: int,
    horizons: List[int],
) -> dict:
    scores_path = locate_scores(output_base, week_end, regime, schema_id)
    features_path = locate_features(output_base, week_end, regime, schema_id)
    rep_path = locate_rep_enriched(output_base, week_end)
    report_path = locate_weekly_report(output_base, week_end)

    scores = pd.read_parquet(scores_path)
    features = pd.read_parquet(features_path)
    rep = pd.read_parquet(rep_path) if rep_path else None

    # asof_date
    if "asof_date" in scores.columns and scores["asof_date"].notna().any():
        asof_date = pd.to_datetime(scores["asof_date"].iloc[0])
    else:
        asof_date = pd.to_datetime(week_end)

    hi = compute_high_info_metrics(scores, rep)

    # cross-sectional returns (guard against stale candles)
    returns_note: Optional[str] = None
    candles_dt = candles_daily.copy()
    candles_dt["date"] = pd.to_datetime(candles_dt["date"], errors="coerce")
    max_candle_date = candles_dt["date"].max()

    xsecs: List[XSecReturnResult] = []
    if pd.isna(max_candle_date) or max_candle_date < asof_date:
        returns_note = (
            "Forward returns not computed: candles_daily max date is "
            f"{max_candle_date.date() if pd.notna(max_candle_date) else 'n/a'} "
            f"but asof_date is {asof_date.date()}. "
            "Update market_daily/candles_daily to include dates on/after asof_date."
        )
        n_total = int(scores["symbol"].notna().sum()) if "symbol" in scores.columns else len(scores)
        for h in horizons:
            xsecs.append(
                XSecReturnResult(
                    horizon_td=h,
                    top_n=top_n,
                    bottom_n=bottom_n,
                    n_total=n_total,
                    n_covered=0,
                    top_mean=np.nan,
                    mid_mean=np.nan,
                    bottom_mean=np.nan,
                    top_minus_bottom=np.nan,
                )
            )
    else:
        for h in horizons:
            xsecs.append(
                compute_cross_sectional_returns(
                    scores=scores,
                    candles_daily=candles_daily,
                    asof_date=asof_date,
                    horizon_td=h,
                    top_n=top_n,
                    bottom_n=bottom_n,
                )
            )
    # ablations
    schema = load_schema(schema_id)
    _, abls = run_ablations(features, schema, top_n=top_n)

    # write per-week report
    out_dir = output_base / "analysis" / "hypothesis_eval" / _week_dir_name(week_end)
    md_path = write_week_report(
        out_dir=out_dir,
        week_end=week_end,
        regime=regime,
        schema_id=schema_id,
        hi=hi,
        xsecs=xsecs,
        abls=abls,
        scores_path=scores_path,
        features_path=features_path,
        rep_path=rep_path,
        report_path=report_path,
        returns_note=returns_note,
    )

    return {
        "week_end": week_end,
        "scores_path": str(scores_path),
        "features_path": str(features_path),
        "rep_path": str(rep_path) if rep_path else None,
        "asof_date": asof_date.isoformat(),
        "price_action_recap_share": hi.price_action_recap_share,
        "high_severity_cluster_count": hi.high_severity_cluster_count,
        "score_std": hi.score_std,
        "score_iqr": hi.score_iqr,
        "top5_abs_share": hi.top5_abs_share,
        **{f"top_minus_bottom_td{r.horizon_td}": r.top_minus_bottom for r in xsecs},
        "evaluation_md": str(md_path),
    }


def main():
    ap = argparse.ArgumentParser(description="Phase 2 hypothesis evaluation (artifact-only)")
    ap.add_argument("--output_base", default="data/derived", help="Base directory for derived artifacts")
    ap.add_argument("--regime", default="news-novelty-v1", help="Regime id (used for locating artifacts)")
    ap.add_argument("--schema", required=True, help="Scoring schema id (e.g., news-novelty-v1b)")
    ap.add_argument("--weeks", default="", help="Comma-separated week_end dates; if empty, use data/derived/reports")
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--bottom_n", type=int, default=10)
    ap.add_argument("--horizons", default="5,10,20", help="Trading-day horizons (comma-separated)")
    args = ap.parse_args()

    output_base = Path(args.output_base)
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]

    if args.weeks.strip():
        weeks = [w.strip() for w in args.weeks.split(",") if w.strip()]
    else:
        weeks = _parse_weeks_from_reports(output_base)

    if not weeks:
        raise SystemExit("No weeks found. Provide --weeks or ensure data/derived/reports/week_ending=* exists.")

    candles_path = locate_candles_daily(output_base)
    candles_daily = pd.read_parquet(candles_path)

    rows: List[dict] = []
    for w in weeks:
        print(f"=== evaluate {w} ===")
        rows.append(
            evaluate_week(
                w,
                output_base=output_base,
                regime=args.regime,
                schema_id=args.schema,
                candles_daily=candles_daily,
                top_n=args.top_n,
                bottom_n=args.bottom_n,
                horizons=horizons,
            )
        )

    out_root = output_base / "analysis" / "hypothesis_eval"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\n✅ Wrote summary: {summary_path}")
   
if __name__ == "__main__":
    main()
