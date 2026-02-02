# src/report_weekly.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from . import config


# ----------------------------
# Paths + loading
# ----------------------------

@dataclass(frozen=True)
class Paths:
    root: Path
    derived: Path
    clusters_dir: Path
    enriched_dir: Path
    reports_dir: Path


def default_paths() -> Paths:
    root = Path(__file__).resolve().parents[1]
    derived = root / "data" / "derived"
    return Paths(
        root=root,
        derived=derived,
        clusters_dir=derived / "news_clusters",
        enriched_dir=derived / "rep_enriched",
        reports_dir=derived / "reports",
    )


def _pick_first_existing(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def load_week(
    paths: Paths,
    week_end: str,
    regime: str,
    schema: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads:
      - scores_weekly.parquet
      - features_weekly.parquet
      - clusters.parquet
      - rep_enriched.parquet

    Supports both current and legacy directory conventions.
    """

    # ---- scores ----
    scores_pref = (
        paths.derived
        / "scores_weekly"
        / f"regime={regime}"
        / f"schema={schema}"
        / f"week_ending={week_end}"
        / "scores_weekly.parquet"
    )
    scores_legacy = (
        paths.derived
        / "scores_weekly"
        / f"week_ending={week_end}"
        / "scores_weekly.parquet"
    )
    scores_p = _pick_first_existing(scores_pref, scores_legacy)

    # ---- features ----
    # Current writer: features_weekly/regime=<regime>/week_ending=<week_end>/features_weekly.parquet
    feats_pref = (
        paths.derived
        / "features_weekly"
        / f"regime={regime}"
        / f"week_ending={week_end}"
        / "features_weekly.parquet"
    )
    # Older variant: features_weekly/regime=<regime>/schema=<schema>/week_ending=<week_end>/features_weekly.parquet
    feats_regime_schema = (
        paths.derived
        / "features_weekly"
        / f"regime={regime}"
        / f"schema={schema}"
        / f"week_ending={week_end}"
        / "features_weekly.parquet"
    )
    feats_legacy = (
        paths.derived
        / "features_weekly"
        / f"week_ending={week_end}"
        / "features_weekly.parquet"
    )
    feats_p = _pick_first_existing(feats_pref, feats_regime_schema, feats_legacy)

    # ---- clusters / enriched ----
    clus_p = paths.clusters_dir / f"week_ending={week_end}" / "clusters.parquet"
    enr_p = paths.enriched_dir / f"week_ending={week_end}" / "rep_enriched.parquet"

    required = [
        ("scores", scores_p),
        ("features", feats_p),
        ("clusters", clus_p),
        ("enriched", enr_p),
    ]
    missing = [(label, p) for (label, p) in required if not p.exists()]

    if missing:
        lines = [f"Missing required files for week {week_end}:"]
        for label, p in missing:
            lines.append(f"  - {label}: {p}")
        lines.append("")
        lines.append("Attempted path resolution:")
        lines.append(f"  scores preferred       : {scores_pref}")
        lines.append(f"  scores legacy          : {scores_legacy}")
        lines.append(f"  features preferred     : {feats_pref}")
        lines.append(f"  features regime+schema : {feats_regime_schema}")
        lines.append(f"  features legacy        : {feats_legacy}")
        lines.append(f"  clusters               : {clus_p}")
        lines.append(f"  enriched               : {enr_p}")
        lines.append("")
        lines.append("This usually means the pipeline did not finish for this week, or directory conventions changed.")
        raise FileNotFoundError("\n".join(lines))

    scores = pd.read_parquet(scores_p)
    feats = pd.read_parquet(feats_p)
    clusters = pd.read_parquet(clus_p)
    enriched = pd.read_parquet(enr_p)

    # Normalize symbol
    for df in (scores, feats, clusters, enriched):
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    return scores, feats, clusters, enriched


# ----------------------------
# Formatting helpers
# ----------------------------

def safe_json_load(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            return {}
    return {}


def fmt_pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:.2f}%"


def fmt_num(x: float, nd: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{nd}f}"


def trunc(s: str, n: int = 140) -> str:
    s = (s or "").strip().replace("\n", " ")
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def conviction_band(ups_adj: float) -> str:
    if ups_adj is None or not np.isfinite(ups_adj):
        return "—"
    a = abs(float(ups_adj))
    if a < 0.25:
        return "Weak"
    if a < 0.75:
        return "Moderate"
    return "Strong"


def ranking_rationale(row: pd.Series) -> str:
    """
    Explain rank using factor contribution columns that are present in the merged df:
      - novelty_factor
      - event_intensity_factor
      - sentiment_factor
      - divergence_factor

    Falls back gracefully if any are missing.
    """
    factors = {
        "Novelty": float(row.get("novelty_factor", np.nan)),
        "Event intensity": float(row.get("event_intensity_factor", np.nan)),
        "Sentiment shift": float(row.get("sentiment_factor", np.nan)),
        "News/price divergence": float(row.get("divergence_factor", np.nan)),
    }

    # keep only finite values
    finite = {k: v for k, v in factors.items() if np.isfinite(v)}

    if not finite:
        return "Insufficient factor data to explain rank."

    # dominant = largest absolute contribution
    dom_name, dom_val = max(finite.items(), key=lambda kv: abs(kv[1]))

    # Provide a directional word based on sign
    direction = "positive" if dom_val >= 0 else "negative"

    # Add a little context from z-scores when available
    z_nv = row.get("z_NV_raw", np.nan)
    z_ei = row.get("z_EI_raw", np.nan)
    z_ss = row.get("z_SS_raw", np.nan)
    z_ar5 = row.get("z_AR5", np.nan)

    bits = []
    if np.isfinite(z_nv):
        bits.append(f"novelty z={fmt_num(float(z_nv), 2)}")
    if np.isfinite(z_ei):
        bits.append(f"event z={fmt_num(float(z_ei), 2)}")
    if np.isfinite(z_ss):
        bits.append(f"sentiment z={fmt_num(float(z_ss), 2)}")
    if np.isfinite(z_ar5):
        bits.append(f"AR5 z={fmt_num(float(z_ar5), 2)}")

    detail = (", ".join(bits)) if bits else "z-scores unavailable"

    return f"Driven primarily by **{dom_name}** ({direction}) with {detail}."



# ----------------------------
# Report generation
# ----------------------------

def build_report_markdown(
    week_end: str,
    regime: str,
    schema: str,
    scores: pd.DataFrame,
    feats: pd.DataFrame,
    clusters: pd.DataFrame,
    enriched: pd.DataFrame,
    top_n: int = 20,
    bottom_n: int = 20,
) -> Tuple[str, Dict[str, Any]]:
    # Merge scores + features for richer narrative
    join_keys = [k for k in ["symbol", "sector", "asof_date"] if k in scores.columns and k in feats.columns]
    if not join_keys:
        join_keys = ["symbol"] if "symbol" in scores.columns and "symbol" in feats.columns else []
    df = scores.merge(feats, on=join_keys, how="left", suffixes=("", "_f")) if join_keys else scores.copy()

    # Choose score column for ranking. Newer outputs may have UPS_adj; older ones have "score".
    score_candidates = ["UPS_adj", "score", "UPS_raw"]
    score_col = next((c for c in score_candidates if c in df.columns), None)
    if score_col is None:
        raise KeyError(
            "scores file is missing a usable score column. "
            f"Tried {score_candidates}. Available columns={list(df.columns)}"
        )

    top = df.sort_values(score_col, ascending=False).head(top_n).copy()

    # Bottom ranking: prefer DPS_adj if present; otherwise use the same score column (ascending).
    if "DPS_adj" in df.columns:
        bot = df.sort_values("DPS_adj", ascending=False).head(bottom_n).copy()
    else:
        bot = df.sort_values(score_col, ascending=True).head(bottom_n).copy()

    # PRICE_ACTION_RECAP percentage from enriched data (if available)
    recap_pct = 0.0
    if not enriched.empty and "event_json" in enriched.columns:
        event_types = enriched["event_json"].apply(
            lambda x: safe_json_load(x).get("event_type_primary", "") if pd.notna(x) else ""
        )
        recap_pct = float(100.0 * (event_types == "PRICE_ACTION_RECAP").mean())

        # Prefer z-scored columns (these are what our pipeline currently writes)
    if "z_NV_raw" in df.columns:
        avg_novelty = float(np.nanmean(df["z_NV_raw"]))
    elif "NS" in df.columns:
        avg_novelty = float(np.nanmean(df["NS"]))
    else:
        avg_novelty = np.nan

    if "z_EI_raw" in df.columns:
        avg_evs = float(np.nanmean(df["z_EI_raw"]))
    elif "EVS" in df.columns:
        avg_evs = float(np.nanmean(df["EVS"]))
    else:
        avg_evs = np.nan

    low_info_reasons = []
    if recap_pct >= 70:
        low_info_reasons.append(f"{recap_pct:.0f}% of clusters are PRICE_ACTION_RECAP")
    if np.isfinite(avg_novelty) and np.isfinite(avg_evs) and abs(avg_novelty) < 0.10 and abs(avg_evs) < 0.10:
        low_info_reasons.append("novelty and event intensity are near zero on average")

    is_low_information_week = bool(low_info_reasons)

    lines: list[str] = []
    lines.append("# Weekly Market Pressure Report")
    lines.append(f"**Week Ending:** {week_end}")
    lines.append(f"**Regime:** `{regime}`  |  **Schema:** `{schema}`")
    lines.append("")
    lines.append(
        "This report ranks S&P 500 names by **Upside Pressure (UPS)** using weekly news novelty, event intensity, "
        "sentiment shift, and market confirmation. Use it as a screen/context tool, not as standalone advice."
    )
    lines.append("")
    lines.append("## Signal quality snapshot")
    lines.append(f"- Avg novelty (z): {fmt_num(avg_novelty)}")
    lines.append(f"- Avg event intensity (z): {fmt_num(avg_evs)}")
    lines.append(f"- PRICE_ACTION_RECAP (% of clusters): {recap_pct:.0f}%")
    lines.append("")

    if is_low_information_week:
        lines.append("## ⚠️ Low-information week")
        lines.append(
            "The system is seeing mostly market wrap / price-move recap content rather than company-specific events. "
            "Treat rankings as low conviction this week."
        )
        for r in low_info_reasons:
            lines.append(f"- {r}")
        lines.append("")

    # Top table
    sector_col = "sector" if "sector" in top.columns else None
    lines.append(f"## Top {len(top)} Upside Pressure (UPS)")
    lines.append("")
    lines.append(f"| Rank | Ticker | Sector | {score_col} | z_NV | z_EI | z_SS | z_AR5 | Dominant driver | Conviction | Rationale |")
    lines.append("|---:|:---|:---|---:|---:|---:|---:|---:|:---|:---|:---|")

    for i, (_, r) in enumerate(top.iterrows(), start=1):
        sym = str(r.get("symbol", ""))
        sec = str(r.get(sector_col, "Unknown")) if sector_col else "Unknown"
        ups_val = float(r.get(score_col, np.nan))
        z_nv = float(r.get("z_NV_raw", np.nan))
        z_ei = float(r.get("z_EI_raw", np.nan))
        z_ss = float(r.get("z_SS_raw", np.nan))
        z_ar5 = float(r.get("z_AR5", np.nan))

        # Dominant driver from factor contributions
        contribs = {
            "Novelty": float(r.get("novelty_factor", np.nan)),
            "Event intensity": float(r.get("event_intensity_factor", np.nan)),
            "Sentiment shift": float(r.get("sentiment_factor", np.nan)),
            "News/price divergence": float(r.get("divergence_factor", np.nan)),
        }
        finite = {k: v for k, v in contribs.items() if np.isfinite(v)}
        dom = max(finite.items(), key=lambda kv: abs(kv[1]))[0] if finite else "—"

        lines.append(
            f"| {i} | {sym} | {sec} | {fmt_num(ups_val, 3)} | "
            f"{fmt_num(z_nv, 2)} | {fmt_num(z_ei, 2)} | {fmt_num(z_ss, 2)} | {fmt_num(z_ar5, 2)} | "
            f"{dom} | {conviction_band(ups_val)} | {ranking_rationale(r)} |"
        )

    lines.append("")

    # Bottom table (light)
    lines.append(f"## Bottom {len(bot)} Downside Pressure (DPS) snapshot")
    lines.append("")
    lines.append("| Rank | Ticker | Sector | DPS_adj |")
    lines.append("|---:|:---|:---|---:|")
    for i, (_, r) in enumerate(bot.iterrows(), start=1):
        sym = str(r.get("symbol", ""))
        sec = str(r.get(sector_col, "Unknown")) if sector_col else "Unknown"
        dps_val = float(r.get("DPS_adj", np.nan))
        lines.append(f"| {i} | {sym} | {sec} | {fmt_num(dps_val, 3)} |")
    lines.append("")

    # Build stamp
    git_sha = os.getenv("GIT_SHA", "")
    github_run_id = os.getenv("RUN_ID", "")
    github_run_attempt = os.getenv("RUN_ATTEMPT", "")
    run_trigger = os.getenv("RUN_TRIGGER", "")
    week_end_requested = os.getenv("WEEK_END_REQUESTED", "")
    run_started_utc = os.getenv("RUN_STARTED_UTC", "")
    python_ver = platform.python_version()

    features_scores_sha256 = ""
    try:
        features_scores_path = Path(__file__).parent / "features_scores.py"
        if features_scores_path.exists():
            features_scores_sha256 = hashlib.sha256(features_scores_path.read_bytes()).hexdigest()
    except Exception:
        pass

    lines.append("## Build Stamp")
    lines.append("")
    if git_sha:
        lines.append(f"- **Git SHA:** `{git_sha}`")
    if github_run_id:
        lines.append(f"- **GitHub Run:** `{github_run_id}` (attempt {github_run_attempt})")
    if run_trigger:
        lines.append(f"- **Trigger:** `{run_trigger}`")
    if week_end_requested:
        lines.append(f"- **Week_end requested:** `{week_end_requested}`")
    if run_started_utc:
        lines.append(f"- **Run started (UTC):** `{run_started_utc}`")
    if features_scores_sha256:
        lines.append(f"- **features_scores.py SHA256:** `{features_scores_sha256}`")
    lines.append(f"- **Python:** {python_ver}")
    lines.append(f"- **pandas:** {pd.__version__}")
    lines.append(f"- **numpy:** {np.__version__}")
    lines.append("")

    meta: Dict[str, Any] = {
        "week_ending_date": week_end,
        "regime": regime,
        "schema": schema,
        "is_low_information_week": is_low_information_week,
        "low_info_reasons": low_info_reasons,
        "recap_pct": recap_pct,
        "avg_novelty_z": float(avg_novelty) if np.isfinite(avg_novelty) else None,
        "avg_event_intensity_z": float(avg_evs) if np.isfinite(avg_evs) else None,
        "cluster_count": int(len(enriched)),
        "config_snapshot": config.get_config_snapshot(),
        "run_metadata": {
            "run_trigger": run_trigger or None,
            "week_end_requested": week_end_requested or None,
            "week_end_resolved": week_end,
            "run_started_utc": run_started_utc or None,
            "git_sha": git_sha or None,
            "github_run_id": github_run_id or None,
            "github_run_attempt": github_run_attempt or None,
        },
        "build": {
            "git_sha": git_sha or None,
            "github_run_id": github_run_id or None,
            "github_run_attempt": github_run_attempt or None,
            "python": python_ver,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "features_scores_sha256": features_scores_sha256 or None,
        },
    }

    return "\n".join(lines), meta


def run(
    week_end: str,
    regime: str = "news-novelty-v1",
    schema: str = "news-novelty-v1b",
    top_n: int = 20,
    bottom_n: int = 20
) -> Path:
    paths = default_paths()
    scores, feats, clusters, enriched = load_week(paths, week_end, regime, schema)

    md, meta = build_report_markdown(
        week_end=week_end,
        regime=regime,
        schema=schema,
        scores=scores,
        feats=feats,
        clusters=clusters,
        enriched=enriched,
        top_n=top_n,
        bottom_n=bottom_n,
    )

    out_dir = paths.reports_dir / f"week_ending={week_end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "weekly_report.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote: {out_path}")

    meta_path = out_dir / "report_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote: {meta_path}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week_end", required=True, help="Week ending Friday (ET) YYYY-MM-DD")
    ap.add_argument("--regime", default="news-novelty-v1", help="Regime ID (e.g., news-novelty-v1)")
    ap.add_argument("--schema", default="news-novelty-v1b", help="Schema ID (e.g., news-novelty-v1b)")
    ap.add_argument("--top_n", type=int, default=20)
    ap.add_argument("--bottom_n", type=int, default=20)
    args = ap.parse_args()

    run(
        args.week_end,
        regime=args.regime,
        schema=args.schema,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
    )


if __name__ == "__main__":
    main()
