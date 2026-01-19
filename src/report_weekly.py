# src/report_weekly.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Paths
# ----------------------------

@dataclass(frozen=True)
class Paths:
    root: Path
    derived: Path
    scores_dir: Path
    features_dir: Path
    clusters_dir: Path
    enriched_dir: Path
    reports_dir: Path

def default_paths() -> Paths:
    root = Path(__file__).resolve().parents[1]
    derived = root / "data" / "derived"
    return Paths(
        root=root,
        derived=derived,
        scores_dir=derived / "scores_weekly",
        features_dir=derived / "features_weekly",
        clusters_dir=derived / "news_clusters",
        enriched_dir=derived / "rep_enriched",
        reports_dir=derived / "reports",
    )

def load_week(paths: Paths, week_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores_p = paths.scores_dir / f"week_ending={week_end}" / "scores_weekly.parquet"
    feats_p  = paths.features_dir / f"week_ending={week_end}" / "features_weekly.parquet"
    clus_p   = paths.clusters_dir / f"week_ending={week_end}" / "clusters.parquet"
    enr_p    = paths.enriched_dir / f"week_ending={week_end}" / "rep_enriched.parquet"

    for p in [scores_p, feats_p, clus_p, enr_p]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    scores = pd.read_parquet(scores_p)
    feats = pd.read_parquet(feats_p)
    clusters = pd.read_parquet(clus_p)
    enriched = pd.read_parquet(enr_p)

    # Normalize symbol
    for df in [scores, feats, clusters, enriched]:
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    return scores, feats, clusters, enriched


# ----------------------------
# Formatting helpers
# ----------------------------

def safe_json_load(x) -> Dict[str, Any]:
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
    return f"{x*100:.2f}%"

def fmt_num(x: float, nd: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{nd}f}"

def trunc(s: str, n: int = 140) -> str:
    s = (s or "").strip().replace("\n", " ")
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"

def bucket_signal_state(ifs: float, ar5: float) -> str:
    """
    Simple v1 state:
      - Divergent: high IFS, low AR5
      - Overheated: low IFS, high AR5
      - Confirmed: both high-ish
      - Early: otherwise
    Uses z-scales in spirit; with sample may be flat.
    """
    if not np.isfinite(ifs) or not np.isfinite(ar5):
        return "Early"
    # Thresholds are mild because z-scores can be small early on
    if ifs > 0.35 and ar5 < 0.0:
        return "Divergent"
    if ifs < 0.0 and ar5 > 0.35:
        return "Overheated"
    if ifs > 0.2 and ar5 > 0.2:
        return "Confirmed"
    return "Early"

def driver_summary_row(row: pd.Series) -> str:
    """
    One short phrase used in the Top table.
    """
    pieces = []
    # Use EVS (event intensity) + SS (sent shift) + NS (novelty) as explanations
    if np.isfinite(row.get("EVS", np.nan)) and row["EVS"] > 0.6:
        pieces.append("Event intensity")
    if np.isfinite(row.get("SS", np.nan)) and row["SS"] > 0.6:
        pieces.append("Sentiment inflection")
    if np.isfinite(row.get("NS", np.nan)) and row["NS"] > 0.6:
        pieces.append("Novelty")
    if np.isfinite(row.get("MCS_up", np.nan)) and row["MCS_up"] > 0.6:
        pieces.append("Market follow-through")
    if not pieces:
        pieces.append("Mixed / low signal")
    # keep it short
    return ", ".join(pieces[:2])

def conviction_band(ups_adj: float) -> str:
    """
    Deterministic, conservative. Not a forecast—just signal strength.
    """
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
    One sentence, rule-based. Chooses the dominant driver.
    Priority:
      1) Strong market confirmation (MCS_up)
      2) Information-driven (IFS + EVS)
      3) Divergence (PD_up_raw)
      4) Default mixed
    """
    mcs = float(row.get("MCS_up", np.nan))
    ifs = float(row.get("IFS", np.nan))
    evs = float(row.get("EVS", np.nan))
    pd_up = float(row.get("PD_up_raw", np.nan))
    ar5 = float(row.get("AR5", np.nan))

    # Dominance checks (tuned to be conservative)
    if np.isfinite(mcs) and mcs >= 0.60 and (not np.isfinite(ifs) or ifs < 0.35):
        return "Ranked primarily due to recent price confirmation rather than new information."
    if np.isfinite(ifs) and np.isfinite(evs) and (ifs + evs) >= 0.80 and (not np.isfinite(mcs) or mcs < 0.45):
        return "Ranked due to information-driven news flow with limited price confirmation so far."
    if np.isfinite(pd_up) and pd_up >= 0.75:
        # Optional nuance: mention price direction if available
        if np.isfinite(ar5) and ar5 < 0:
            return "Ranked due to strong divergence: constructive news signals while recent price action remains weak."
        return "Ranked due to divergence between news signals and recent price action."
    return "Rank reflects a mix of weaker signals with no clear dominant driver."

def top_clusters_for_symbol(
    clusters: pd.DataFrame,
    enriched: pd.DataFrame,
    symbol: str,
    top_k: int = 3
) -> pd.DataFrame:
    """
    Return top_k clusters for a symbol, prioritized by event severity (sev_final proxy)
    and then by cluster_size.
    """
    c = clusters[clusters["symbol"] == symbol].copy()
    if c.empty:
        return c

    e = enriched[enriched["symbol"] == symbol].copy()
    if e.empty:
        c["sev_proxy"] = 0.0
        c["sent_proxy"] = 0.0
        c["evt_json"] = None
        c["sent_json"] = None
        return c.sort_values(["cluster_size"], ascending=False).head(top_k)

    # Join enrichment
    tmp = c.merge(e[["cluster_id", "sentiment_json", "event_json"]], on="cluster_id", how="left")
    tmp["sent"] = tmp["sentiment_json"].apply(safe_json_load)
    tmp["evt"] = tmp["event_json"].apply(safe_json_load)

    # Severity proxy = event_severity * event_confidence (consistent with EI inputs)
    tmp["sev_proxy"] = tmp["evt"].apply(lambda d: float(d.get("event_severity", 0.0) or 0.0) * float(d.get("event_confidence", 0.0) or 0.0))
    tmp["sent_proxy"] = tmp["sent"].apply(lambda d: float(d.get("sent_score", 0.0) or 0.0) * float(d.get("confidence", 0.0) or 0.0))

    # Prefer meaningful events, then larger clusters
    tmp = tmp.sort_values(["sev_proxy", "cluster_size"], ascending=False).head(top_k)

    # Bring out a few strings for rendering
    tmp["event_type_primary"] = tmp["evt"].apply(lambda d: d.get("event_type_primary", ""))
    tmp["event_rationale"] = tmp["evt"].apply(lambda d: d.get("rationale", ""))
    tmp["sent_label"] = tmp["sent"].apply(lambda d: d.get("sent_label", ""))
    tmp["sent_rationale"] = tmp["sent"].apply(lambda d: d.get("rationale", ""))
    return tmp


# ----------------------------
# Report generation
# ----------------------------

def build_report_markdown(
    week_end: str,
    scores: pd.DataFrame,
    feats: pd.DataFrame,
    clusters: pd.DataFrame,
    enriched: pd.DataFrame,
    top_n: int = 20,
    bottom_n: int = 20,
) -> str:
    # Merge scores + features to enable richer narrative
    df = scores.merge(feats, on=["symbol", "sector", "asof_date"], how="left", suffixes=("", "_f"))

    # Sort for top and bottom lists
    top = df.sort_values("UPS_adj", ascending=False).head(top_n).copy()
    bot = df.sort_values("DPS_adj", ascending=False).head(bottom_n).copy()

    # Diagnostics / signal quality
    # These are simple "trust builders" rather than over-optimized stats
    avg_novelty = float(np.nanmean(df.get("NS", pd.Series(dtype=float)).to_numpy())) if "NS" in df.columns else np.nan
    avg_evs = float(np.nanmean(df.get("EVS", pd.Series(dtype=float)).to_numpy())) if "EVS" in df.columns else np.nan
    avg_price_action = float(np.nanmean(df.get("price_action_rate", pd.Series(dtype=float)).to_numpy())) if "price_action_rate" in df.columns else np.nan

    # Calculate PRICE_ACTION_RECAP percentage from enriched data
    if not enriched.empty and "event_json" in enriched.columns:
        event_types = enriched["event_json"].apply(lambda x: safe_json_load(x).get("event_type_primary", "") if pd.notna(x) else "")
        recap_pct = 100.0 * (event_types == "PRICE_ACTION_RECAP").mean()
    else:
        recap_pct = 0.0

    # Sector concentration (top basket)
    sec_counts = top["sector"].value_counts(dropna=False)
    sec_summary = ", ".join([f"{k}: {v}" for k, v in sec_counts.items()])

    # Cover summary paragraph (kept calm)
    summary_lines = []
    summary_lines.append(f"# Weekly Market Pressure Report")
    summary_lines.append(f"**Week Ending:** {week_end}")
    summary_lines.append("")
    summary_lines.append(
        f"This report ranks S&P 500 names by **Upside Pressure (UPS)** using weekly news novelty, event intensity, sentiment shift, "
        f"and market confirmation. Use it as a **screen and context tool**, not as a standalone trading instruction."
    )
    summary_lines.append("")
    summary_lines.append("## Signal quality snapshot")
    summary_lines.append(f"- Avg novelty (z): {fmt_num(avg_novelty)}")
    summary_lines.append(f"- Avg event intensity (z): {fmt_num(avg_evs)}")
    summary_lines.append(f"- Avg price-action news rate: {fmt_num(avg_price_action, 2)}")
    summary_lines.append(f"- PRICE_ACTION_RECAP (% of clusters): {recap_pct:.0f}%")
    summary_lines.append(f"- Sector mix (Top {len(top)}): {sec_summary if sec_summary else '—'}")
    summary_lines.append("")

    # Top-20 table
    summary_lines.append(f"## Top {len(top)} Upside Pressure (UPS)")
    summary_lines.append("")
    summary_lines.append("| Rank | Ticker | Sector | UPS_adj | Conviction | Drivers | Rationale | Signal state |")
    summary_lines.append("|---:|:---|:---|---:|:---|:---|:---|:---|")

    for _, r in top.iterrows():
        rank = int(r.get("rank_UPS", 0)) if np.isfinite(r.get("rank_UPS", np.nan)) else ""
        sym = r["symbol"]
        sec = r.get("sector", "Unknown")
        ups_val = float(r.get("UPS_adj", np.nan))
        ups = fmt_num(ups_val, 3)
        conv = conviction_band(ups_val)
        drivers = driver_summary_row(r)
        rat = ranking_rationale(r)
        state = bucket_signal_state(float(r.get("IFS", np.nan)),
                                    float(r.get("z_AR5", np.nan)) if "z_AR5" in r.index else float(r.get("AR5", np.nan)))
        summary_lines.append(f"| {rank} | {sym} | {sec} | {ups} | {conv} | {drivers} | {rat} | {state} |")

    summary_lines.append("")

    # Stock cards
    summary_lines.append("## Stock cards (Top UPS)")
    summary_lines.append("")
    for _, r in top.iterrows():
        sym = r["symbol"]
        sec = r.get("sector", "Unknown")
        ups = float(r.get("UPS_adj", np.nan))
        ifs = float(r.get("IFS", np.nan))
        evs = float(r.get("EVS", np.nan))
        mcs = float(r.get("MCS_up", np.nan))
        ar5 = float(r.get("AR5", np.nan))
        vs = float(r.get("VS_raw", np.nan))
        vr = float(r.get("VR_pct", np.nan))
        state = bucket_signal_state(ifs, float(r.get("z_AR5", np.nan)) if "z_AR5" in r.index else ar5)

        # Pull top clusters for this symbol
        tc = top_clusters_for_symbol(clusters, enriched, sym, top_k=3)

        # Compose "what changed" bullets from event/sent rationales
        what_changed = []
        event_tags = []
        for _, cr in tc.iterrows():
            et = str(cr.get("event_type_primary", "") or "").strip()
            if et:
                event_tags.append(et)
            er = str(cr.get("event_rationale", "") or "").strip()
            sr = str(cr.get("sent_rationale", "") or "").strip()
            head = trunc(str(cr.get("rep_headline", "") or ""), 120)
            line = f"- **{head}**"
            if er:
                line += f" — {trunc(er, 120)}"
            elif sr:
                line += f" — {trunc(sr, 120)}"
            what_changed.append(line)

        event_tags = sorted(list(set([t for t in event_tags if t])))

        # "What would weaken" heuristic based on event types
        weaken = []
        if any(t in {"EARNINGS_GUIDANCE"} for t in event_tags):
            weaken.append("Follow-on guidance or margin commentary that contradicts the initial read")
        if any(t in {"REGULATORY_LEGAL"} for t in event_tags):
            weaken.append("New filings, adverse rulings, or regulatory actions that escalate scope")
        if any(t in {"CAPITAL_STRUCTURE"} for t in event_tags):
            weaken.append("Higher financing costs, dilution, or covenant stress")
        if any(t in {"OPERATIONS_SUPPLY"} for t in event_tags):
            weaken.append("Evidence the operational issue is broader or longer-lived than expected")
        if any(t in {"PRODUCT_MARKET"} for t in event_tags):
            weaken.append("Competitive response, contract repricing, or demand weakness")
        if not weaken:
            weaken.append("A reversal in news flow (lower novelty) combined with weak price confirmation")

        risks = []
        if np.isfinite(vr) and vr > 0.90:
            risks.append("High volatility regime (position-level risk elevated)")
        if np.isfinite(ar5) and abs(ar5) > 0.05:
            risks.append("Recent sharp move; risk of mean reversion")
        if np.isfinite(vs) and vs > 0.5:
            risks.append("Unusually high trading volume (attention risk / crowdedness)")
        if not risks:
            risks.append("Normal idiosyncratic and market risk")

        summary_lines.append(f"### {sym} — {sec}")
        conv = conviction_band(ups)
        rat = ranking_rationale(r)
        summary_lines.append(f"- **UPS_adj:** {fmt_num(ups, 3)} | **Conviction:** {conv} | **Signal state:** {state}")
        summary_lines.append(f"- Ranking rationale: {rat}")
        summary_lines.append(f"- Components: IFS {fmt_num(ifs)} · EVS {fmt_num(evs)} · MCS_up {fmt_num(mcs)}")
        summary_lines.append(f"- Market context: AR5 {fmt_pct(ar5)} · VS {fmt_num(vs)} · VR_pct {fmt_num(vr, 2)}")
        if event_tags:
            summary_lines.append(f"- Event tags: `{', '.join(event_tags[:6])}`")
        summary_lines.append("")
        summary_lines.append("**What changed this week**")
        if what_changed:
            summary_lines.extend(what_changed)
        else:
            summary_lines.append("- No representative clusters available (coverage gap).")
        summary_lines.append("")
        summary_lines.append("**Key risks**")
        for rr in risks[:3]:
            summary_lines.append(f"- {rr}")
        summary_lines.append("")
        summary_lines.append("**What would weaken this signal**")
        for ww in weaken[:3]:
            summary_lines.append(f"- {ww}")
        summary_lines.append("")

    # Bottom-20 DPS snapshot (lightweight)
    summary_lines.append(f"## Bottom {len(bot)} Downside Pressure (DPS) snapshot")
    summary_lines.append("")
    summary_lines.append("| Rank | Ticker | Sector | DPS_adj | Primary concern |")
    summary_lines.append("|---:|:---|:---|---:|:---|")
    for _, r in bot.iterrows():
        rank = int(r.get("rank_DPS", 0)) if np.isfinite(r.get("rank_DPS", np.nan)) else ""
        sym = r["symbol"]
        sec = r.get("sector", "Unknown")
        dps = fmt_num(float(r.get("DPS_adj", np.nan)), 3)

        concern = "Mixed / low signal"
        if np.isfinite(r.get("SS", np.nan)) and r["SS"] < -0.6:
            concern = "Sentiment deterioration"
        elif np.isfinite(r.get("EVS", np.nan)) and r["EVS"] < -0.6:
            concern = "Event intensity negative"
        elif np.isfinite(r.get("MCS_down", np.nan)) and r["MCS_down"] > 0.6:
            concern = "Market downside confirmation"

        summary_lines.append(f"| {rank} | {sym} | {sec} | {dps} | {concern} |")
    summary_lines.append("")

    # Boilerplate (stable)
    summary_lines.append("## What this is / isn't")
    summary_lines.append("")
    summary_lines.append("**What this report is**")
    summary_lines.append("- A systematic scan of weekly news-driven market pressure (UPS/DPS)")
    summary_lines.append("- A way to surface information flow shifts you may not have time to track manually")
    summary_lines.append("- A starting point for your own due diligence and risk decisions")
    summary_lines.append("")
    summary_lines.append("**What this report is not**")
    summary_lines.append("- Investment advice")
    summary_lines.append("- A prediction of price movements")
    summary_lines.append("- A substitute for portfolio-level risk management")
    summary_lines.append("")

    return "\n".join(summary_lines)


def run(week_end: str, top_n: int = 20, bottom_n: int = 20) -> Path:
    paths = default_paths()
    scores, feats, clusters, enriched = load_week(paths, week_end)

    md = build_report_markdown(
        week_end=week_end,
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
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--week_end", required=True, help="Week ending Friday (ET) YYYY-MM-DD")
    ap.add_argument("--top_n", type=int, default=20)
    ap.add_argument("--bottom_n", type=int, default=20)
    args = ap.parse_args()

    run(args.week_end, top_n=args.top_n, bottom_n=args.bottom_n)
