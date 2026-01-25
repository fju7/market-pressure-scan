# src/diagnostics.py
"""
Diagnostic tools: signal absence vs model blindness detection.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

@dataclass
class CoverageDiagnostics:
    """Coverage and composition diagnostics."""
    n_symbols_with_news: int
    n_clusters: int
    cluster_size_distribution: dict
    event_type_distribution: dict
    high_confidence_share: float
    price_action_recap_share: float

def compute_coverage_diagnostics(
    rep_enriched: pd.DataFrame,
    clusters: Optional[pd.DataFrame] = None,
) -> CoverageDiagnostics:
    """
    Compute coverage and composition metrics.
    Answers: "How much signal is available this week?"
    """
    n_symbols = rep_enriched["symbol"].nunique() if "symbol" in rep_enriched.columns else 0
    n_clusters = len(clusters) if clusters is not None and not clusters.empty else 0
    
    # Cluster size distribution
    if clusters is not None and "cluster_id" in clusters.columns:
        sizes = clusters.groupby("cluster_id").size()
        cluster_dist = {
            "min": int(sizes.min()) if len(sizes) > 0 else 0,
            "median": int(sizes.median()) if len(sizes) > 0 else 0,
            "max": int(sizes.max()) if len(sizes) > 0 else 0,
            "mean": float(sizes.mean()) if len(sizes) > 0 else 0.0,
        }
    else:
        cluster_dist = {"min": 0, "median": 0, "max": 0, "mean": 0.0}
    
    # Event type distribution
    event_dist = {}
    price_action_share = 0.0
    if "event_json" in rep_enriched.columns:
        events = []
        for e in rep_enriched["event_json"].dropna():
            if isinstance(e, str):
                import json
                e = json.loads(e)
            events.append(e.get("event_type_primary", "UNKNOWN"))
        
        if events:
            event_counts = pd.Series(events).value_counts()
            event_dist = event_counts.to_dict()
            price_action_share = event_counts.get("PRICE_ACTION_RECAP", 0) / len(events)
    
    # High confidence share
    high_conf_share = 0.0
    if "event_json" in rep_enriched.columns:
        confs = []
        for e in rep_enriched["event_json"].dropna():
            if isinstance(e, str):
                import json
                e = json.loads(e)
            confs.append(e.get("event_confidence", 0.0))
        
        if confs:
            high_conf_share = sum(1 for c in confs if c >= 0.7) / len(confs)
    
    return CoverageDiagnostics(
        n_symbols_with_news=n_symbols,
        n_clusters=n_clusters,
        cluster_size_distribution=cluster_dist,
        event_type_distribution=event_dist,
        high_confidence_share=high_conf_share,
        price_action_recap_share=price_action_share,
    )

def compute_counterfactual_scores(
    features: pd.DataFrame,
    baseline_filters: dict,
) -> dict:
    """
    Compute scores with looser filters to detect model blindness.
    
    If counterfactuals produce candidates but baseline doesn't,
    odds are "model blindness / over-filtering" not "signal absent".
    """
    # Baseline (with filters)
    df_baseline = features.copy()
    exclude_types = baseline_filters.get("exclude_event_types", [])
    if exclude_types and "event_type_primary" in df_baseline.columns:
        df_baseline = df_baseline[~df_baseline["event_type_primary"].isin(exclude_types)]
    
    baseline_topk = df_baseline.nlargest(10, "score") if "score" in df_baseline.columns else pd.DataFrame()
    
    # Counterfactual 1: No event type filter
    df_cf1 = features.copy()
    cf1_topk = df_cf1.nlargest(10, "score") if "score" in df_cf1.columns else pd.DataFrame()
    
    # Counterfactual 2: Looser novelty threshold
    df_cf2 = features.copy()
    if "novelty" in df_cf2.columns:
        df_cf2 = df_cf2[df_cf2["novelty"] >= 0.3]  # vs 0.6 baseline
    cf2_topk = df_cf2.nlargest(10, "score") if "score" in df_cf2.columns else pd.DataFrame()
    
    return {
        "baseline_candidates": len(baseline_topk),
        "counterfactual_no_filter_candidates": len(cf1_topk),
        "counterfactual_loose_novelty_candidates": len(cf2_topk),
        "baseline_top_symbols": baseline_topk["symbol"].tolist() if "symbol" in baseline_topk.columns else [],
        "cf1_top_symbols": cf1_topk["symbol"].tolist() if "symbol" in cf1_topk.columns else [],
        "cf2_top_symbols": cf2_topk["symbol"].tolist() if "symbol" in cf2_topk.columns else [],
    }

def compute_skip_reasons(
    week_summary: dict,
    skip_thresholds: dict,
) -> dict:
    """
    Generate structured skip reasons with codes and values.
    Turns "SKIP" into an explorable decision.
    """
    reasons = []
    is_skip = False
    
    # Check each threshold
    event_intensity = week_summary.get("event_intensity", 0.0)
    min_ei = skip_thresholds.get("min_event_intensity", 0.75)
    if event_intensity < min_ei:
        reasons.append({
            "code": "EVENT_INTENSITY_TOO_LOW",
            "value": round(event_intensity, 3),
            "threshold": min_ei,
        })
        is_skip = True
    
    price_action_share = week_summary.get("price_action_recap_share", 0.0)
    max_pa = skip_thresholds.get("max_price_action_recap_share", 0.80)
    if price_action_share > max_pa:
        reasons.append({
            "code": "PRICE_ACTION_RECAP_SHARE_TOO_HIGH",
            "value": round(price_action_share, 3),
            "threshold": max_pa,
        })
        is_skip = True
    
    high_sev_clusters = week_summary.get("high_severity_clusters", 0)
    min_hsc = skip_thresholds.get("min_high_severity_clusters", 2)
    if high_sev_clusters < min_hsc:
        reasons.append({
            "code": "HIGH_SEVERITY_CLUSTERS_TOO_LOW",
            "value": high_sev_clusters,
            "threshold": min_hsc,
        })
        is_skip = True
    
    return {
        "is_skip": is_skip,
        "reasons": reasons,
    }

def compute_sensitivity_check(
    candidate_symbols: list[str],
    universe_symbols: list[str],
    next_week_returns: pd.DataFrame,  # columns: symbol, return_1w, volatility_1w
) -> dict:
    """
    Sanity check: Are top candidates actually different from universe?
    
    If candidates are indistinguishable from random, features aren't informative.
    If they're systematically higher-vol, you at least have "pressure detection".
    """
    if next_week_returns.empty:
        return {"error": "No next-week returns available"}
    
    candidates = next_week_returns[next_week_returns["symbol"].isin(candidate_symbols)]
    universe = next_week_returns[next_week_returns["symbol"].isin(universe_symbols)]
    
    return {
        "candidate_median_abs_return": float(candidates["return_1w"].abs().median()) if len(candidates) > 0 else 0.0,
        "universe_median_abs_return": float(universe["return_1w"].abs().median()) if len(universe) > 0 else 0.0,
        "candidate_median_volatility": float(candidates["volatility_1w"].median()) if len(candidates) > 0 else 0.0,
        "universe_median_volatility": float(universe["volatility_1w"].median()) if len(universe) > 0 else 0.0,
        "n_candidates": len(candidates),
        "n_universe": len(universe),
    }

def write_diagnostics(
    week_end: str,
    coverage: CoverageDiagnostics,
    counterfactual: dict,
    skip_reasons: dict,
    sensitivity: Optional[dict],
    output_dir: Path,
) -> None:
    """Write all diagnostics to JSON for review."""
    diagnostics = {
        "week_end": week_end,
        "coverage": {
            "n_symbols_with_news": coverage.n_symbols_with_news,
            "n_clusters": coverage.n_clusters,
            "cluster_size_distribution": coverage.cluster_size_distribution,
            "event_type_distribution": coverage.event_type_distribution,
            "high_confidence_share": round(coverage.high_confidence_share, 3),
            "price_action_recap_share": round(coverage.price_action_recap_share, 3),
        },
        "counterfactual": counterfactual,
        "skip": skip_reasons,
        "sensitivity": sensitivity or {},
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_path = output_dir / "diagnostics.json"
    
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    
    print(f"✅ Wrote diagnostics to {diag_path}")
