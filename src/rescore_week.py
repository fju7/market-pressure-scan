# src/rescore_week.py
"""
Offline rescoring - rebuild scores from stored artifacts without network fetch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.scoring_schema import load_schema, write_schema_provenance
from src.derived_paths import DerivedPaths
from src.io_atomic import write_parquet_atomic

def hash_dataframe(df: pd.DataFrame) -> str:
    """Compute stable hash of DataFrame for provenance."""
    # Use parquet bytes for stable hash
    import io
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]

def rescore_week(
    week_end: str,
    schema_id: str,
    *,
    offline: bool = True,
    rebuild_features: bool = False,
    output_base: Path = Path("data/derived"),
) -> dict:
    """
    Rescore a single week using specified schema.
    
    Args:
        week_end: Week ending date YYYY-MM-DD
        schema_id: Scoring schema to use
        offline: If True, require all artifacts present (no network fetch)
        rebuild_features: If True, rebuild features from rep_enriched + candles
        output_base: Base directory for outputs
        
    Returns:
        dict with rescoring metadata
    """
    schema = load_schema(schema_id)
    dp = DerivedPaths(base=output_base)
    
    print(f"📊 Rescoring week {week_end} with schema {schema_id}")
    print(f"   Hash: {schema.content_hash}")
    print(f"   Offline: {offline}")
    
    # Verify inputs exist
    rep_enriched_dir = dp.week_dir("rep_enriched", week_end, regime=None)
    rep_enriched_path = rep_enriched_dir / "rep_enriched.parquet"
    
    if not rep_enriched_path.exists():
        raise FileNotFoundError(
            f"Cannot rescore offline: missing {rep_enriched_path}\n"
            f"Run ingestion first or disable --offline"
        )
    
    # Load rep_enriched
    rep_enriched = pd.read_parquet(rep_enriched_path)
    rep_hash = hash_dataframe(rep_enriched)
    print(f"✓ Loaded {len(rep_enriched)} enriched reps (hash: {rep_hash})")
    
    # Load or rebuild features
    if rebuild_features:
        print("🔨 Rebuilding features from rep_enriched + candles...")
        # TODO: Call feature builder here
        # For now, require features to exist
        raise NotImplementedError("rebuild_features not yet implemented")
    else:
        features_dir = dp.week_dir("features_weekly", week_end, regime=None)
        features_path = features_dir / "features_weekly.parquet"
        
        if not features_path.exists():
            raise FileNotFoundError(
                f"Cannot rescore: missing {features_path}\n"
                f"Run feature generation or use --rebuild_features"
            )
        
        features = pd.read_parquet(features_path)
        feat_hash = hash_dataframe(features)
        print(f"✓ Loaded {len(features)} features (hash: {feat_hash})")
    
    # Apply schema to compute scores
    print(f"🧮 Applying scoring schema {schema_id}...")
    scores = apply_scoring_schema(features, schema)
    score_hash = hash_dataframe(scores)
    
    # Write outputs to schema-namespaced directory
    scores_dir = output_base / "scores_weekly" / f"schema={schema_id}" / f"week_ending={week_end}"
    scores_dir.mkdir(parents=True, exist_ok=True)
    
    scores_path = scores_dir / "scores_weekly.parquet"
    write_parquet_atomic(scores, scores_path)
    print(f"✅ Wrote {len(scores)} scores to {scores_path}")
    
    # Write schema provenance
    schema_prov_path = write_schema_provenance(schema, scores_dir)
    print(f"✅ Wrote schema provenance to {schema_prov_path}")
    
    # Write metadata
    metadata = {
        "week_end": week_end,
        "schema_id": schema_id,
        "schema_hash": schema.content_hash,
        "offline": offline,
        "rebuild_features": rebuild_features,
        "input_hashes": {
            "rep_enriched": rep_hash,
            "features": feat_hash if not rebuild_features else None,
        },
        "output_hash": score_hash,
        "output_path": str(scores_path),
    }
    
    meta_path = scores_dir / "rescore_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Wrote metadata to {meta_path}")
    
    return metadata

def apply_scoring_schema(features: pd.DataFrame, schema: ScoringSchema) -> pd.DataFrame:
    """
    Apply scoring schema to features DataFrame.
    
    This is the pure scoring function: features + schema -> scores
    """
    import numpy as np
    
    weights = schema.get_weights()
    filters = schema.get_filters()
    
    df = features.copy()
    
    # Apply filters
    exclude_types = filters.get("exclude_event_types", [])
    if exclude_types and "event_type_primary" in df.columns:
        df = df[~df["event_type_primary"].isin(exclude_types)]
    
    # Ensure scoring columns exist
    for col in ["novelty", "event_intensity", "sentiment", "divergence"]:
        if col not in df.columns:
            df[col] = 0.0
    
    # Normalize to z-scores for fair weighting
    def zscore(s):
        vals = s.fillna(0.0).astype(float).to_numpy()
        mu = np.nanmean(vals)
        sd = np.nanstd(vals)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        z = (vals - mu) / sd
        return pd.Series(np.clip(z, -3, 3), index=s.index)
    
    df["novelty_z"] = zscore(df["novelty"])
    df["event_intensity_z"] = zscore(df["event_intensity"])
    df["sentiment_z"] = zscore(df["sentiment"])
    df["divergence_z"] = zscore(df["divergence"])
    
    # Compute weighted score
    df["score"] = (
        weights.get("novelty", 0.0) * df["novelty_z"] +
        weights.get("event_intensity", 0.0) * df["event_intensity_z"] +
        weights.get("sentiment", 0.0) * df["sentiment_z"] +
        weights.get("divergence", 0.0) * df["divergence_z"]
    )
    
    # Rank
    df["rank"] = df["score"].rank(ascending=False, method="min").astype(int)
    
    return df

def main():
    p = argparse.ArgumentParser(description="Rescore week with specified schema (offline)")
    p.add_argument("--week_end", required=True, help="Week ending date YYYY-MM-DD")
    p.add_argument("--schema", required=True, help="Schema ID (e.g., news-novelty-v1b)")
    p.add_argument("--offline", action="store_true", help="Require all artifacts present (no fetch)")
    p.add_argument("--rebuild_features", action="store_true", help="Rebuild features from rep_enriched")
    args = p.parse_args()
    
    metadata = rescore_week(
        week_end=args.week_end,
        schema_id=args.schema,
        offline=args.offline,
        rebuild_features=args.rebuild_features,
    )
    
    print(f"\n✅ Rescoring complete")
    print(f"   Schema: {metadata['schema_id']} (hash: {metadata['schema_hash']})")
    print(f"   Output: {metadata['output_path']}")

if __name__ == "__main__":
    main()
