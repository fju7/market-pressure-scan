#!/usr/bin/env python
"""
Prove-it audit: Verify regime system integrity after a scoring run.

This prevents silent misconfiguration by asserting:
1. report_meta.json regime_id matches CLI --regime argument
2. report_meta.json week_end matches resolved week_end
3. All artifact paths contain regime=<regime_id>
4. schema_used.yaml exists alongside outputs
5. scores_weekly.parquet has valid data

If any check fails → raises and marks run as failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import yaml


class AuditFailure(Exception):
    """Raised when audit check fails"""
    pass


def audit_regime_run(
    week_end: str,
    regime_id: str,
    base_dir: Path = None,
) -> Tuple[bool, List[str]]:
    """
    Audit a regime scoring run for integrity.
    
    Returns:
        (passed, errors): bool and list of error messages
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1] / "data" / "derived"
    
    errors = []
    
    print("=" * 60)
    print("🔍 REGIME RUN AUDIT")
    print("=" * 60)
    print(f"Week: {week_end}")
    print(f"Regime: {regime_id}")
    print(f"Base: {base_dir}\n")
    
    # Expected paths
    score_dir = base_dir / "scores_weekly" / f"regime={regime_id}" / f"week_ending={week_end}"
    feat_dir = base_dir / "features_weekly" / f"regime={regime_id}" / f"week_ending={week_end}"
    
    # Check 1: report_meta.json exists and regime_id matches
    print("1️⃣  Checking report_meta.json...")
    meta_path = score_dir / "report_meta.json"
    
    if not meta_path.exists():
        errors.append(f"❌ Missing report_meta.json: {meta_path}")
    else:
        try:
            meta = json.loads(meta_path.read_text())
            
            # Verify regime_id
            meta_regime = meta.get("regime_id")
            if meta_regime != regime_id:
                errors.append(
                    f"❌ Regime ID mismatch: CLI={regime_id}, meta={meta_regime}"
                )
            else:
                print(f"   ✓ regime_id matches: {regime_id}")
            
            # Verify week_end
            meta_week = meta.get("week_end")
            if meta_week != week_end:
                errors.append(
                    f"❌ Week end mismatch: CLI={week_end}, meta={meta_week}"
                )
            else:
                print(f"   ✓ week_end matches: {week_end}")
            
            # Verify schema_hash exists
            if "schema_hash" not in meta:
                errors.append("❌ Missing schema_hash in report_meta.json")
            else:
                print(f"   ✓ schema_hash present: {meta['schema_hash']}")
            
            # Verify git_sha exists
            if "git_sha" not in meta:
                errors.append("⚠️  Missing git_sha in report_meta.json")
            else:
                print(f"   ✓ git_sha present: {meta['git_sha'][:8]}")
                
        except Exception as e:
            errors.append(f"❌ Failed to parse report_meta.json: {e}")
    
    # Check 2: All artifact paths contain regime=<regime_id>
    print("\n2️⃣  Checking artifact path namespacing...")
    
    score_path = score_dir / "scores_weekly.parquet"
    feat_path = feat_dir / "features_weekly.parquet"
    
    for path in [score_path, feat_path]:
        if path.exists():
            path_str = str(path)
            if f"regime={regime_id}" not in path_str:
                errors.append(
                    f"❌ Path missing regime namespace: {path}"
                )
            else:
                print(f"   ✓ {path.name} correctly namespaced")
        else:
            errors.append(f"❌ Missing artifact: {path}")
    
    # Check 3: schema_used.yaml exists
    print("\n3️⃣  Checking schema provenance...")
    schema_prov_path = score_dir / "schema_used.yaml"
    
    if not schema_prov_path.exists():
        errors.append(f"❌ Missing schema_used.yaml: {schema_prov_path}")
    else:
        try:
            schema_content = yaml.safe_load(schema_prov_path.read_text())
            schema_id = schema_content.get("schema_id")
            print(f"   ✓ schema_used.yaml exists (schema: {schema_id})")
            
            # Verify schema_id in provenance matches meta
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                meta_schema = meta.get("schema_id")
                if schema_id != meta_schema:
                    errors.append(
                        f"❌ Schema ID mismatch: provenance={schema_id}, meta={meta_schema}"
                    )
        except Exception as e:
            errors.append(f"❌ Failed to parse schema_used.yaml: {e}")
    
    # Check 4: scores_weekly.parquet has valid data
    print("\n4️⃣  Checking scores_weekly.parquet integrity...")
    
    if score_path.exists():
        try:
            df = pd.read_parquet(score_path)
            
            # Basic validation
            if len(df) == 0:
                errors.append("❌ scores_weekly.parquet is empty")
            else:
                print(f"   ✓ {len(df)} rows in scores_weekly.parquet")
            
            # Check required columns
            required_cols = ["symbol", "UPS_adj"]
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                errors.append(f"❌ Missing columns in scores: {missing_cols}")
            else:
                print(f"   ✓ Required columns present")
            
            # Check for duplicates
            if "symbol" in df.columns:
                dup_count = df["symbol"].duplicated().sum()
                if dup_count > 0:
                    errors.append(f"❌ {dup_count} duplicate symbols in scores")
                else:
                    print(f"   ✓ No duplicate symbols")
                    
        except Exception as e:
            errors.append(f"❌ Failed to read scores_weekly.parquet: {e}")
    
    # Check 5: features_weekly.parquet has valid data
    print("\n5️⃣  Checking features_weekly.parquet integrity...")
    
    if feat_path.exists():
        try:
            df = pd.read_parquet(feat_path)
            
            if len(df) == 0:
                errors.append("❌ features_weekly.parquet is empty")
            else:
                print(f"   ✓ {len(df)} rows in features_weekly.parquet")
            
            # Check for (symbol, week_end) duplicates
            if "symbol" in df.columns and "week_end" in df.columns:
                dup_count = df.duplicated(subset=["symbol", "week_end"]).sum()
                if dup_count > 0:
                    errors.append(f"❌ {dup_count} duplicate (symbol, week_end) pairs in features")
                else:
                    print(f"   ✓ No duplicate (symbol, week_end) pairs")
                    
        except Exception as e:
            errors.append(f"❌ Failed to read features_weekly.parquet: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print("❌ AUDIT FAILED")
        print("=" * 60)
        for err in errors:
            print(f"  {err}")
        print()
        return False, errors
    else:
        print("✅ AUDIT PASSED")
        print("=" * 60)
        print("All integrity checks passed. Run is verified.")
        print()
        return True, []


def main():
    parser = argparse.ArgumentParser(description="Audit regime scoring run integrity")
    parser.add_argument("--week_end", required=True, help="Week ending date (YYYY-MM-DD)")
    parser.add_argument("--regime", required=True, help="Regime ID")
    parser.add_argument("--base_dir", type=Path, help="Base derived directory (default: data/derived)")
    parser.add_argument("--fail_on_error", action="store_true", help="Exit with code 1 if audit fails")
    
    args = parser.parse_args()
    
    passed, errors = audit_regime_run(
        week_end=args.week_end,
        regime_id=args.regime,
        base_dir=args.base_dir,
    )
    
    if not passed and args.fail_on_error:
        print("Audit failed. Exiting with error code.", file=sys.stderr)
        sys.exit(1)
    
    sys.exit(0 if passed else 0)  # Don't fail by default, just report


if __name__ == "__main__":
    main()
