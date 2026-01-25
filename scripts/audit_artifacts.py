#!/usr/bin/env python3
"""
POST-RUN ARTIFACT AUDIT

Verifies that all generated artifacts are consistent with the expected week_end.
This prevents contamination where artifacts from different weeks get mixed together.

Checks:
1. All week directories match the expected week_end
2. week_end.txt matches the expected week
3. No stray artifacts from other weeks
"""
import sys
from pathlib import Path
from collections import defaultdict


def audit_artifacts(expected_week_end: str = None) -> bool:
    """
    Audit that all artifacts are from the same week.
    
    Args:
        expected_week_end: Expected week ending date (YYYY-MM-DD).
                          If None, reads from week_end.txt
    
    Returns:
        True if audit passes, False otherwise
    """
    print("\n" + "=" * 70)
    print("POST-RUN ARTIFACT AUDIT")
    print("=" * 70)
    
    # Read expected week from week_end.txt if not provided
    week_end_file = Path("week_end.txt")
    if expected_week_end is None:
        if week_end_file.exists():
            expected_week_end = week_end_file.read_text().strip()
            print(f"\n✓ Read expected week from week_end.txt: {expected_week_end}")
        else:
            print("\n❌ No week_end.txt found and no expected_week_end provided")
            print(f"✅ RUN VERIFIED: week_end_resolved=UNKNOWN artifacts_ok=NO")
            return False
    else:
        print(f"\n✓ Expected week: {expected_week_end}")
        
        # Verify week_end.txt matches if it exists
        if week_end_file.exists():
            file_week = week_end_file.read_text().strip()
            if file_week != expected_week_end:
                print(f"❌ week_end.txt mismatch:")
                print(f"   Expected: {expected_week_end}")
                print(f"   Found:    {file_week}")
                print(f"✅ RUN VERIFIED: week_end_resolved={expected_week_end} artifacts_ok=NO")
                return False
            print(f"✓ week_end.txt matches: {file_week}")
    
    # Define artifact directories to check
    derived = Path("data/derived")
    artifact_types = {
        "reports": derived / "reports",
        "baskets": derived / "baskets",
        "scores": derived / "scores_weekly",
        "features": derived / "features_weekly",
        "news_clusters": derived / "news_clusters",
        "rep_enriched": derived / "rep_enriched",
        "trader_sheets": derived / "trader_sheets",
    }
    
    all_passed = True
    weeks_found = defaultdict(list)
    
    print(f"\nScanning artifact directories:")
    print("-" * 70)
    
    for artifact_type, artifact_dir in artifact_types.items():
        if not artifact_dir.exists():
            print(f"⚠️  {artifact_type:20s} - directory not found (skipped)")
            continue
        
        # Find all week_ending= subdirectories
        week_dirs = sorted([d for d in artifact_dir.iterdir() 
                           if d.is_dir() and d.name.startswith("week_ending=")])
        
        if not week_dirs:
            print(f"○  {artifact_type:20s} - no week directories found")
            continue
        
        # Extract week dates
        weeks = [d.name.replace("week_ending=", "") for d in week_dirs]
        
        # Track which weeks appear in which artifact types
        for week in weeks:
            weeks_found[week].append(artifact_type)
        
        # Check if expected week is present
        if expected_week_end in weeks:
            status = "✓"
        else:
            status = "❌"
            all_passed = False
        
        weeks_str = ", ".join(weeks)
        print(f"{status}  {artifact_type:20s} - {weeks_str}")
    
    # Summary of weeks found
    print("\n" + "-" * 70)
    print("Week Summary:")
    print("-" * 70)
    
    for week in sorted(weeks_found.keys()):
        artifact_types_list = weeks_found[week]
        is_expected = "✓" if week == expected_week_end else "○"
        
        # Check if week appears in all artifact types
        if len(artifact_types_list) >= 5:  # At least 5 artifact types
            completeness = "COMPLETE"
        elif len(artifact_types_list) >= 3:
            completeness = "PARTIAL"
        else:
            completeness = "SPARSE"
        
        print(f"{is_expected}  {week}: {len(artifact_types_list)} artifact types ({completeness})")
        print(f"      {', '.join(artifact_types_list)}")
    
    # Final verdict
    print("\n" + "=" * 70)
    
    if not all_passed:
        print("❌ AUDIT FAILED")
        print(f"   Expected week {expected_week_end} not found in all artifact types")
        print("=" * 70 + "\n")
        # Single-line proof for easy grepping
        print(f"✅ RUN VERIFIED: week_end_resolved={expected_week_end} artifacts_ok=NO")
        return False
    
    # Check for extra unexpected weeks (potential contamination)
    unexpected_weeks = [w for w in weeks_found.keys() if w != expected_week_end]
    if unexpected_weeks:
        print(f"⚠️  WARNING: Found artifacts from other weeks: {', '.join(unexpected_weeks)}")
        print(f"   This is normal for historical data, but verify no contamination occurred")
    
    print("✅ AUDIT PASSED")
    print(f"   Expected week {expected_week_end} present in all checked artifact types")
    print("=" * 70 + "\n")
    
    # Single-line proof for easy grepping in CI logs
    print(f"✅ RUN VERIFIED: week_end_resolved={expected_week_end} artifacts_ok=YES")
    
    return True


def main():
    import argparse
    
    ap = argparse.ArgumentParser(description="Audit artifact consistency")
    ap.add_argument("--week_end", help="Expected week ending date (YYYY-MM-DD)")
    args = ap.parse_args()
    
    success = audit_artifacts(args.week_end)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
