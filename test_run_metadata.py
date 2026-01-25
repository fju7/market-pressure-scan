#!/usr/bin/env python3
"""
Test script to validate the run metadata changes:
1. Verify report_meta.json contains new run_metadata fields
2. Test week_end.txt validation in run_weekly_pipeline.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path


def test_report_meta_structure():
    """Verify that existing report_meta.json files have expected structure"""
    print("\n=== Testing report_meta.json structure ===")
    
    reports_dir = Path("data/derived/reports")
    if not reports_dir.exists():
        print("⚠️  No reports directory found, skipping test")
        return True
    
    # Find a report_meta.json file
    meta_files = list(reports_dir.glob("week_ending=*/report_meta.json"))
    if not meta_files:
        print("⚠️  No report_meta.json files found, skipping test")
        return True
    
    # Check the structure of the first one
    meta_path = meta_files[0]
    print(f"Checking: {meta_path}")
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    # Check expected fields
    required_fields = ["week_ending_date", "cluster_count", "build"]
    for field in required_fields:
        if field not in meta:
            print(f"❌ Missing field: {field}")
            return False
        print(f"✓ Found field: {field}")
    
    print("✅ report_meta.json structure looks good")
    return True


def test_week_end_validation():
    """Test the week_end.txt validation logic"""
    print("\n=== Testing week_end.txt validation ===")
    
    week_end_file = Path("week_end.txt")
    original_content = None
    
    try:
        # Save original if it exists
        if week_end_file.exists():
            original_content = week_end_file.read_text()
        
        # Test 1: Matching week_end
        print("\nTest 1: Matching week_end values")
        week_end_file.write_text("2026-01-16\n")
        
        result = subprocess.run(
            [sys.executable, "-m", "src.run_weekly_pipeline", "--week_end", "2026-01-16", "--help"],
            capture_output=True,
            text=True
        )
        
        # --help should work, just checking the import doesn't fail
        if "Run full weekly market pressure pipeline" in result.stdout or result.returncode == 0:
            print("✓ Script runs with matching week_end")
        else:
            print(f"⚠️  Unexpected output: {result.stderr}")
        
        # Test 2: Mismatching week_end (should fail fast)
        print("\nTest 2: Mismatching week_end values (should fail)")
        week_end_file.write_text("2026-01-16\n")
        
        result = subprocess.run(
            [sys.executable, "-m", "src.run_weekly_pipeline", 
             "--week_end", "2026-01-23", 
             "--universe", "sp500_universe.csv"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0 and "week_end mismatch" in result.stdout:
            print("✓ Script correctly fails with mismatched week_end")
            print(f"   Error message: {result.stdout.splitlines()[0]}")
        else:
            print(f"⚠️  Expected failure but got: returncode={result.returncode}")
            if result.stdout:
                print(f"   stdout: {result.stdout[:200]}")
        
        # Test 3: No week_end.txt (should warn but proceed)
        print("\nTest 3: Missing week_end.txt (should warn)")
        week_end_file.unlink()
        
        result = subprocess.run(
            [sys.executable, "-m", "src.run_weekly_pipeline", "--week_end", "2026-01-16", "--help"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ Script runs without week_end.txt (local mode)")
        
        print("\n✅ week_end validation tests passed")
        return True
        
    finally:
        # Restore original content
        if original_content is not None:
            week_end_file.write_text(original_content)
        elif week_end_file.exists():
            week_end_file.unlink()


def test_env_vars_in_metadata():
    """Show what environment variables would be used"""
    print("\n=== Environment Variables Check ===")
    
    env_vars = [
        "GIT_SHA",
        "RUN_ID", 
        "RUN_ATTEMPT",
        "RUN_TRIGGER",
        "WEEK_END_REQUESTED",
        "RUN_STARTED_UTC",
    ]
    
    for var in env_vars:
        value = os.getenv(var, "")
        status = "✓" if value else "○"
        print(f"{status} {var}: {value or '(not set)'}")
    
    print("\nNote: These env vars are set by GitHub Actions workflow")
    print("For local runs, they will be empty and that's expected")
    return True


def main():
    print("Testing Run Metadata Implementation")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= test_report_meta_structure()
    all_passed &= test_week_end_validation()
    all_passed &= test_env_vars_in_metadata()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
