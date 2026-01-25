#!/usr/bin/env python3
"""
Demo: Show how the new run_metadata fields will appear in report_meta.json

This demonstrates what the metadata will look like for:
1. Manual (workflow_dispatch) runs
2. Scheduled runs
"""
import json
import os


def demo_manual_run():
    """Simulate metadata for a manual workflow_dispatch run"""
    print("\n=== Manual Run (workflow_dispatch) ===")
    print("\nExpected environment variables:")
    print("  RUN_TRIGGER=workflow_dispatch")
    print("  WEEK_END_REQUESTED=2026-01-16  (from user input)")
    print("  GIT_SHA=abc123...")
    print("  RUN_STARTED_UTC=2026-01-16T21:05:30Z")
    
    run_metadata = {
        "run_trigger": "workflow_dispatch",
        "week_end_requested": "2026-01-16",
        "week_end_resolved": "2026-01-16",
        "run_started_utc": "2026-01-16T21:05:30Z",
        "git_sha": "abc123def456...",
    }
    
    print("\nResulting run_metadata in report_meta.json:")
    print(json.dumps({"run_metadata": run_metadata}, indent=2))


def demo_scheduled_run():
    """Simulate metadata for a scheduled run"""
    print("\n=== Scheduled Run (cron schedule) ===")
    print("\nExpected environment variables:")
    print("  RUN_TRIGGER=schedule")
    print("  WEEK_END_REQUESTED=  (empty, auto-computed)")
    print("  GIT_SHA=xyz789...")
    print("  RUN_STARTED_UTC=2026-01-23T21:05:00Z")
    
    run_metadata = {
        "run_trigger": "schedule",
        "week_end_requested": None,  # Null for scheduled runs
        "week_end_resolved": "2026-01-23",  # Auto-computed Friday
        "run_started_utc": "2026-01-23T21:05:00Z",
        "git_sha": "xyz789abc123...",
    }
    
    print("\nResulting run_metadata in report_meta.json:")
    print(json.dumps({"run_metadata": run_metadata}, indent=2))


def demo_week_end_validation():
    """Show the week_end.txt validation"""
    print("\n=== Single Source of Truth: week_end.txt ===")
    print("\nThe workflow writes week_end.txt after computing the week:")
    print("  echo '2026-01-23' > week_end.txt")
    
    print("\nThen run_weekly_pipeline.py validates it:")
    print("  ✓ week_end.txt contains:  2026-01-23")
    print("  ✓ --week_end argument:    2026-01-23")
    print("  → Validation passed!")
    
    print("\nIf there's a mismatch (artifact contamination):")
    print("  ❌ week_end.txt contains:  2026-01-16")
    print("  ❌ --week_end argument:    2026-01-23")
    print("  → CRITICAL ERROR: Pipeline fails fast!")


def main():
    print("=" * 70)
    print("Run Metadata Implementation Demo")
    print("=" * 70)
    
    demo_manual_run()
    demo_scheduled_run()
    demo_week_end_validation()
    
    print("\n" + "=" * 70)
    print("Benefits:")
    print("  1. Clear audit trail: know exactly how each week was triggered")
    print("  2. No confusion between manual and scheduled runs")
    print("  3. Fail-fast validation prevents artifact contamination")
    print("  4. Timestamps enable debugging timing-sensitive issues")
    print("=" * 70)


if __name__ == "__main__":
    main()
