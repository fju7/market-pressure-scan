# Artifact Durability & Audit Trail Implementation

## Summary

Complete implementation of run metadata tracking, artifact auditing, and durability mechanisms to prevent artifact contamination and ensure every run is reproducible.

---

## 1. Run Metadata Tracking

**File:** `src/report_weekly.py`

Every `report_meta.json` now includes:

```json
"run_metadata": {
  "run_trigger": "workflow_dispatch",  // or "schedule"
  "week_end_requested": "2026-01-16",  // user input (null for scheduled)
  "week_end_resolved": "2026-01-16",   // actual computed week
  "run_started_utc": "2026-01-16T21:05:30Z",
  "git_sha": "abc123..."
}
```

**Benefits:**
- Clear audit trail for every report
- Distinguish manual vs scheduled runs
- No confusion about artifact provenance

---

## 2. Single Source of Truth Validation

**File:** `src/run_weekly_pipeline.py`

```python
# Validates week_end.txt matches --week_end argument
# FAILS FAST if mismatch detected
```

**Workflow:** `.github/workflows/weekly.yml`
- Sets `RUN_TRIGGER`, `WEEK_END_REQUESTED`, `RUN_STARTED_UTC` env vars
- Writes `week_end.txt` after computing the week

**Benefits:**
- Prevents artifact contamination
- Impossible for scripts to use different weeks
- Clear error message on mismatch

---

## 3. Post-Run Artifact Audit

**File:** `scripts/audit_artifacts.py`

Verifies all artifact types contain the expected week:
- reports
- baskets  
- scores
- features
- news_clusters
- rep_enriched
- trader_sheets

**Output includes:**
```
======================================================================
POST-RUN ARTIFACT AUDIT
======================================================================

✓ reports              - 2026-01-16
✓ baskets              - 2026-01-16
✓ scores               - 2026-01-16
...

✅ AUDIT PASSED
   Expected week 2026-01-16 present in all checked artifact types
======================================================================

✅ RUN VERIFIED: week_end_resolved=2026-01-16 artifacts_ok=YES
```

**Single-line proof for easy grepping:**
```bash
# In CI logs:
grep "RUN VERIFIED" 
# Returns: ✅ RUN VERIFIED: week_end_resolved=2026-01-16 artifacts_ok=YES
```

---

## 4. Artifact Durability

**Workflow:** `.github/workflows/weekly.yml`

Every run uploads a complete week bundle:

```yaml
- name: Upload week artifacts
  if: always()  # Captures artifacts even on failure
  uses: actions/upload-artifact@v4
  with:
    name: week_${{ steps.week_end.outputs.week_end }}_run_${{ github.run_id }}
```

**Bundle includes:**
- `week_end.txt` (single source of truth)
- `basket.csv`
- `weekly_report.md` + `report_meta.json`
- `trader_sheet.pdf` + `trader_sheet.csv`
- `scores_weekly.parquet`
- `features_weekly.parquet`
- `scoreboard.csv`
- `weeks_log.csv`

**Benefits:**
- Every run is downloadable and reproducible
- Unique name prevents overwrites: `week_2026-01-16_run_123456789`
- `if: always()` captures partial results on failure
- Zero risk of committing generated data to git
- Manual runs retained and accessible

---

## Testing

### Run the audit locally:
```bash
python scripts/audit_artifacts.py --week_end 2026-01-16
```

### Demo run metadata:
```bash
python demo_run_metadata.py
```

### Test validation:
```bash
python test_run_metadata.py
```

---

## CI Integration

The workflow now runs:
1. Compute `week_end` → write to `week_end.txt`
2. Set environment variables (run trigger, requested week, timestamp)
3. Run weekly pipeline (validates `week_end.txt`)
4. **Post-run artifact audit** ← Verifies consistency
5. **Upload week bundle** ← Ensures durability
6. Create notification issue

---

## Grepping CI Logs

```bash
# Check if artifacts are consistent:
grep "RUN VERIFIED"

# Check which weeks were processed:
grep "POST-RUN ARTIFACT AUDIT" -A 20

# Verify week_end validation passed:
grep "week_end validation passed"
```

---

## Resolution

✅ **Request 1:** Run metadata stamped into `report_meta.json`  
✅ **Request 2:** Single source of truth validation (fail-fast on mismatch)  
✅ **Request 3:** Post-run artifact audit with grep-able proof  
✅ **Request 4:** Artifact bundle uploaded every run for durability

**Result:** Impossible to mix up artifacts, every run is traceable and downloadable.
