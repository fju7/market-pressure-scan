# Regime System - Quick Reference Card

## 🚀 Common Commands

### Run Scoring (Single Week)
```bash
# v1 (baseline - 40% novelty, 0% divergence)
python -m src.features_scores \
  --universe sp500_universe.csv \
  --week_end 2026-01-09 \
  --regime news-novelty-v1

# v1b (divergence-heavy - 20% novelty, 45% divergence)
python -m src.features_scores \
  --universe sp500_universe.csv \
  --week_end 2026-01-09 \
  --regime news-novelty-v1b
```

### Audit Run
```bash
python scripts/audit_regime_run.py \
  --week_end 2026-01-09 \
  --regime news-novelty-v1 \
  --fail_on_error
```

### Backfill Multiple Weeks
```bash
python scripts/backfill_regime_ledger.py \
  --start_week 2025-10-25 \
  --end_week 2026-01-16 \
  --regimes news-novelty-v1 news-novelty-v1b
```

### Compare Results
```bash
python - <<'PY'
import pandas as pd

v1 = pd.read_parquet("data/derived/scores_weekly/regime=news-novelty-v1/week_ending=2026-01-09/scores_weekly.parquet")
v1b = pd.read_parquet("data/derived/scores_weekly/regime=news-novelty-v1b/week_ending=2026-01-09/scores_weekly.parquet")

print("v1 top 10:", v1.nlargest(10, "UPS_adj")["symbol"].tolist())
print("v1b top 10:", v1b.nlargest(10, "UPS_adj")["symbol"].tolist())
PY
```

## 📂 Output Locations

### v1 (baseline)
```
data/derived/scores_weekly/regime=news-novelty-v1/week_ending=2026-01-09/
├── scores_weekly.parquet     ← Scores
├── report_meta.json          ← Provenance
└── schema_used.yaml          ← Exact config used
```

### v1b (divergence-heavy)
```
data/derived/scores_weekly/regime=news-novelty-v1b/week_ending=2026-01-09/
├── scores_weekly.parquet
├── report_meta.json
└── schema_used.yaml
```

## 🔍 Verification

### Check Provenance
```bash
cat data/derived/scores_weekly/regime=news-novelty-v1b/week_ending=2026-01-09/report_meta.json
```

Should show:
```json
{
  "regime_id": "news-novelty-v1b",
  "schema_hash": "5ba304f03c31389a",
  "git_sha": "facf01ed...",
  ...
}
```

### Run Audit
```bash
python scripts/audit_regime_run.py --week_end 2026-01-09 --regime news-novelty-v1b --fail_on_error
```

Expected output:
```
✅ AUDIT PASSED
All integrity checks passed. Run is verified.
```

## 📊 Schema Comparison

| Aspect | v1 (Baseline) | v1b (Divergence) |
|--------|---------------|-------------------|
| Novelty Weight | 40% | 20% |
| Divergence Weight | 0% | 45% |
| Event Intensity Weight | 35% | 25% |
| Sentiment Weight | 25% | 10% |
| Min Event Intensity | 0.75 | 0.35 (looser) |
| Max PA Recap Share | 80% | 85% (looser) |
| Min High Severity Clusters | 2 | 1 (looser) |
| **Philosophy** | Pure news signal | Counter-narrative trades |

## 🛠️ Troubleshooting

### "TypeError: DerivedPaths.__init__() got unexpected keyword argument 'regime'"
**Fix:** Use direct path construction in `default_paths()`:
```python
features_base = derived / "features_weekly" / f"regime={regime_id}"
scores_base = derived / "scores_weekly" / f"regime={regime_id}"
```

### "Audit failed: regime_id mismatch"
**Cause:** CLI arg doesn't match output
**Fix:** Check `--regime` argument matches actual run

### "Missing schema_used.yaml"
**Cause:** Old run before provenance system
**Fix:** Re-run with current `features_scores.py`

## 📝 Integration Checklist

- [x] features_scores.py accepts --regime arg
- [x] Outputs go to regime-namespaced paths
- [x] report_meta.json includes regime_id + schema_hash
- [x] schema_used.yaml written alongside outputs
- [x] Audit script validates integrity
- [ ] CI/CD runs both regimes (optional)
- [ ] Ledger tracks historical decisions (optional)

## 🎯 Key Files

| File | Purpose |
|------|---------|
| `src/features_scores.py` | Main scoring pipeline (regime-aware) |
| `src/scoring_schema.py` | Schema loader with SHA256 hashing |
| `src/io_atomic.py` | Atomic write utility |
| `configs/scoring_schemas/news-novelty-v1.yaml` | v1 config |
| `configs/scoring_schemas/news-novelty-v1b.yaml` | v1b config |
| `scripts/audit_regime_run.py` | Prove-it audit |
| `scripts/backfill_regime_ledger.py` | Backfill + ledger |

## ⚡ Quick Wins

1. **Verify current run:**
   ```bash
   python scripts/audit_regime_run.py --week_end <DATE> --regime <ID> --fail_on_error
   ```

2. **Compare two regimes:**
   ```bash
   # Just check report_meta.json
   diff <(jq . data/derived/scores_weekly/regime=news-novelty-v1/week_ending=2026-01-09/report_meta.json) \
        <(jq . data/derived/scores_weekly/regime=news-novelty-v1b/week_ending=2026-01-09/report_meta.json)
   ```

3. **See what schema was used:**
   ```bash
   cat data/derived/scores_weekly/regime=<ID>/week_ending=<DATE>/schema_used.yaml
   ```

## 🚨 Guard Rails

1. **Audit after every run:**
   ```bash
   python scripts/audit_regime_run.py --week_end $WEEK_END --regime $REGIME --fail_on_error
   ```

2. **Check for overwrites:**
   ```bash
   # Both should exist
   ls data/derived/scores_weekly/regime=news-novelty-v1/week_ending=2026-01-09/
   ls data/derived/scores_weekly/regime=news-novelty-v1b/week_ending=2026-01-09/
   ```

3. **Verify schema hash changed:**
   ```bash
   # Should be different
   jq .schema_hash data/derived/scores_weekly/regime=news-novelty-v1/week_ending=*/report_meta.json
   jq .schema_hash data/derived/scores_weekly/regime=news-novelty-v1b/week_ending=*/report_meta.json
   ```

## 📚 Documentation

- **REGIME_INTEGRATION_SUMMARY.md** - Executive summary
- **REGIME_INTEGRATION_COMPLETE.md** - Full implementation details
- **CI_INTEGRATION_GUIDE.md** - GitHub Actions examples
- **INCREMENTAL_SYSTEM.md** - Original system docs
- **This file** - Quick reference

---

**Need Help?** Check audit output first, then review provenance files.
