from pathlib import Path
import json
import pandas as pd

SCORES_BASE = Path("data/derived/scores_weekly")
OUT = Path("data/derived/analysis/weeks_log_by_schema.csv")

def main():
    rows = []

    for regime_dir in sorted(SCORES_BASE.glob("regime=*")):
        regime = regime_dir.name.split("=", 1)[1]
        for schema_dir in sorted(regime_dir.glob("schema=*")):
            schema = schema_dir.name.split("=", 1)[1]
            for week_dir in sorted(schema_dir.glob("week_ending=*")):
                week_end = week_dir.name.split("=", 1)[1]
                meta_path = week_dir / "rescore_meta.json"
                if not meta_path.exists():
                    continue

                meta = json.loads(meta_path.read_text())
                rows.append({
                    "week_end": week_end,
                    "regime": regime,
                    "schema": schema,
                    "schema_hash": meta.get("schema_hash"),
                    "features_hash": (meta.get("input_hashes") or {}).get("features"),
                    "output_hash": meta.get("output_hash"),
                    "feature_mapping_used": json.dumps(meta.get("feature_mapping_used", {}), sort_keys=True),
                    "warnings": "; ".join(meta.get("feature_mapping_warnings", [])) if meta.get("feature_mapping_warnings") else "",
                })

    df = pd.DataFrame(rows).sort_values(["week_end", "regime", "schema"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")

if __name__ == "__main__":
    main()
