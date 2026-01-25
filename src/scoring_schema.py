# src/scoring_schema.py
"""
Versioned scoring schema loader with content hashing for provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import yaml

@dataclass(frozen=True)
class ScoringSchema:
    schema_id: str
    raw: dict
    content_hash: str
    
    def get_skip_rules(self) -> dict:
        return self.raw.get("skip_rules", {})
    
    def get_weights(self) -> dict:
        return self.raw.get("weights", {})
    
    def get_filters(self) -> dict:
        return self.raw.get("filters", {})

def load_schema(
    schema_id: str,
    schemas_dir: Path = Path("configs/scoring_schemas")
) -> ScoringSchema:
    """
    Load scoring schema from YAML with content hash.
    
    Args:
        schema_id: Schema identifier (e.g., "news-novelty-v1")
        schemas_dir: Directory containing schema YAML files
        
    Returns:
        ScoringSchema with raw config and content hash
        
    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If schema_id in file doesn't match filename
    """
    p = schemas_dir / f"{schema_id}.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"Unknown schema_id={schema_id}. Expected {p}\n"
            f"Available: {[f.stem for f in schemas_dir.glob('*.yaml')]}"
        )
    
    content = p.read_text(encoding="utf-8")
    raw = yaml.safe_load(content)
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    
    if raw.get("schema_id") != schema_id:
        raise ValueError(
            f"schema_id mismatch inside yaml: {raw.get('schema_id')} vs {schema_id}"
        )
    
    return ScoringSchema(schema_id=schema_id, raw=raw, content_hash=h)

def write_schema_provenance(
    schema: ScoringSchema,
    output_dir: Path,
    filename: str = "schema_used.yaml"
) -> Path:
    """
    Write exact schema YAML used for this run (provenance).
    
    This makes it impossible to "think" you ran v1b when you actually ran v1.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    # Write schema with hash header
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Schema content hash: {schema.content_hash}\n")
        f.write(f"# Auto-generated provenance record\n\n")
        yaml.safe_dump(schema.raw, f, default_flow_style=False, sort_keys=False)
    
    return output_path
