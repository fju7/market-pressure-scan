# src/regime_config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class SkipRules:
    enabled: bool
    min_total_novelty: float
    min_high_severity_clusters: int
    max_price_action_share: float

@dataclass(frozen=True)
class ScoringWeights:
    novelty: float
    severity: float
    sentiment: float
    divergence: float

@dataclass(frozen=True)
class RegimeConfig:
    name: str
    skip_rules: SkipRules
    scoring_weights: ScoringWeights
    evaluation_mode: str  # "shadow" or "real" if you ever want

def load_regime(regime: str, root: Path = Path("config/regimes")) -> RegimeConfig:
    p = root / f"{regime}.yaml"
    data = yaml.safe_load(p.read_text())

    sr = data["skip_rules"]
    sw = data["scoring"]["weights"]

    return RegimeConfig(
        name=data["name"],
        skip_rules=SkipRules(
            enabled=bool(sr["enabled"]),
            min_total_novelty=float(sr["min_total_novelty"]),
            min_high_severity_clusters=int(sr["min_high_severity_clusters"]),
            max_price_action_share=float(sr["max_price_action_share"]),
        ),
        scoring_weights=ScoringWeights(
            novelty=float(sw["novelty"]),
            severity=float(sw["severity"]),
            sentiment=float(sw["sentiment"]),
            divergence=float(sw["divergence"]),
        ),
        evaluation_mode=str(data.get("evaluation", {}).get("mode", "shadow")),
    )
