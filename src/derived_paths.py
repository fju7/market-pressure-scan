# src/derived_paths.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DerivedPaths:
    base: Path = Path("data/derived")

    def week_dir(self, artifact: str, week_end: str, regime: str | None = None) -> Path:
        """
        Returns a directory like:
          data/derived/<artifact>/week_ending=YYYY-MM-DD/
        or (if regime provided):
          data/derived/<artifact>/regime=<regime>/week_ending=YYYY-MM-DD/
        """
        if regime:
            return self.base / artifact / f"regime={regime}" / f"week_ending={week_end}"
        return self.base / artifact / f"week_ending={week_end}"

    def file(self, artifact: str, filename: str, week_end: str, regime: str | None = None) -> Path:
        d = self.week_dir(artifact, week_end, regime)
        d.mkdir(parents=True, exist_ok=True)
        return d / filename
