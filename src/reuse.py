from __future__ import annotations

from pathlib import Path


def should_skip(out_path: Path, force: bool) -> bool:
    """
    Return True if the output path exists and --force is not set.
    """
    return out_path.exists() and not force
