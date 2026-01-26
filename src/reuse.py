from __future__ import annotations

from pathlib import Path

def should_skip(out_path: Path, force: bool) -> bool:
    """Return True if output exists and we're not forcing a rebuild."""
    return out_path.exists() and not force
def should_skip(out_path: Path, force: bool) -> bool:

from __future__ import annotations

from pathlib import Path

def should_skip(out_path: Path, force: bool) -> bool:
    """
    Return True if the output path exists and force is not set.
    Used to short-circuit expensive steps on reruns.
    """
    return out_path.exists() and not force
