def main() -> None:

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil import tz

from .io_atomic import write_parquet_atomic
from .reuse import should_skip
from .scoring_schema import load_schema, write_schema_provenance

NY = tz.gettz("America/New_York")


# =============================================================================
# Repo root helper (more robust than parents[1])
# =============================================================================
def _find_repo_root(start: Path) -> Path:
    """
    Walk upward looking for common repo markers.
    Falls back to the prior behavior if no marker is found.
    """
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    # fallback: assume src/ is one level under repo root
    return start.resolve().parents[1]


# =============================================================================
# Debug / diagnostics
# =============================================================================
def _debug_env_stamp() -> None:
    print("=== FEATURES_SCORES DEBUG STAMP ===")
    print("module_file:", __file__)
    print("python:", platform.python_version())
    print("pandas:", pd.__version__, "numpy:", np.__version__)
    try:
        src = Path(__file__).read_bytes()
        print("features_scores.py sha256:", hashlib.sha256(src).hexdigest()[:16])
    except Exception as e:
        print("sha256 read failed:", e)
    print("===================================")


def dump_df(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    """
    Safe debug dump. Does NOT re-import or embed module content.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.parquet"
    try:
        write_parquet_atomic(df, p)
        print(f"[debug] wrote {p}")
    except Exception as e:
        print(f"[debug] failed to write {p}: {e}")


# =============================================================================
# JSON coercion helper (prevents silent all-zero features)
# =============================================================================
def _as_dict(x) -> dict:
    if isinstance(x, dict):
        return x
    if isinstance(x, str) and x.strip().startswith(("{", "[")):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}

# ...existing code from user-provided implementation continues...
