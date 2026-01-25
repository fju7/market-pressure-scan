# src/io_atomic.py
from __future__ import annotations
from pathlib import Path
import os
import pandas as pd

def write_parquet_atomic(df: pd.DataFrame, path: str | Path, **to_parquet_kwargs) -> None:
    """Write parquet atomically to prevent corruption on interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, **to_parquet_kwargs)
    os.replace(tmp, path)  # atomic on same filesystem
