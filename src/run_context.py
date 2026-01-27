# src/run_context.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import os


WEEK_END_FILE = Path("week_end.txt")


@dataclass(frozen=True)
class RunContext:
    week_end: date


def compute_week_end_today(tz: str = "America/New_York") -> date:
    today = datetime.now(ZoneInfo(tz)).date()
    # week_end = most recent Friday (or today if Friday)
    if today.weekday() == 4:
        return today
    days_since_fri = (today.weekday() - 4) % 7
    return today - timedelta(days=days_since_fri)


def load_week_end_from_file(path: Path = WEEK_END_FILE) -> date | None:
    if not path.exists():
        return None
    s = path.read_text().strip()
    if not s:
        return None
    return date.fromisoformat(s)


def get_week_end(arg_week_end: str | None = None) -> date:
    """
    Canonical week_end resolution order:
      1) explicit arg (dev/backfill only)
      2) env WEEK_END
      3) week_end.txt (repo root)
      4) compute from current date (NY timezone)
    """
    if arg_week_end:
        return date.fromisoformat(arg_week_end)

    env = os.environ.get("WEEK_END", "").strip()
    if env:
        return date.fromisoformat(env)

    f = load_week_end_from_file()
    if f:
        return f

    return compute_week_end_today()


def enforce_match(arg_week_end: str | None, canonical: date) -> None:
    """
    If user passes --week_end, it must match canonical (env/file).
    This prevents “disagreeing week_end” bugs.
    """
    if not arg_week_end:
        return
    provided = date.fromisoformat(arg_week_end)
    if provided != canonical:
        raise SystemExit(
            f"week_end mismatch: provided {provided.isoformat()} "
            f"but canonical is {canonical.isoformat()} (env WEEK_END / week_end.txt)"
        )
