from __future__ import annotations
from datetime import datetime
from typing import Union, Literal
import pandas as pd

DateLike = Union[str, datetime, pd.Timestamp]


def parse_and_normalize_date(
    date_like: DateLike,
    *,
    align_to: Literal["monthly", "daily", None] = "monthly",
) -> pd.Timestamp:
    """
    Parse a date-like input into a pd.Timestamp, then optionally align:
      - align_to="monthly": snap to month-end and normalize to midnight
      - align_to="daily": normalize to midnight
      - align_to=None: leave the parsed timestamp (no normalize)
    """
    ts = pd.to_datetime(date_like)
    if align_to == "monthly":
        # turn into end of month, then midnight
        ts = ts.to_period("M").to_timestamp(how="end").normalize()
    elif align_to == "daily":
        ts = ts.normalize()
    return ts
