#!/usr/bin/env python3
"""
Filter a CSV file by a date range.

Usage
-----
uv run scripts/split_dataset_by_date.py \
    --csv data/datashare.csv \
    --date-col DATE \
    --date-format '%Y%m%d' \
    --start 20020101 \
    --end 20211231 \
    --output data/filtered_chars_2002_2021.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """Return parsed command-line arguments."""
    parser = argparse.ArgumentParser(description="Filter a CSV file by date range.")
    parser.add_argument("--csv", required=True, help="Path to input CSV file.")
    parser.add_argument(
        "--date-col",
        required=True,
        help="Name of the column containing dates.",
    )
    parser.add_argument(
        "--date-format",
        required=True,
        help="Strftime pattern describing the date column (e.g. '%%Y-%%m-%%d').",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Inclusive start date (same format as --date-format). Omit for open start.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Inclusive end date (same format as --date-format). Omit for open end.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. If omitted, result is printed to stdout.",
    )
    return parser.parse_args()


def filter_by_date(
    df: pd.DataFrame,
    column: str,
    fmt: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """
    Return *df* restricted to rows whose *column* values lie in [start, end].

    Parameters
    ----------
    df : pandas.DataFrame
        Data source.
    column : str
        Name of the date column.
    fmt : str
        Strftime pattern used to parse *column*.
    start, end : str | None
        Inclusive bounds; pass *None* for an open bound.

    Returns
    -------
    pandas.DataFrame
        Filtered data.
    """
    dates = pd.to_datetime(df[column], format=fmt, errors="coerce")
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= dates >= pd.to_datetime(start, format=fmt)
    if end is not None:
        mask &= dates <= pd.to_datetime(end, format=fmt)
    return df.loc[mask]


def main() -> None:
    """Entry point."""
    args = parse_arguments()

    df = pd.read_csv(args.csv)
    if args.date_col not in df.columns:
        sys.exit(f"Column {args.date_col!r} not found in {args.csv}")

    filtered = filter_by_date(
        df,
        column=args.date_col,
        fmt=args.date_format,
        start=args.start,
        end=args.end,
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(args.output, index=False)
    else:
        filtered.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
