#!/usr/bin/env python
"""CLI entry point for data aggregation."""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from aggregator import aggregate_from_yaml


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate data sources to a single table.")
    p.add_argument("config", type=Path, help="YAML aggregation spec.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/aggregated.csv"),
        help="Destination CSV file.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_cli()
    df: pd.DataFrame = aggregate_from_yaml(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved aggregated dataset → {args.out}")


if __name__ == "__main__":
    main()
