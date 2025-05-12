#!/usr/bin/env python
"""CLI entry point for data aggregation."""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from aggregator.aggregate import aggregate_from_yaml


def main():
    parser = argparse.ArgumentParser(description="Aggregate data based on YAML config.")
    parser.add_argument(
        "config", type=str, help="Path to the aggregation YAML config file."
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to save the aggregated output file.",
    )
    args = parser.parse_args()

    # aggregate_from_yaml now returns a tuple (DataFrame, AggregationConfig)
    df, cfg = aggregate_from_yaml(args.config)

    output_path = Path(args.out)
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure parent directory exists

    # Determine output format
    file_format = "csv"  # Default
    parquet_engine = "auto"
    parquet_compression = "snappy"

    if cfg.output:  # Check if output configuration exists in YAML
        if cfg.output.format:
            file_format = cfg.output.format
        if cfg.output.parquet_engine:
            parquet_engine = cfg.output.parquet_engine
        if cfg.output.parquet_compression:
            parquet_compression = cfg.output.parquet_compression
    else:
        # Fallback: Infer format from output file extension if not in YAML
        if output_path.suffix.lower() == ".parquet":
            file_format = "parquet"
        elif output_path.suffix.lower() == ".csv":
            file_format = "csv"
        # else, it remains the default "csv"

    # Save the DataFrame
    if file_format == "parquet":
        df.to_parquet(
            output_path, engine=parquet_engine, compression=parquet_compression
        )
        print(f"Aggregated data saved to {output_path} (Parquet format)")
    elif file_format == "csv":
        df.to_csv(output_path, index=False)
        print(f"Aggregated data saved to {output_path} (CSV format)")
    else:
        print(f"Error: Unsupported output format '{file_format}'. Defaulting to CSV.")
        df.to_csv(output_path, index=False)
        print(f"Aggregated data saved to {output_path} (CSV format)")


if __name__ == "__main__":
    main()
