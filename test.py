#!/usr/bin/env python3
"""
Example script to test the LocalConnector and the ydata-profiling analyzer
on PAPER/datashare.csv, producing a capped explorative report.
"""

import sys
from pathlib import Path

from paper_data.ingestion.local import LocalConnector  # type: ignore[import-untyped]
from paper_data.wrangling.analyzer.ui import display_report  # type: ignore[import-untyped]


def main():
    # 1) Locate the CSV file under PAPER/
    csv_path = Path("PAPER") / "datashare.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path!s} not found.", file=sys.stderr)
        sys.exit(1)

    # 2) Load the data
    try:
        connector = LocalConnector(str(csv_path))
        df = connector.get_data()
        print(f"✔ Successfully loaded data from {csv_path} (shape={df.shape})")
    except Exception as e:
        print(f"ERROR: failed to load data: {e}", file=sys.stderr)
        sys.exit(1)

    # 3) Generate and view the profiling report
    #    - max_rows: keep only the last 500 rows
    #    - explorative: enable, but capped at 1000 rows
    #    - minimal: False for full detail
    try:
        report_path = display_report(
            df,
            output_path="datashare_profile.html",
            max_rows=500,
            # minimal=False,
            # explorative=True,
        )
        print(f"✔ Profile report written to: {report_path}")
    except Exception as e:
        print(f"ERROR: failed to generate profiling report: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
