"""
Basic example: read semicolon-delimited data and launch the
ydata-profiling UI via paper_data’s analyzer.
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

from paper_data.wrangling.analyzer.ui import display_report  # type: ignore[import-untyped]


def main():
    parser = argparse.ArgumentParser(
        description="Generate a profiling report for sample_data.csv using paper_data analyzer."
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="Limit profiling to the last N rows of the dataset.",
    )
    parser.add_argument(
        "--explorative",
        action="store_true",
        help="Enable explorative analysis (limited to 1000 rows).",
    )
    parser.add_argument(
        "--no-minimal",
        dest="minimal",
        action="store_false",
        help="Disable minimal mode for more detailed profiling.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sample_data_profile.html",
        help="Filename for the generated HTML report (saved in PAPER/ directory).",
    )
    args = parser.parse_args()

    # 1) Locate the example CSV in the script directory
    here = Path(__file__).parent
    csv_path = here / "sample_data.csv"
    if not csv_path.exists():
        print(f"ERROR: sample_data.csv not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    # 2) Read the data (semicolon delimiter)
    df = pd.read_csv(csv_path, sep=";")

    # 3) Launch the profiling report
    #    This will generate an HTML file under PAPER/ and open it in your browser.
    report_path = display_report(
        df,
        output_path=args.output,
        max_rows=args.n_rows,
        minimal=args.minimal,
        explorative=args.explorative,
    )

    print(f"Profile report written to: {report_path}")


if __name__ == "__main__":
    main()
