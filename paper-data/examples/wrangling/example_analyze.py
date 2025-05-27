"""
Basic example: read semicolon-delimited data and launch the
ydata-profiling UI via paper_data’s analyzer.
"""

import pandas as pd
import sys
from pathlib import Path

from paper_data.wrangling.analyzer.ui import display_report

# 1) Locate the example CSV
here = Path(__file__).parent
csv_path = here / "sample_data.csv"
if not csv_path.exists():
    print(f"ERROR: sample_data.csv not found at {csv_path}", file=sys.stderr)
    sys.exit(1)

# 2) Read the data (semicolon delimiter)
df = pd.read_csv(csv_path, sep=";")

# 3) Launch the profiling report
#    This will generate an HTML file and open it in your browser.

output_html = here / "sample_data_profile.html"
report_path = display_report(df, output_path=str(output_html))

print(f"Profile report written to: {report_path}")
