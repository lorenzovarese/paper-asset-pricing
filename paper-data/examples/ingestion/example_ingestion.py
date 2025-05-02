"""
Example ingestion script for PAPER data.

Downloads three datasets into a local `PAPER/` folder in the current working directory:
  1) Welch & Goyal monthly return predictors via HTTP CSV export
  2) Empirical Asset Pricing via Machine Learning data (zip) from Dacheng Xiu’s website
     https://dachxiu.chicagobooth.edu/download/datashare.zip
  3) CRSP monthly returns (1957–2021) via WRDS
"""

import sys
from pathlib import Path
from getpass import getpass
import os

# 1) HTTP connector for Welch & Goyal data
from paper_data.ingestion.http import HTTPConnector  # type: ignore[import-untyped]

# 3) WRDS connector for CRSP data
from paper_data.ingestion.wrds_conn import WRDSConnector  # type: ignore[import-untyped]


def main():
    out_dir = Path.cwd() / "PAPER"
    out_dir.mkdir(exist_ok=True)

    # 1) Welch & Goyal predictors (monthly) via public CSV export
    sheet_id = "1OIZg6htTK60wtnCVXvxAujvG1aKEOVYv"
    monthly_gid = "1660564386"
    csv_url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        f"/export?format=csv&gid={monthly_gid}"
    )
    try:
        welch_df = HTTPConnector(csv_url).get_data()
        path_welch = out_dir / "welch_goyal_monthly.csv"
        welch_df.to_csv(path_welch, index=False)
        print(f"✔ Saved Welch & Goyal monthly data to {path_welch}")
    except Exception as e:
        print(
            f"ERROR: could not ingest Welch & Goyal data: {e}",
            file=sys.stderr,
        )

    # 2) Empirical Asset Pricing ML dataset (zip)
    print(
        "It is not possible to download the ZIP file directly from the website. "
        "Due to SSL certificate issues, you need to download it manually from "
        "https://dachxiu.chicagobooth.edu/download/datashare.zip and place it in the "
        "PAPER/ directory."
    )

    # 3) CRSP monthly returns (1957-01 to 2021-12) via WRDS
    query = """
    SELECT permno, date, ret
      FROM crsp.msf
     WHERE date BETWEEN '1957-01-01' AND '2021-12-31'
    """
    wrds_user = os.environ.get("WRDS_USER") or input("Enter your WRDS username: ")
    wrds_pass = os.environ.get("WRDS_PASSWORD") or getpass("Enter your WRDS password: ")
    try:
        wrds = WRDSConnector(query, user=wrds_user, password=wrds_pass)
        df_crsp = wrds.get_data()
        path_crsp = out_dir / "crsp_monthly_returns.parquet"
        df_crsp.to_parquet(path_crsp, index=False)
        print(f"✔ Saved CRSP monthly returns to {path_crsp}")
    except Exception as e:
        print(
            "ERROR: could not ingest CRSP returns: "
            f"{e}\n"
            "→ Please verify your WRDS credentials or ~/.pgpass configuration.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
