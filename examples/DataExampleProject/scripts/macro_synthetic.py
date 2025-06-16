"""
Generate a synthetic macro-level dataset with three columns:
    • date – month-end over 24 months
    • macroindicator1 – continuous in the interval [0.01, 0.40]
    • macroindicator2 – continuous in the interval [100, 3000]

The dataset is written to CSV.
"""

import pandas as pd
import numpy as np

# --- Configuration ---------------------------------------------------------
END_DATE = "2025-12-31"  # last month-end date
N_MONTHS = 25  # total number of months
SEED = 306  # reproducibility
CSV_OUT = "macro_synthetic.csv"
# ---------------------------------------------------------------------------

np.random.seed(SEED)


def main():
    # Generate month-end dates
    dates = pd.date_range(end=END_DATE, periods=N_MONTHS, freq="ME")

    #    ─ GDP growth:   –2 % … +8 %  (typical recession to boom range)
    #    ─ CPI level:    start at 100 and compound monthly inflation of 0.1 % … 0.8 %
    #    ─ Unemployment: 3 % … 12 %
    gdp_growth = np.random.uniform(-0.02, 0.08, size=N_MONTHS)
    monthly_cpi_rt = np.random.uniform(0.001, 0.008, size=N_MONTHS)
    cpi_level = 100 * np.cumprod(1 + monthly_cpi_rt)
    unemployment = np.random.uniform(0.03, 0.12, size=N_MONTHS)

    gdp_growth = np.round(gdp_growth, 8)
    cpi_level = np.round(cpi_level, 8)
    unemployment = np.round(unemployment, 8)

    # Assemble DataFrame
    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y%m%d"),
            "gdp_growth": gdp_growth,
            "cpi": cpi_level,
            "unemployment": unemployment,
        }
    )

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")

    # Persist to disk
    df.to_csv(CSV_OUT, index=False)
    print(f"Dataset saved to '{CSV_OUT}'")


if __name__ == "__main__":
    main()
