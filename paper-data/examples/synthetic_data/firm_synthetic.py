"""
Generate a synthetic macro-level dataset with three columns:
    • date – month-end over 24 months
    • permco (1-5)
    • return (percentage)
    • volume (float)
    • marketcap (float)

The dataset is written to CSV.
"""

import pandas as pd
import numpy as np

# --- Configuration ---------------------------------------------------------
END_DATE = "2025-06-30"  # last month-end date
N_MONTHS = 24  # total number of months
PERMCO_RANGE = range(1, 6)  # permco identifiers 1-5
SEED = 306  # reproducibility
CSV_OUT = "firm_synthetic.csv"
# ---------------------------------------------------------------------------

np.random.seed(SEED)


def main():
    # Generate month-end dates (inclusive of END_DATE)
    dates = pd.date_range(end=END_DATE, periods=N_MONTHS, freq="ME")

    # Assemble observations
    records = []
    for date in dates:
        for permco in PERMCO_RANGE:
            rtn_pct = np.random.uniform(-0.2, 0.2)  # return in percentage points
            volume = np.random.uniform(1e5, 2e6)  # random trading volume
            marketcap = np.random.uniform(1e8, 1e11)  # random market capitalisation
            records.append((date, permco, rtn_pct, volume, marketcap))

    # Build DataFrame
    df = pd.DataFrame(
        records,
        columns=["date", "permco", "return", "volume", "marketcap"],
        dtype=object,  # keep numeric precision; 'date' remains datetime64
    )

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")

    # Persist to disk
    df.to_csv(CSV_OUT, index=False)
    print(f"Dataset saved to '{CSV_OUT}'")


if __name__ == "__main__":
    main()
