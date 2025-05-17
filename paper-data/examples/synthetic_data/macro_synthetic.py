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
END_DATE = "2025-06-30"  # last month-end observation
N_MONTHS = 24  # number of months
SEED = 306  # reproducibility
CSV_OUT = "macro_synthetic.csv"
# ---------------------------------------------------------------------------

np.random.seed(SEED)


def main():
    # Generate month-end dates
    dates = pd.date_range(end=END_DATE, periods=N_MONTHS, freq="M")

    # Draw synthetic macro indicators
    macroindicator1 = np.random.uniform(0.01, 0.40, size=N_MONTHS)
    macroindicator2 = np.random.uniform(100, 3000, size=N_MONTHS)

    # Assemble DataFrame
    df = pd.DataFrame(
        {
            "date": dates,
            "macroindicator1": macroindicator1,
            "macroindicator2": macroindicator2,
        }
    )

    # Persist to disk
    df.to_csv(CSV_OUT, index=False)
    print(f"Dataset saved to '{CSV_OUT}'")


if __name__ == "__main__":
    main()
