"""
Generate a synthetic Fama-French factor dataset.
"""

import pandas as pd
import numpy as np

# --- Configuration ---------------------------------------------------------
END_DATE = "2025-12-31"  # last month-end date
N_MONTHS = 24  # total number of months
SEED = 306  # reproducibility
CSV_OUT = "synthetic_factors.csv"
# ---------------------------------------------------------------------------

np.random.seed(SEED)


def main():
    # Generate month-end dates
    dates = pd.date_range(end=END_DATE, periods=N_MONTHS, freq="ME")

    # Generate synthetic factor returns
    # MKT-RF (Market Excess Return)
    mkt_rf = np.random.normal(0.005, 0.03, size=N_MONTHS)  # Mean 0.5%, Std 3%
    # SMB (Small Minus Big)
    smb = np.random.normal(0.002, 0.02, size=N_MONTHS)  # Mean 0.2%, Std 2%
    # HML (High Minus Low)
    hml = np.random.normal(0.003, 0.025, size=N_MONTHS)  # Mean 0.3%, Std 2.5%
    # RF (Risk-Free Rate)
    rf = np.random.uniform(
        0.0001, 0.0005, size=N_MONTHS
    )  # Daily risk-free rate, convert to monthly

    # Round to 8 decimal places for consistency
    mkt_rf = np.round(mkt_rf, 8)
    smb = np.round(smb, 8)
    hml = np.round(hml, 8)
    rf = np.round(rf, 8)

    # Assemble DataFrame
    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y%m%d"),
            "mkt_rf": mkt_rf,
            "smb": smb,
            "hml": hml,
            "rf": rf,
        }
    )

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")

    # Persist to disk
    df.to_csv(CSV_OUT, index=False)
    print(f"Dataset saved to '{CSV_OUT}'")


if __name__ == "__main__":
    main()
