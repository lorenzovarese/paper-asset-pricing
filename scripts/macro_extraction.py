from pathlib import Path
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd

from core.settings import DATA_DIR


def load_monthly_data(path: str) -> pd.DataFrame:
    """
    Load the raw CSV and return a numeric, month-end indexed DataFrame.

    Parameters
    ----------
    path : str
        Location of the source CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned data indexed by the last day of each month.
    """
    df = pd.read_csv(path)

    # remove thousands separators and coerce all non-date columns to float
    df = df.replace({",": ""}, regex=True)
    numeric_cols = df.columns.drop("yyyymm")
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # convert yyyymm to a proper month-end timestamp
    df["date"] = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m") + MonthEnd(0)
    df = df.set_index("date").sort_index()
    return df.drop(columns="yyyymm")


def construct_welch_goyal_predictors(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the eight macroeconomic predictors defined in Welch & Goyal (2008).

    Parameters
    ----------
    raw : pd.DataFrame
        Monthly data containing at least the columns
        ['Index', 'D12', 'E12', 'b/m', 'ntis',
         'tbl', 'lty', 'AAA', 'BAA', 'svar'].

    Returns
    -------
    pd.DataFrame
        Predictor time-series with identical index to *raw*.
    """
    predictors = pd.DataFrame(index=raw.index)

    # log‐dividend–price and log‐earnings–price ratios
    predictors["dp"] = np.log(raw["D12"]) - np.log(raw["Index"])
    predictors["ep"] = np.log(raw["E12"]) - np.log(raw["Index"])

    # direct mappings or spreads
    predictors["bm"] = raw["b/m"]
    predictors["ntis"] = raw["ntis"]
    predictors["tbl"] = raw["tbl"]
    predictors["tms"] = raw["lty"] - raw["tbl"]  # term spread
    predictors["dfy"] = raw["BAA"] - raw["AAA"]  # default yield spread
    predictors["svar"] = raw["svar"]

    return predictors


if __name__ == "__main__":
    original_macro_dataset_path = DATA_DIR / "Welch_Goyal_2024_macro_monthly.csv"
    raw = load_monthly_data(str(original_macro_dataset_path))
    wg_predictors = construct_welch_goyal_predictors(raw)
    wg_predictors.to_csv(DATA_DIR / "wg_8_predictors.csv")
