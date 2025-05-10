from pathlib import Path
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd

from core.settings import DATA_DIR


def load_monthly_data(
    path: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Load the raw CSV and return a numeric, month-end indexed DataFrame.

    Parameters
    ----------
    path : str
        Location of the source CSV.
    start, end : str | pd.Timestamp | None
        Optional inclusive date bounds. Accepts 'YYYY-MM', 'YYYY-MM-DD',
        or any pandas-parsable datetime object.

    Returns
    -------
    pd.DataFrame
        Cleaned monthly data restricted to the requested range.
    """
    df = pd.read_csv(path)
    df = df.replace({",": ""}, regex=True)
    numeric_cols = df.columns.drop("yyyymm")
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    date = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m") + MonthEnd(0)
    df = df.drop(columns="yyyymm")
    df.index = pd.Index(date)
    df.index.name = "date"

    if start is not None or end is not None:
        df = df.loc[
            pd.Timestamp(start) if start else None : pd.Timestamp(end) if end else None
        ]

    return df.sort_index()


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
    predictors["dp"] = np.log(raw["D12"]) - np.log(raw["Index"])
    predictors["ep"] = np.log(raw["E12"]) - np.log(raw["Index"])
    predictors["bm"] = raw["b/m"]
    predictors["ntis"] = raw["ntis"]
    predictors["tbl"] = raw["tbl"]
    predictors["tms"] = raw["lty"] - raw["tbl"]
    predictors["dfy"] = raw["BAA"] - raw["AAA"]
    predictors["svar"] = raw["svar"]
    return predictors


if __name__ == "__main__":
    original_macro_dataset_path: Path = DATA_DIR / "Welch_Goyal_2024_macro_monthly.csv"

    raw = load_monthly_data(
        path=str(original_macro_dataset_path),
        start="1957-01",
        end="2021-12",
    )

    wg_predictors = construct_welch_goyal_predictors(raw)
    wg_predictors.to_csv(DATA_DIR / "wg_8_predictors.csv")
