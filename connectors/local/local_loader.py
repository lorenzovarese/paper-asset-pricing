import pandas as pd
from pathlib import Path
from tqdm import tqdm
from core.settings import DATA_DIR

VERBOSE = True


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies global column standardization:
    - trims whitespace
    - converts to lowercase
    """
    df.columns = df.columns.str.strip().str.lower()
    return df


def load_and_preprocess(filename: str, date_column: str = "date") -> pd.DataFrame:
    """
    Internal loader with common preprocessing:
      - applies global column name standardization
      - parses `date_column` as datetime after standardization
      - sets MultiIndex(date, permno)
    """
    path = Path(DATA_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    df = _standardize_columns(df)

    if date_column.lower() not in df.columns:
        raise ValueError(
            f"Expected date column '{date_column}' not found after standardization."
        )

    df[date_column.lower()] = pd.to_datetime(df[date_column.lower()])
    df["permno"] = df["permno"].astype(int)
    df = df.set_index(["date", "permno"]).sort_index()
    return df


def merge_features_and_crsp(
    features: pd.DataFrame,
    crsp: pd.DataFrame,
    on: list = ["date", "permno"],
    lsuffix: str = "_feat",
    rsuffix: str = "_crsp",
) -> pd.DataFrame:
    """
    Inner join two DataFrames on MultiIndex (date, permno).
    Suffixes are applied to overlapping column names.
    """
    if not (
        isinstance(features.index, pd.MultiIndex)
        and isinstance(crsp.index, pd.MultiIndex)
    ):
        raise ValueError("Both inputs must be MultiIndex indexed by (date, permno)")

    merged = features.join(crsp, how="inner", lsuffix=lsuffix, rsuffix=rsuffix)

    if merged.empty:
        raise RuntimeError(
            "No overlapping (date,permno) between features and CRSP data."
        )

    return merged


if __name__ == "__main__":
    with tqdm(total=3, desc="Loading pipeline", ncols=80) as pbar:
        if VERBOSE:
            print("Loading datashare...")
        features = load_and_preprocess("datashare.csv", date_column="DATE")
        pbar.update(1)

        if VERBOSE:
            print("Loading local CRSP...")
        crsp = load_and_preprocess("crsp_returns.csv", date_column="date")
        pbar.update(1)

        if VERBOSE:
            print("Merging features and CRSP...")
        merged_data = merge_features_and_crsp(features, crsp)
        pbar.update(1)

    if VERBOSE:
        print("\nMerged data preview:")
    print(merged_data.head())
