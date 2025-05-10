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


def load_and_preprocess(
    filename: str, date_column: str = "date", date_format: str = "%Y%m%d"
) -> pd.DataFrame:
    """
    Load and preprocess a CSV file:
      - Reads the file from the specified path.
      - Standardizes column names (trims whitespace, converts to lowercase).
      - Parses the specified `date_column` as datetime using the given format.
      - Converts the 'permno' column to integer type.
      - Sets a MultiIndex with levels ('date', 'permno') and sorts the index.

    Args:
        filename (str): Name of the CSV file to load.
        date_column (str): Name of the column containing date information (default: "date").
        date_format (str): Format of the date in the `date_column` (default: "%Y%m%d").

    Returns:
        pd.DataFrame: Preprocessed DataFrame with a MultiIndex of ('date', 'permno').

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the `date_column` is not found after standardization.
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

    df[date_column.lower()] = pd.to_datetime(
        df[date_column.lower()], format=date_format
    )
    df["permno"] = df["permno"].astype(int)
    df = df.set_index(["date", "permno"]).sort_index()
    return df


def print_date_stats(df: pd.DataFrame, name: str) -> None:
    """
    Print basic statistics for the 'date' level of a MultiIndex DataFrame:
      - earliest date
      - latest date
      - total number of rows
      - number of unique year-month periods
    """
    dates = df.index.get_level_values("date")
    earliest = dates.min().date()
    latest = dates.max().date()
    total = len(dates)
    unique_m = dates.to_series().dt.to_period("M").nunique()
    print(
        f"{name} → min date: {earliest}, max date: {latest}, rows: {total}, unique YM: {unique_m}"
    )


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
    # Run from root with: uv run -m connectors.local.local_loader
    with tqdm(total=3, desc="Loading pipeline", ncols=80) as pbar:
        if VERBOSE:
            print("Loading datashare...")
        features = load_and_preprocess("datashare.csv", date_column="DATE")
        if VERBOSE:
            print("Datashare data loaded.")
            print_date_stats(features, "Datashare")
        pbar.update(1)

        if VERBOSE:
            print("Loading local CRSP...")
        crsp = load_and_preprocess("crsp_returns.csv", date_column="date")
        if VERBOSE:
            print("CRSP data loaded.")
            print_date_stats(crsp, "CRSP")
        pbar.update(1)

        if VERBOSE:
            print("Merging features and CRSP...")
        merged_data = merge_features_and_crsp(features, crsp)
        if VERBOSE:
            print("Merged data loaded.")
            print_date_stats(merged_data, "Merged Data")
        pbar.update(1)

    if VERBOSE:
        print("\nMerged data preview:")
    print(merged_data.head())
