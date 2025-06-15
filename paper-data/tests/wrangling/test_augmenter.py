import polars as pl
from datetime import date
from paper_data.wrangling.augmenter import merge_datasets, lag_columns  # type: ignore


def test_merge():
    df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pl.DataFrame({"a": [1], "c": [5]})
    merged = merge_datasets(df1, df2, on_cols=["a"], how="left")
    assert "c" in merged.columns


def test_lag_time_series():
    df = pl.DataFrame({"date": [date(2020, 1, 31), date(2020, 2, 29)], "x": [10, 20]})
    lagged = lag_columns(df, "date", None, ["x"], 1, False, False, False)
    assert "x_lag_1" in lagged.columns
