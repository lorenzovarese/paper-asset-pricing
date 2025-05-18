"""Data cleaning library for paper data."""

from __future__ import annotations

import polars as pl


def impute_monthly(
    df: pl.DataFrame,
    date_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pl.DataFrame:
    """
    Monthly imputation for Polars DataFrame:
      • numeric columns  → cross-sectional median
      • categorical cols → cross-sectional mode (first mode if ties)

    Parameters
    ----------
    df        : Polars DataFrame with a date column (pl.Date or pl.Datetime)
    date_col  : Name of the date column in the DataFrame.
    numeric_cols  : list of numeric columns to fill with medians.
    categorical_cols : list of categorical / discrete columns to fill with modes.

    Returns
    -------
    pl.DataFrame  (copy; original df remains unchanged)
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")
    if not isinstance(df[date_col].dtype, (pl.Date, pl.Datetime)):
        raise ValueError(
            f"Date column '{date_col}' must be of Date or Datetime type for monthly imputation."
        )

    out = df.clone()
    month_key_expr = pl.col(date_col).dt.month_start().alias("month_key")

    # Validate that all specified columns exist in the DataFrame
    all_impute_cols = numeric_cols + categorical_cols
    missing_cols = [c for c in all_impute_cols if c not in out.columns]
    if missing_cols:
        raise ValueError(
            f"Columns specified for imputation not found in DataFrame: {missing_cols}"
        )

    # Check for all-NaN groups before imputation
    for col in numeric_cols + categorical_cols:
        # Use a lazy frame for efficient group-by and aggregation
        null_check_df = (
            out.lazy()
            .group_by(month_key_expr)
            .agg(pl.col(col).drop_nulls().count().alias("non_null_count"))
            .filter(pl.col("non_null_count") == 0)
            .collect()
        )  # Collect to check if empty

        if not null_check_df.is_empty():
            problematic_months = null_check_df["month_key"].to_list()
            raise ValueError(
                f"Column '{col}' has all NaNs/nulls in month(s): {list(problematic_months)}. Cannot impute."
            )

    # Apply numeric imputation (median)
    if numeric_cols:
        print(f"Imputing numeric columns by monthly median: {numeric_cols}")
        for col in numeric_cols:
            null_count = out[col].null_count()
            if null_count > 0:
                print(f"Column '{col}' has {null_count} nulls to fill.")
            else:
                print(f"Column '{col}' has no nulls to fill.")
            out = out.with_columns(
                pl.col(col).fill_null(pl.col(col).median().over(month_key_expr))
            )

    # Apply categorical imputation (mode)
    if categorical_cols:
        print(f"Imputing categorical columns by monthly mode: {categorical_cols}")
        for col in categorical_cols:
            out = out.with_columns(
                pl.col(col).fill_null(pl.col(col).mode().first().over(month_key_expr))
            )

    return out
