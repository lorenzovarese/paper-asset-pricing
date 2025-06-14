"""Data cleaning library for paper data."""

from __future__ import annotations

import polars as pl
import logging

logger = logging.getLogger(__name__)


def _log_null_count(df: pl.DataFrame, col_name: str) -> None:
    """
    Prints the number of null values in a specified column of a Polars DataFrame.

    This is a helper function primarily for logging/debugging purposes within
    data cleaning routines.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to inspect.
    col_name : str
        The name of the column for which to count and print nulls.
    """
    if col_name not in df.columns:
        logger.warning(
            f"  Warning: Column '{col_name}' not found in DataFrame. Skipping null count."
        )
        return

    null_count = df[col_name].null_count()
    if null_count > 0:
        logger.info(f"  Column '{col_name}' has {null_count} nulls to fill.")
    else:
        logger.info(f"  Column '{col_name}' has no nulls to fill.")


def impute_monthly(
    df: pl.DataFrame,
    date_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    fallback_to_zero: bool = False,
) -> pl.DataFrame:
    """
    Monthly imputation for Polars DataFrame:
      • numeric columns  → cross-sectional median
      • categorical cols → cross-sectional mode (first mode if ties)
    If `fallback_to_zero` is True, any remaining NaNs after the primary
    imputation (due to all-NaN months) are filled with 0.

    Parameters
    ----------
    df        : Polars DataFrame with a date column (pl.Date or pl.Datetime)
    date_col  : Name of the date column in the DataFrame.
    numeric_cols  : list of numeric columns to fill with medians.
    categorical_cols : list of categorical / discrete columns to fill with modes.
    fallback_to_zero : bool, default False
                       If True, fills remaining nulls with 0.

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

    # Check for all-NaN groups before imputation, but only if not using the fallback.
    if not fallback_to_zero:
        for col in all_impute_cols:
            null_check_df = (
                out.lazy()
                .group_by(month_key_expr)
                .agg(pl.col(col).drop_nulls().count().alias("non_null_count"))
                .filter(pl.col("non_null_count") == 0)
                .collect()
            )

            if not null_check_df.is_empty():
                problematic_months = null_check_df["month_key"].to_list()
                raise ValueError(
                    f"Column '{col}' has all NaNs/nulls in month(s): {list(problematic_months)}. "
                    "Cannot impute. Consider using the 'fallback_to_zero: true' option in your config."
                )

    # Apply numeric imputation (median)
    if numeric_cols:
        logger.info(f"Imputing numeric columns by monthly median: {numeric_cols}")
        for col in numeric_cols:
            _log_null_count(out, col)
            impute_expr = pl.col(col).fill_null(
                pl.col(col).median().over(month_key_expr)
            )
            if fallback_to_zero:
                impute_expr = impute_expr.fill_null(0)
            out = out.with_columns(impute_expr)

    # Apply categorical imputation (mode)
    if categorical_cols:
        logger.info(
            f"Imputing categorical columns by monthly mode: {categorical_cols}"
        )
        for col in categorical_cols:
            _log_null_count(out, col)
            impute_expr = pl.col(col).fill_null(
                pl.col(col).mode().first().over(month_key_expr)
            )
            if fallback_to_zero:
                impute_expr = impute_expr.fill_null(0)
            out = out.with_columns(impute_expr)

    return out


def scale_to_range(
    df: pl.DataFrame,
    cols: list[str],
    date_col: str,
    target_min: float,
    target_max: float,
) -> pl.DataFrame:
    """
    Cross-sectional min-max scaling to a specified interval [target_min, target_max] within each month:
        scaled = target_min + (x - min) * (target_max - target_min) / (max - min)

    Handles cases where (max - min) is zero by setting scaled values to the midpoint of the target range.
    Truncates to 8 decimal places.

    Parameters
    ----------
    df         : polars.DataFrame
                 Must contain a date column (pl.Date or pl.Datetime) named `date_col`.
    cols       : list[str]
                 Numeric columns to transform.
    date_col   : str
                 Name of the date column in the DataFrame used for monthly grouping.
    target_min : float
                 The lower bound of the target scaling range.
    target_max : float
                 The upper bound of the target scaling range.

    Returns
    -------
    polars.DataFrame
           New dataframe; values in *cols* are replaced by scaled values.
           The original *df* is left untouched.
    """
    out = df.clone()

    if date_col not in out.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")
    if not isinstance(out[date_col].dtype, (pl.Date, pl.Datetime)):
        raise ValueError(
            f"Date column '{date_col}' must be of Date or Datetime type for monthly scaling."
        )

    # Define the month key expression
    month_key_expr = pl.col(date_col).dt.month_start()

    # Validate that all specified columns exist in the DataFrame
    missing_cols = [c for c in cols if c not in out.columns]
    if missing_cols:
        raise ValueError(
            f"Columns specified for scaling not found in DataFrame: {missing_cols}"
        )

    # Calculate the range span (target_max - target_min)
    target_range_span = target_max - target_min
    target_midpoint = (target_min + target_max) / 2.0

    logger.info(f"Scaling columns {cols} to range [{target_min}, {target_max}]...")

    expressions = []
    for col in cols:
        # Explicitly cast to Float64 to prevent arithmetic errors on string types.
        # `strict=False` will turn uncastable values (e.g., non-numeric strings) into nulls.
        col_expr = pl.col(col).cast(pl.Float64, strict=False)

        # Calculate min and max within each month group using the numeric expression
        min_val = col_expr.min().over(month_key_expr)
        max_val = col_expr.max().over(month_key_expr)

        # Calculate the actual range for the current month and column
        actual_range = max_val - min_val

        # Define the scaling expression
        scaled_expression = (
            pl.when(actual_range == 0)  # If min == max within the group
            .then(
                pl.lit(target_midpoint, dtype=pl.Float64)
            )  # Set to midpoint of target range
            .otherwise(
                target_min + (col_expr - min_val) * target_range_span / actual_range
            )
            .clip(
                target_min, target_max
            )  # Clip to ensure values are within bounds after calculation
            .round(8)  # Round to 8 decimal places
            .alias(col)
        )
        expressions.append(scaled_expression)

    # Apply all scaling expressions
    out = out.with_columns(expressions)

    return out
