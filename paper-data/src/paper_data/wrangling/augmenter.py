"""Data augmentation library for paper data."""

from typing import Literal
import polars as pl


def merge_datasets(
    left_df: pl.DataFrame,
    right_df: pl.DataFrame,
    on_cols: list[str],
    how: Literal["left", "inner", "outer", "full"] = "left",
) -> pl.DataFrame:
    """
    Merges two Polars DataFrames.

    Args:
        left_df: The left DataFrame.
        right_df: The right DataFrame.
        on_cols: A list of column names to merge on.
        how: The type of join (e.g., "left", "inner", "outer", "full").

    Returns:
        The merged Polars DataFrame.
    """
    print(f"Merging datasets on columns: {on_cols} with how='{how}'")
    merged_df = left_df.join(right_df, on=on_cols, how=how)
    print(f"Merge complete. Resulting shape: {merged_df.shape}")
    return merged_df


def lag_columns(
    df: pl.DataFrame,
    date_col: str,
    id_col: str | None,
    cols_to_lag: list[str],
    periods: int,
    drop_after_lag: bool,
) -> pl.DataFrame:
    """
    Lags or leads specified columns in a Polars DataFrame.

    If an `id_col` is provided, the lagging is performed within each group
    defined by `id_col`, ordered by `date_col`. Otherwise, it's a simple
    time-series lag ordered by `date_col`.

    Args:
        df: The input Polars DataFrame.
        date_col: The name of the date column to order by.
        id_col: The name of the identifier column for panel data lagging (e.g., 'permco').
                If None, a simple time-series lag is performed.
        cols_to_lag: A list of column names to apply the lag/lead operation to.
        periods: The number of periods to shift.
                 Positive for lagging (e.g., 1 for previous period's value).
                 Negative for leading (e.g., -1 for next period's value).
        drop_after_lag: If True, the original (unlagged/unled) columns are dropped
                        from the output DataFrame.

    Returns:
        A new Polars DataFrame with the lagged/led columns.
    """
    if not cols_to_lag:
        print("No columns specified for lagging. Returning original DataFrame.")
        return df

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")
    if not isinstance(df[date_col].dtype, (pl.Date, pl.Datetime)):
        raise ValueError(
            f"Date column '{date_col}' must be of Date or Datetime type for lagging."
        )

    # Validate that all specified columns exist in the DataFrame
    missing_cols = [c for c in cols_to_lag if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Columns specified for lagging not found in DataFrame: {missing_cols}"
        )

    out_df = df.clone()
    suffix = f"_lag_{periods}" if periods > 0 else f"_lead_{abs(periods)}"

    print(
        f"Applying {'lag' if periods > 0 else 'lead'} of {abs(periods)} periods to columns: {cols_to_lag}"
    )

    expressions = []
    for col in cols_to_lag:
        if id_col:
            if id_col not in df.columns:
                raise ValueError(
                    f"Identifier column '{id_col}' not found in DataFrame for panel lagging."
                )
            # Lag within each group defined by id_col, ordered by date_col
            expr = pl.col(col).shift(periods).over(id_col).alias(f"{col}{suffix}")
        else:
            # Simple time-series lag, ordered by date_col
            expr = pl.col(col).shift(periods).alias(f"{col}{suffix}")
        expressions.append(expr)

    # Ensure the DataFrame is sorted by id_col (if present) and date_col before applying shift
    if id_col:
        out_df = out_df.sort([id_col, date_col])
    else:
        out_df = out_df.sort(date_col)

    out_df = out_df.with_columns(expressions)

    if drop_after_lag:
        print(f"Dropping original columns after lagging: {cols_to_lag}")
        out_df = out_df.drop(cols_to_lag)

    print(f"Lag operation complete. Resulting shape: {out_df.shape}")
    return out_df
