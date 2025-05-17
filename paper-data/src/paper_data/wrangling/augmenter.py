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
