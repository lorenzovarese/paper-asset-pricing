"""
Custom transformation script to construct macroeconomic predictors based on the
methodology of Welch & Goyal (2024).

This script is designed to be called from the 'run_script' operation
in the paper-data pipeline.
"""

from __future__ import annotations
import polars as pl


def transform(df: pl.DataFrame) -> pl.DataFrame:
    """
    Takes the raw monthly macroeconomic data and constructs the final predictors.

    Args:
        df: A Polars DataFrame containing the raw data ingested from the
            Welch & Goyal Google Sheet. Column names are expected to be lowercase.

    Returns:
        A Polars DataFrame containing the calculated predictors, ready for export.
    """
    # Ensure the input is a Polars DataFrame
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Input must be a Polars DataFrame, but got {type(df)}")

    # --- 1. Data Cleaning and Preparation ---
    # The 'to_lowercase_cols: true' in the config handles column naming.
    # The date parsing is also handled by the ingestion step.
    # We just need to ensure the date is set to the end of the month.
    df = df.with_columns(pl.col("yyyymm").dt.month_end().alias("date")).drop("yyyymm")

    # --- 2. Predictor Construction ---
    # The '_macro' suffix is added to some predictors (ep and bm) to avoid potential name clashes
    # with firm-level characteristics later in the pipeline.
    predictors = df.select(
        pl.col("date"),
        # Dividend-Price Ratio (dp)
        (pl.col("d12") / pl.col("index")).log().alias("dp"),
        # Earnings-Price Ratio (ep)
        (pl.col("e12") / pl.col("index")).log().alias("ep_macro"),
        # Book-to-Market Ratio (bm)
        pl.col("b/m").alias("bm_macro"),
        # Net T-Bill Issuance (ntis)
        pl.col("ntis"),
        # Treasury Bill Rate (tbl)
        pl.col("tbl"),
        # Term Spread (tms)
        (pl.col("lty") - pl.col("tbl")).alias("tms"),
        # Default Yield Spread (dfy)
        (pl.col("baa") - pl.col("aaa")).alias("dfy"),
        # Stock Variance (svar)
        pl.col("svar"),
    )

    # --- 3. Filtering and Sorting ---
    # Filter for the desired date range, as in the original script.
    final_df = predictors.filter(
        pl.col("date").is_between(
            pl.date(1956, 12, 1), pl.date(2021, 12, 31), closed="both"
        )
    ).sort("date")

    return final_df
