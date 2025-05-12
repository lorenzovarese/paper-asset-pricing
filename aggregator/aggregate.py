"""Core data-aggregation logic."""

from __future__ import annotations
import functools
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

from connectors.local.local_loader import load_and_preprocess
from connectors.local.local_loader import (
    _standardize_columns,
)  # Import the standardization helper

from .schema import (
    AggregationConfig,
    TransformationConfig,
    OneHotConfig,
    FillNaConfig,
    LagConfig,
)

# ---- helper registry ---------------------------------------------------- #


def _load_local(path: Path) -> pd.DataFrame:  # noqa: D401
    """
    Load a file, prioritizing `load_and_preprocess` with `DATA_DIR` logic,
    but falling back to direct load with subsequent standardization if needed.
    Ensures date columns are parsed consistently if possible.
    """
    try:
        # Assumes load_and_preprocess uses DATA_DIR and path.name is the filename.
        # Default date_column="date", date_format="%Y%m%d" are used by load_and_preprocess.
        return load_and_preprocess(path.name)
    except FileNotFoundError:
        # Fallback: Load directly using the full path from YAML.
        # This might happen if DATA_DIR is not set up as expected by load_and_preprocess
        # or if the YAML path is absolute/outside DATA_DIR.
        print(
            f"Info: load_and_preprocess failed for '{path.name}'. Falling back to direct load for '{path}'."
        )
        df = pd.read_csv(path)
        df = _standardize_columns(df)  # Apply column name standardization.

        # Attempt to apply the same date parsing logic as in load_and_preprocess.
        # These are the defaults in load_and_preprocess.
        date_column_name_std = "date"  # Standardized (lowercase) column name
        date_format_str = "%Y%m%d"  # Default format used by load_and_preprocess

        if date_column_name_std in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[date_column_name_std]):
                try:
                    df[date_column_name_std] = pd.to_datetime(
                        df[date_column_name_std], format=date_format_str
                    )
                except ValueError:
                    # If specific format fails, try to infer. This handles cases where
                    # the date might be in a different string format (e.g., YYYY-MM-DD).
                    print(
                        f"Warning: Could not parse date column '{date_column_name_std}' in '{path}' with format '{date_format_str}'. Attempting to infer format."
                    )
                    try:
                        df[date_column_name_std] = pd.to_datetime(
                            df[date_column_name_std]
                        )
                    except ValueError as e_infer:
                        print(
                            f"Error: Failed to parse date column '{date_column_name_std}' in '{path}' even with inference: {e_infer}. Date column may not be correctly processed."
                        )
                        # Depending on requirements, you might want to raise an error here.
            # If already datetime, do nothing.
        # The indexing part of load_and_preprocess (set_index) is not replicated here,
        # as DataAggregator.load() will call reset_index() if a non-default index is set.
        return df


CONNECTOR_REGISTRY = {
    "local": _load_local,
    # "wrds": load_wrds, …
}

# ---- public API ----------------------------------------------------------- #


class DataAggregator:
    """Transforms multiple raw data sources into a single modelling table.

    Users build an instance from a YAML file via :pyfunc:`aggregate_from_yaml`.
    """

    def __init__(self, cfg: AggregationConfig) -> None:
        self.cfg = cfg
        self._frames: dict[str, pd.DataFrame] = {}

    # --------------------------------------------------------------------- #
    # loading & merging
    # --------------------------------------------------------------------- #
    def load(self) -> "DataAggregator":
        """Read every data source into memory."""
        for src in self.cfg.sources:
            loader = CONNECTOR_REGISTRY[src.connector]
            # load once and normalize column names for case-insensitive joins
            df = loader(src.path)

            # If the loader returned a DataFrame with a non-default index
            # (e.g., DateIndex or MultiIndex set by load_and_preprocess),
            # reset it to ensure join keys are columns for subsequent operations.
            if not isinstance(df.index, pd.RangeIndex):
                df = df.reset_index()

            # normalize all column names to lowercase
            df.columns = df.columns.str.lower()

            # ensure every join key exists
            missing = [
                col.lower() for col in src.join_on if col.lower() not in df.columns
            ]

            if missing:
                raise KeyError(
                    f"Source '{src.name}' ({src.path}) does not contain "
                    f"join key(s) {missing}. "
                    f"Available columns: {list(df.columns)[:10]} ..."
                )

            self._frames[src.name] = df
        return self

    def merge(self) -> pd.DataFrame:
        """Left-fold merge on the key columns declared in YAML."""
        if not self._frames:
            raise RuntimeError("Call load() first.")
        it = iter(self.cfg.sources)
        base = self._frames[next(it).name]
        for src in it:
            right = self._frames[src.name]
            # perform merge on the same lowercase join keys
            keys = [col.lower() for col in src.join_on]
            base = base.merge(right, how="left", on=keys)

        return base

    # --------------------------------------------------------------------- #
    # transformations
    # --------------------------------------------------------------------- #
    def apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sequentially apply every transformation from the YAML spec."""
        if not self.cfg.transformations:  # Handle case with no transformations
            return df
        return functools.reduce(self._apply_one, self.cfg.transformations, df)

    # private ----------------------------------------------------------------
    @staticmethod
    def _apply_one(df: pd.DataFrame, tr: TransformationConfig) -> pd.DataFrame:
        if isinstance(tr, OneHotConfig):
            dummies = pd.get_dummies(df[tr.column], prefix=tr.prefix, dtype=int)
            df = pd.concat([df, dummies], axis=1)
            if tr.drop_original:
                df = df.drop(columns=tr.column)
        elif isinstance(tr, FillNaConfig):
            if tr.method in {"ffill", "bfill"}:
                df = df.fillna(
                    method=tr.method
                )  # Applies to all columns, generally okay for ffill/bfill
            else:
                # For methods like 'mean', 'median', etc., apply only to numeric columns
                # to avoid TypeError on non-numeric data.
                numeric_cols = df.select_dtypes(include=np.number).columns
                if not numeric_cols.empty:
                    fill_values = getattr(
                        df[numeric_cols], tr.method
                    )()  # Calculate mean/median etc. only on numeric columns
                    df[numeric_cols] = df[numeric_cols].fillna(fill_values)
                # Non-numeric columns will retain their NaNs if not handled by ffill/bfill
                # or a more specific fillna configuration for them.
        elif isinstance(tr, LagConfig):
            for col in tr.columns:
                df[f"{col}_lag{tr.periods}"] = df[col].shift(tr.periods)
        return df


def aggregate_from_yaml(
    spec_path: str | Path,
) -> tuple[pd.DataFrame, AggregationConfig]:
    """One-shot helper: parse YAML → load → merge → transform → DataFrame.
    Returns the aggregated DataFrame and the parsed configuration.
    """
    with open(spec_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = AggregationConfig.model_validate(raw)
    agg = DataAggregator(cfg).load()
    merged = agg.merge()
    transformed_df = agg.apply_transformations(merged)
    return transformed_df, cfg
