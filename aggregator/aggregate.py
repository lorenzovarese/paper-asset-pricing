"""Core data-aggregation logic."""

from __future__ import annotations
import functools
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np
import yaml

from connectors.local.local_loader import load_and_preprocess
from connectors.local.local_loader import (
    _standardize_columns,
)

from .schema import (
    AggregationConfig,
    OneHotConfig,
    LagConfig,
    FillNaGroupedConfig,
    OutputConfig,
    SourceConfig,
    TransformationConfig,
)

# ---- helper registry ---------------------------------------------------- #


def _load_local(path: Path) -> pd.DataFrame:
    """
    Load a file, prioritizing `load_and_preprocess` with `DATA_DIR` logic,
    but falling back to direct load with subsequent standardization if needed.
    Ensures date columns are parsed consistently if possible.
    """
    try:
        # Try the custom loader first, assuming it handles dates correctly
        # or that its output will be re-checked.
        df = load_and_preprocess(path.name)
        # After load_and_preprocess, standardize columns and re-check date
        df = _standardize_columns(df)  # Standardize here as well
        date_column_name_std = "date"
        date_format_str = "%Y%m%d"

        if date_column_name_std in df.columns:
            # If it's already datetime, great.
            if pd.api.types.is_datetime64_any_dtype(df[date_column_name_std]):
                return df
            # If it's integer or object (string) that looks like YYYYMMDD
            elif pd.api.types.is_integer_dtype(df[date_column_name_std]) or (
                pd.api.types.is_object_dtype(df[date_column_name_std])
                and df[date_column_name_std].astype(str).str.match(r"^\d{8}$").all()
            ):
                try:
                    print(
                        f"Info: Attempting to parse integer/string date column '{date_column_name_std}' in '{path.name}' with format '{date_format_str}'."
                    )
                    df[date_column_name_std] = pd.to_datetime(
                        df[date_column_name_std], format=date_format_str
                    )
                except ValueError as e_fmt:
                    print(
                        f"Warning: Failed to parse '{date_column_name_std}' in '{path.name}' with format '{date_format_str}': {e_fmt}. Attempting inference."
                    )
                    try:
                        df[date_column_name_std] = pd.to_datetime(
                            df[date_column_name_std]
                        )  # General inference
                    except ValueError as e_infer:
                        print(
                            f"Error: Failed to parse date column '{date_column_name_std}' in '{path.name}' even with inference: {e_infer}. Date column may not be correctly processed."
                        )
            else:  # Other dtype, try general inference
                try:
                    print(
                        f"Info: Date column '{date_column_name_std}' in '{path.name}' is not datetime, integer, or YYYYMMDD string. Attempting general date inference."
                    )
                    df[date_column_name_std] = pd.to_datetime(df[date_column_name_std])
                except ValueError as e_general_infer:
                    print(
                        f"Error: Failed to parse date column '{date_column_name_std}' in '{path.name}' with general inference: {e_general_infer}. Date column may not be correctly processed."
                    )
        return df

    except FileNotFoundError:
        print(
            f"Info: load_and_preprocess failed for '{path.name}'. Falling back to direct CSV load for '{path}'."
        )
        df_fallback = pd.read_csv(path)  # Renamed to avoid conflict
        df_fallback = _standardize_columns(df_fallback)
        date_column_name_std = "date"
        date_format_str = "%Y%m%d"

        if date_column_name_std in df_fallback.columns:
            # Check if it's already datetime (e.g. if read_csv parsed it)
            if pd.api.types.is_datetime64_any_dtype(df_fallback[date_column_name_std]):
                return df_fallback  # Already correct
            # If it's integer (like YYYYMMDD) or a string of 8 digits
            elif pd.api.types.is_integer_dtype(df_fallback[date_column_name_std]) or (
                pd.api.types.is_object_dtype(df_fallback[date_column_name_std])
                and df_fallback[date_column_name_std]
                .astype(str)
                .str.match(r"^\d{8}$")
                .all()
            ):
                try:
                    print(
                        f"Info: Attempting to parse integer/string date column '{date_column_name_std}' in fallback load of '{path}' with format '{date_format_str}'."
                    )
                    df_fallback[date_column_name_std] = pd.to_datetime(
                        df_fallback[date_column_name_std], format=date_format_str
                    )
                except ValueError as e_fmt_fb:
                    print(
                        f"Warning: Failed to parse '{date_column_name_std}' in fallback load of '{path}' with format '{date_format_str}': {e_fmt_fb}. Attempting inference."
                    )
                    try:
                        df_fallback[date_column_name_std] = pd.to_datetime(
                            df_fallback[date_column_name_std]
                        )  # General inference
                    except ValueError as e_infer_fb:
                        print(
                            f"Error: Failed to parse date column '{date_column_name_std}' in fallback load of '{path}' even with inference: {e_infer_fb}. Date column may not be correctly processed."
                        )
            else:  # Other dtype, try general inference
                try:
                    print(
                        f"Info: Date column '{date_column_name_std}' in fallback load of '{path}' is not datetime, integer, or YYYYMMDD string. Attempting general date inference."
                    )
                    df_fallback[date_column_name_std] = pd.to_datetime(
                        df_fallback[date_column_name_std]
                    )
                except ValueError as e_general_infer_fb:
                    print(
                        f"Error: Failed to parse date column '{date_column_name_std}' in fallback load of '{path}' with general inference: {e_general_infer_fb}. Date column may not be correctly processed."
                    )
        return df_fallback


CONNECTOR_REGISTRY = {
    "local": _load_local,
}

# ---- public API ----------------------------------------------------------- #


class DataAggregator:
    """Transforms multiple raw data sources into a single modelling table."""

    def __init__(self, cfg: AggregationConfig) -> None:
        self.cfg = cfg
        self._frames: dict[str, pd.DataFrame] = {}

    def load(self) -> "DataAggregator":
        """Read every data source into memory."""
        for src_cfg in self.cfg.sources:
            loader = CONNECTOR_REGISTRY[src_cfg.connector]
            df = loader(src_cfg.path)
            if not isinstance(df.index, pd.RangeIndex):
                df = df.reset_index()
            df.columns = df.columns.str.lower()
            missing_keys = [
                col.lower() for col in src_cfg.join_on if col.lower() not in df.columns
            ]
            if missing_keys:
                raise KeyError(
                    f"Source '{src_cfg.name}' ({src_cfg.path}) is missing join key(s): {missing_keys}. "
                    f"Available columns after lowercasing: {list(df.columns)[:10]}..."
                )
            self._frames[src_cfg.name] = df
        return self

    def merge(self) -> pd.DataFrame:
        """
        Merges firm-level sources first, then merges macro-level sources onto the combined firm data.
        """
        if not self._frames:
            raise RuntimeError("Call load() first.")

        firm_source_configs = [s for s in self.cfg.sources if s.level == "firm"]
        macro_source_configs = [s for s in self.cfg.sources if s.level == "macro"]

        if not firm_source_configs:
            if macro_source_configs:  # If only macro sources, merge them if more than one, else return the single one
                print(
                    "Warning: No firm-level sources found. Merging macro sources only."
                )
                base_df = self._frames[macro_source_configs[0].name]
                for i in range(1, len(macro_source_configs)):
                    macro_cfg = macro_source_configs[i]
                    right_df = self._frames[macro_cfg.name]
                    keys = [col.lower() for col in macro_cfg.join_on]
                    base_df = base_df.merge(right_df, how="left", on=keys)
                return base_df
            else:
                raise ValueError("No sources configured to merge.")

        # Start with the first firm-level source as the base
        current_base_cfg = firm_source_configs[0]
        base_df = self._frames[current_base_cfg.name]
        print(f"Starting merge with base firm-level source: {current_base_cfg.name}")

        # Merge remaining firm-level sources
        for i in range(1, len(firm_source_configs)):
            firm_cfg = firm_source_configs[i]
            right_df = self._frames[firm_cfg.name]
            keys = [col.lower() for col in firm_cfg.join_on]
            print(f"Merging firm-level source: {firm_cfg.name} on keys: {keys}")
            base_df = base_df.merge(right_df, how="left", on=keys)

        # Merge macro-level sources
        for macro_cfg in macro_source_configs:
            right_df = self._frames[macro_cfg.name]
            keys = [col.lower() for col in macro_cfg.join_on]
            print(f"Merging macro-level source: {macro_cfg.name} on keys: {keys}")
            base_df = base_df.merge(right_df, how="left", on=keys)

        return base_df

    def apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sequentially apply every transformation from the YAML spec."""
        if not self.cfg.transformations:
            return df
        return functools.reduce(self._apply_one, self.cfg.transformations, df)

    @staticmethod
    def _apply_one(df: pd.DataFrame, tr: TransformationConfig) -> pd.DataFrame:
        if isinstance(tr, OneHotConfig):
            dummies = pd.get_dummies(df[tr.column], prefix=tr.prefix, dtype=int)
            df = pd.concat([df, dummies], axis=1)
            if tr.drop_original:
                df = df.drop(columns=tr.column)
        elif isinstance(tr, LagConfig):
            for col in tr.columns:
                # Ensure column exists before trying to lag
                if col in df.columns:
                    df[f"{col}_lag{tr.periods}"] = (
                        df.groupby(level=0)[col].shift(tr.periods)
                        if isinstance(df.index, pd.MultiIndex)
                        else df[col].shift(tr.periods)
                    )
                else:
                    print(f"Warning: Column '{col}' not found for lagging. Skipping.")
        elif isinstance(tr, FillNaGroupedConfig):
            if tr.group_by_column not in df.columns:
                print(
                    f"Warning: Group_by_column '{tr.group_by_column}' for fillna_grouped not found. Skipping."
                )
                return df

            target_cols_for_fill = []
            if tr.columns:
                for col_name in tr.columns:
                    if col_name not in df.columns:
                        print(
                            f"Warning: Specified column '{col_name}' for fillna_grouped not found. Skipping this column."
                        )
                        continue
                    if not pd.api.types.is_numeric_dtype(df[col_name]):
                        print(
                            f"Warning: Specified column '{col_name}' for fillna_grouped is not numeric. Skipping this column."
                        )
                        continue
                    target_cols_for_fill.append(col_name)
            else:
                target_cols_for_fill = df.select_dtypes(
                    include=np.number
                ).columns.tolist()

            if not target_cols_for_fill:
                print(
                    "Warning: No numeric columns to process for fillna_grouped. Skipping."
                )
                return df

            print(
                f"Applying fillna_grouped (method: {tr.method}) on columns: {target_cols_for_fill} grouped by '{tr.group_by_column}'"
            )

            if tr.method == "median":
                df[target_cols_for_fill] = df.groupby(tr.group_by_column)[
                    target_cols_for_fill
                ].transform(lambda x: x.fillna(x.median()))
            elif tr.method == "mean":
                df[target_cols_for_fill] = df.groupby(tr.group_by_column)[
                    target_cols_for_fill
                ].transform(lambda x: x.fillna(x.mean()))
        return df


def aggregate_from_yaml(
    spec_path: str | Path,
) -> tuple[pd.DataFrame, AggregationConfig]:
    """One-shot helper: parse YAML → load → merge → transform → DataFrame."""
    with open(spec_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = AggregationConfig.model_validate(raw)
    agg = DataAggregator(cfg).load()
    merged = agg.merge()
    transformed_df = agg.apply_transformations(merged)
    return transformed_df, cfg
