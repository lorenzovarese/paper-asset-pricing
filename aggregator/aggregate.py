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
    TransformationConfig,
)


# ---- helper registry ---------------------------------------------------- #
# ... (rest of _load_local and CONNECTOR_REGISTRY remains the same) ...
def _load_local(path: Path) -> pd.DataFrame:
    """
    Load a file, prioritizing `load_and_preprocess` with `DATA_DIR` logic,
    but falling back to direct load with subsequent standardization if needed.
    Ensures date columns are parsed consistently if possible.
    """
    try:
        # Try the custom loader first, assuming it handles dates correctly
        # or that its output will be re-checked.
        df = load_and_preprocess(path.name)  # Assuming DATA_DIR is set correctly
        # After load_and_preprocess, standardize columns and re-check date
        df = _standardize_columns(df)  # Standardize here as well
        date_column_name_std = "date"
        date_format_str = "%Y%m%d"

        if date_column_name_std in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[date_column_name_std]):
                return df
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
                        )
                    except ValueError as e_infer:
                        print(
                            f"Error: Failed to parse date column '{date_column_name_std}' in '{path.name}' even with inference: {e_infer}. Date column may not be correctly processed."
                        )
            else:
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

    except FileNotFoundError:  # This will catch if load_and_preprocess fails due to DATA_DIR issue or file not found
        print(
            f"Info: load_and_preprocess failed for '{path.name}'. Falling back to direct CSV load for '{path}'."
        )
        # Ensure path is absolute or relative to current working directory for fallback
        df_fallback = pd.read_csv(path)
        df_fallback = _standardize_columns(df_fallback)
        date_column_name_std = "date"
        date_format_str = "%Y%m%d"

        if date_column_name_std in df_fallback.columns:
            if pd.api.types.is_datetime64_any_dtype(df_fallback[date_column_name_std]):
                return df_fallback
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
                        )
                    except ValueError as e_infer_fb:
                        print(
                            f"Error: Failed to parse date column '{date_column_name_std}' in fallback load of '{path}' even with inference: {e_infer_fb}. Date column may not be correctly processed."
                        )
            else:
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
            # Ensure path is resolved correctly, especially if relative paths are used in YAML
            # Assuming paths in YAML are relative to where the script is run or a defined base path
            # For simplicity, let's assume Path(src_cfg.path) works as intended.
            # If DATA_DIR is used by load_and_preprocess, src_cfg.path should be just the filename.
            # If src_cfg.path is absolute, it should work.
            # If src_cfg.path is relative, it's relative to CWD.
            # The _load_local function handles path.name for load_and_preprocess and path for fallback.

            # Let's assume src_cfg.path is a Path object already from Pydantic
            # If it's a string, Path(src_cfg.path) is fine.
            df = loader(Path(src_cfg.path))

            if not isinstance(df.index, pd.RangeIndex):
                df = df.reset_index()  # Ensure join keys are columns
            df.columns = df.columns.str.lower()

            # Standardize join_on keys to lower case for checking
            join_on_lower = [col.lower() for col in src_cfg.join_on]

            missing_keys = [col for col in join_on_lower if col not in df.columns]
            if missing_keys:
                raise KeyError(
                    f"Source '{src_cfg.name}' ({src_cfg.path}) is missing join key(s): {missing_keys}. "
                    f"Available columns after lowercasing and reset_index: {list(df.columns)[:10]}..."
                )
            self._frames[src_cfg.name] = df
        print("Stop here")
        return self

    def merge(self) -> pd.DataFrame:
        """
        Merges firm-level sources based on a primary base, then merges macro-level sources.
        """
        if not self._frames:
            raise RuntimeError("Call load() first.")

        firm_source_configs = [s for s in self.cfg.sources if s.level == "firm"]
        macro_source_configs = [s for s in self.cfg.sources if s.level == "macro"]

        base_df = None

        if not firm_source_configs:
            if macro_source_configs:
                print(
                    "Warning: No firm-level sources found. Merging macro sources only."
                )
                if not self._frames:  # Should be caught by the first check
                    raise RuntimeError("Call load() first.")

                base_df = self._frames[macro_source_configs[0].name].copy()
                print(
                    f"Starting merge with base macro-level source: {macro_source_configs[0].name}"
                )
                for i in range(1, len(macro_source_configs)):
                    macro_cfg = macro_source_configs[i]
                    right_df = self._frames[macro_cfg.name]
                    keys = [col.lower() for col in macro_cfg.join_on]
                    print(
                        f"Merging macro-level source: {macro_cfg.name} on keys: {keys}"
                    )
                    base_df = base_df.merge(right_df, how="left", on=keys)
                return base_df
            else:
                raise ValueError("No sources configured to merge.")

        # Determine the primary firm-level base DataFrame
        primary_firm_config = None
        primary_firm_sources_marked = [
            s_cfg for s_cfg in firm_source_configs if s_cfg.is_primary_firm_base
        ]

        # Validator in AggregationConfig already ensures len(primary_firm_sources_marked) <= 1
        if len(primary_firm_sources_marked) == 1:
            primary_firm_config = primary_firm_sources_marked[0]
            print(
                f"Identified explicit primary firm-level source: {primary_firm_config.name}"
            )
        elif len(firm_source_configs) == 1:
            # If only one firm source, it's implicitly the primary base
            primary_firm_config = firm_source_configs[0]
            print(
                f"Using the only firm-level source as de-facto primary: {primary_firm_config.name}"
            )
        else:  # Multiple firm sources, none marked as primary
            raise ValueError(
                "Multiple firm-level sources are provided. Please specify exactly one as the primary base "
                "by setting 'is_primary_firm_base: True' for that source in the YAML configuration."
            )

        base_df = self._frames[primary_firm_config.name].copy()
        print(
            f"Starting merge with primary firm-level source: {primary_firm_config.name}"
        )

        # Merge other (non-primary) firm-level sources onto the primary base using a left join
        other_firm_source_configs = [
            s_cfg
            for s_cfg in firm_source_configs
            if s_cfg.name != primary_firm_config.name
        ]

        for firm_cfg in other_firm_source_configs:
            right_df = self._frames[firm_cfg.name]
            keys = [col.lower() for col in firm_cfg.join_on]
            print(
                f"Left-merging firm-level source: {firm_cfg.name} onto base on keys: {keys}"
            )
            base_df = base_df.merge(right_df, how="left", on=keys)
            # Pandas merge handles suffixes (_x, _y) for overlapping non-key columns by default.

        # Merge macro-level sources onto the (now combined) firm base_df
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
            # Ensure column exists before trying to one-hot encode
            if tr.column.lower() not in df.columns:
                print(
                    f"Warning: Column '{tr.column}' for one-hot encoding not found. Skipping."
                )
                return df
            dummies = pd.get_dummies(
                df[tr.column.lower()], prefix=tr.prefix, dtype=int
            )  # Use lowercased column name
            df = pd.concat([df, dummies], axis=1)
            if tr.drop_original:
                df = df.drop(columns=tr.column.lower())  # Use lowercased column name
        elif isinstance(tr, LagConfig):
            # Ensure level=0 (permno if MultiIndex) is handled correctly if index was set before merge
            # Current load() resets index, so groupby(level=0) might not be applicable here.
            # Lagging should ideally happen on a per-entity basis if 'permno' is present.
            # For now, assuming simple shift on the whole column.
            # If 'permno' is a column, we should group by 'permno' before lagging.

            # Standardize column names for lagging
            lag_cols_lower = [col.lower() for col in tr.columns]

            # Check if 'permno' exists for grouped lagging
            group_keys = []
            if "permno" in df.columns:  # Assuming 'permno' is the firm identifier
                group_keys.append("permno")

            for col_to_lag in lag_cols_lower:
                if col_to_lag not in df.columns:
                    print(
                        f"Warning: Column '{col_to_lag}' not found for lagging. Skipping."
                    )
                    continue

                new_lag_col_name = f"{col_to_lag}_lag{tr.periods}"
                if group_keys:
                    print(
                        f"Applying grouped lag on '{col_to_lag}' by {group_keys} for {tr.periods} periods."
                    )
                    # Ensure data is sorted by group keys and date for consistent lags
                    sort_keys = (
                        group_keys + ["date"] if "date" in df.columns else group_keys
                    )
                    if all(key in df.columns for key in sort_keys):
                        df_sorted = df.sort_values(by=sort_keys)
                        df[new_lag_col_name] = df_sorted.groupby(group_keys)[
                            col_to_lag
                        ].shift(tr.periods)
                        # If original df was not sorted, this might misalign.
                        # It's better to assign back to the original df after computing on sorted,
                        # but pandas handles index alignment.
                        # Let's re-assign to original df index to be safe.
                        df[new_lag_col_name] = df.groupby(group_keys, group_keys=False)[
                            col_to_lag
                        ].shift(tr.periods)

                    else:
                        print(
                            f"Warning: Cannot sort for grouped lag as one of {sort_keys} is missing. Applying simple shift."
                        )
                        df[new_lag_col_name] = df[col_to_lag].shift(tr.periods)
                else:
                    print(
                        f"Applying simple lag on '{col_to_lag}' for {tr.periods} periods."
                    )
                    df[new_lag_col_name] = df[col_to_lag].shift(tr.periods)

        elif isinstance(tr, FillNaGroupedConfig):
            group_by_col_lower = tr.group_by_column.lower()
            if group_by_col_lower not in df.columns:
                print(
                    f"Warning: Group_by_column '{group_by_col_lower}' for fillna_grouped not found. Skipping."
                )
                return df

            target_cols_for_fill = []
            if tr.columns:
                for col_name in tr.columns:
                    col_name_lower = col_name.lower()
                    if col_name_lower not in df.columns:
                        print(
                            f"Warning: Specified column '{col_name_lower}' for fillna_grouped not found. Skipping this column."
                        )
                        continue
                    if not pd.api.types.is_numeric_dtype(df[col_name_lower]):
                        print(
                            f"Warning: Specified column '{col_name_lower}' for fillna_grouped is not numeric. Skipping this column."
                        )
                        continue
                    target_cols_for_fill.append(col_name_lower)
            else:  # If tr.columns is None or empty, fill all numeric columns
                target_cols_for_fill = df.select_dtypes(
                    include=np.number
                ).columns.tolist()
                # Exclude the group_by_column itself if it's numeric and was auto-selected
                if group_by_col_lower in target_cols_for_fill:
                    target_cols_for_fill.remove(group_by_col_lower)

            if not target_cols_for_fill:
                print(
                    "Warning: No numeric columns to process for fillna_grouped. Skipping."
                )
                return df

            print(
                f"Applying fillna_grouped (method: {tr.method}) on columns: {target_cols_for_fill} grouped by '{group_by_col_lower}'"
            )

            if tr.method == "median":
                df[target_cols_for_fill] = df.groupby(
                    group_by_col_lower, group_keys=False
                )[target_cols_for_fill].transform(lambda x: x.fillna(x.median()))
            elif tr.method == "mean":
                df[target_cols_for_fill] = df.groupby(
                    group_by_col_lower, group_keys=False
                )[target_cols_for_fill].transform(lambda x: x.fillna(x.mean()))
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
