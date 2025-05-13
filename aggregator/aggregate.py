"""Core data-aggregation logic."""

from __future__ import annotations
import functools
from pathlib import Path
from typing import Union, List  # Added List

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
    ImputationConfig,
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
        self._source_original_columns: dict[str, List[str]] = {}  # Added

    def load(self) -> "DataAggregator":
        """Read every data source into memory."""
        for src_cfg in self.cfg.sources:
            loader = CONNECTOR_REGISTRY[src_cfg.connector]
            df = loader(
                src_cfg.path
            )  # Pydantic should convert path string to Path object

            if not isinstance(df.index, pd.RangeIndex):
                df = df.reset_index()  # Ensure join keys are columns

            # Standardize column names after loading and potential index reset
            df.columns = df.columns.str.lower()
            # Store standardized column names for later reference (e.g., macro checks)
            self._source_original_columns[src_cfg.name] = list(df.columns)

            # Standardize join_on keys to lower case for checking
            join_on_lower = [col.lower() for col in src_cfg.join_on]

            missing_keys = [col for col in join_on_lower if col not in df.columns]
            if missing_keys:
                raise KeyError(
                    f"Source '{src_cfg.name}' ({src_cfg.path}) is missing join key(s): {missing_keys}. "
                    f"Available columns after lowercasing and reset_index: {list(df.columns)[:10]}..."
                )
            self._frames[src_cfg.name] = df
        # print("Stop here") # Removed debug print
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
                if not self._frames:
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

        primary_firm_config = None
        primary_firm_sources_marked = [
            s_cfg for s_cfg in firm_source_configs if s_cfg.is_primary_firm_base
        ]

        if len(primary_firm_sources_marked) == 1:
            primary_firm_config = primary_firm_sources_marked[0]
            print(
                f"Identified explicit primary firm-level source: {primary_firm_config.name}"
            )
        elif len(firm_source_configs) == 1:
            primary_firm_config = firm_source_configs[0]
            print(
                f"Using the only firm-level source as de-facto primary: {primary_firm_config.name}"
            )
        else:
            raise ValueError(
                "Multiple firm-level sources are provided. Please specify exactly one as the primary base "
                "by setting 'is_primary_firm_base: True' for that source in the YAML configuration."
            )

        base_df = self._frames[primary_firm_config.name].copy()
        print(
            f"Starting merge with primary firm-level source: {primary_firm_config.name}"
        )

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

        for macro_cfg in macro_source_configs:
            right_df = self._frames[macro_cfg.name]
            keys = [col.lower() for col in macro_cfg.join_on]
            print(f"Merging macro-level source: {macro_cfg.name} on keys: {keys}")
            base_df = base_df.merge(right_df, how="left", on=keys)

        return base_df

    def apply_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies data imputation based on the configuration."""
        if not self.cfg.imputation:
            print("No imputation config provided. Skipping imputation step.")
            return df

        imp_cfg = self.cfg.imputation
        print(
            f"Starting imputation: method='{imp_cfg.method}', group_by='{imp_cfg.group_by_column}'."
        )

        # 1. Macro data check: Non-join key columns from macro sources must not have NaNs after merge.
        print("Checking for NaNs in data columns from macro sources...")
        for src_cfg_iter in self.cfg.sources:
            if src_cfg_iter.level == "macro":
                original_cols_this_macro = self._source_original_columns.get(
                    src_cfg_iter.name, []
                )
                join_keys_this_macro = [k.lower() for k in src_cfg_iter.join_on]

                for col_name in original_cols_this_macro:
                    if col_name in df.columns:  # Check if column exists in merged_df
                        if col_name in join_keys_this_macro:
                            # Join keys are not considered "data columns" for this specific check's purpose
                            continue

                        # This is a data column (non-join key) from a macro source
                        if df[col_name].isnull().any():
                            num_nans = df[col_name].isnull().sum()
                            total_rows = len(df)
                            raise ValueError(
                                f"Error: Data column '{col_name}' from macro source '{src_cfg_iter.name}' contains {num_nans} missing values "
                                f"(out of {total_rows}) in the merged DataFrame. Macro data columns must be complete after joins."
                            )
        print(
            "Macro column check complete: No NaNs found in non-join key columns from macro sources."
        )

        # 2. Identify target columns for imputation
        group_by_col_lower = imp_cfg.group_by_column.lower()
        if group_by_col_lower not in df.columns:
            raise ValueError(
                f"Imputation group_by_column '{group_by_col_lower}' not found in DataFrame columns: {df.columns.tolist()}."
            )

        target_cols_for_imputation: List[str] = []
        if imp_cfg.target_columns:
            print(
                f"Using user-specified target columns for imputation: {imp_cfg.target_columns}"
            )
            temp_target_cols = [col.lower() for col in imp_cfg.target_columns]
            for col in temp_target_cols:
                if col not in df.columns:
                    print(
                        f"Warning: Specified imputation target column '{col}' not found. Skipping."
                    )
                    continue
                if not pd.api.types.is_numeric_dtype(df[col]):
                    print(
                        f"Warning: Specified imputation target column '{col}' is not numeric. Skipping."
                    )
                    continue
                if col == group_by_col_lower:
                    print(
                        f"Warning: Specified imputation target column '{col}' is the group_by_column. Skipping."
                    )
                    continue
                target_cols_for_imputation.append(col)
        else:
            print(
                "Auto-identifying numeric columns for imputation (excluding macro data cols and group_by col)..."
            )
            all_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            # Identify data columns from macro sources to exclude them from general imputation
            macro_data_cols_to_exclude = set()
            for src_cfg_iter in self.cfg.sources:
                if src_cfg_iter.level == "macro":
                    original_cols = self._source_original_columns.get(
                        src_cfg_iter.name, []
                    )
                    join_keys = [k.lower() for k in src_cfg_iter.join_on]
                    for col in original_cols:
                        if col in df.columns and col not in join_keys:
                            macro_data_cols_to_exclude.add(col)

            target_cols_for_imputation = [
                col
                for col in all_numeric_cols
                if col != group_by_col_lower and col not in macro_data_cols_to_exclude
            ]

        if not target_cols_for_imputation:
            print(
                "No suitable columns found for imputation after filtering. Skipping imputation step."
            )
            return df
        print(f"Columns targeted for imputation: {target_cols_for_imputation}")

        # 3. Pre-check for missing value thresholds within groups
        print(
            f"Performing pre-check for missing value thresholds on columns, grouped by '{group_by_col_lower}'..."
        )

        for col_to_check in target_cols_for_imputation:

            def check_group_missing_ratio(group_series: pd.Series):
                # group_series.name will be the group key when used with groupby().apply() and group_keys=True
                group_identifier = group_series.name

                if group_series.empty:
                    return

                missing_count = group_series.isnull().sum()
                total_count = len(group_series)
                if total_count == 0:
                    return

                missing_ratio = missing_count / total_count

                if missing_ratio > imp_cfg.missing_threshold_error:
                    raise ValueError(
                        f"Error: Column '{col_to_check}' in group '{group_by_col_lower}={group_identifier}' has {missing_ratio * 100:.2f}% missing values, "
                        f"exceeding error threshold of {imp_cfg.missing_threshold_error * 100:.2f}%."
                    )
                if missing_ratio > imp_cfg.missing_threshold_warning:
                    print(
                        f"Warning: Column '{col_to_check}' in group '{group_by_col_lower}={group_identifier}' has {missing_ratio * 100:.2f}% missing values, "
                        f"exceeding warning threshold of {imp_cfg.missing_threshold_warning * 100:.2f}%."
                    )

            # Using apply to access group names for richer error/warning messages
            df.groupby(group_by_col_lower, group_keys=True)[col_to_check].apply(
                check_group_missing_ratio
            )

        print(f"Pre-check complete. Applying imputation method: '{imp_cfg.method}'...")

        # 4. Actual imputation using transform
        for col_to_impute in target_cols_for_imputation:
            if imp_cfg.method == "mean":

                def fill_value_func(x):
                    return x.fillna(x.mean())
            elif imp_cfg.method == "median":

                def fill_value_func(x):
                    return x.fillna(x.median())
            else:  # Should be caught by Pydantic Literal type
                raise ValueError(
                    f"Internal Error: Invalid imputation method '{imp_cfg.method}' despite schema validation."
                )

            df[col_to_impute] = df.groupby(group_by_col_lower, group_keys=False)[
                col_to_impute
            ].transform(fill_value_func)

        print("Imputation process completed.")
        return df

    def apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sequentially apply every transformation from the YAML spec."""
        if not self.cfg.transformations:
            return df
        return functools.reduce(self._apply_one, self.cfg.transformations, df)

    @staticmethod
    def _apply_one(df: pd.DataFrame, tr: TransformationConfig) -> pd.DataFrame:
        if isinstance(tr, OneHotConfig):
            if tr.column.lower() not in df.columns:
                print(
                    f"Warning: Column '{tr.column}' for one-hot encoding not found. Skipping."
                )
                return df
            dummies = pd.get_dummies(df[tr.column.lower()], prefix=tr.prefix, dtype=int)
            df = pd.concat([df, dummies], axis=1)
            if tr.drop_original:
                df = df.drop(columns=tr.column.lower())
        elif isinstance(tr, LagConfig):
            lag_cols_lower = [col.lower() for col in tr.columns]
            group_keys = []
            if "permno" in df.columns:
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
                    sort_keys = (
                        group_keys + ["date"] if "date" in df.columns else group_keys
                    )
                    # Check if all sort_keys are present before attempting to sort
                    if not all(key in df.columns for key in sort_keys):
                        print(
                            f"Warning: Not all sort keys ({sort_keys}) found for grouped lag on '{col_to_lag}'. Applying simple shift."
                        )
                        df[new_lag_col_name] = df[col_to_lag].shift(tr.periods)
                    else:
                        # Using group_keys=False to prevent adding group keys to index of the result of shift
                        df[new_lag_col_name] = (
                            df.sort_values(by=sort_keys)
                            .groupby(group_keys, group_keys=False)[col_to_lag]
                            .shift(tr.periods)
                        )
                        # Reindex to match original df's index if sorting changed order significantly
                        # However, pandas assignment by column name usually handles index alignment.
                        # If df was not sorted by group_keys, date initially, this could be an issue.
                        # A safer assignment after sorting and grouping:
                        # lagged_series = df.sort_values(by=sort_keys).groupby(group_keys)[col_to_lag].shift(tr.periods)
                        # df[new_lag_col_name] = lagged_series # relies on pandas index alignment
                        # The current approach of assigning directly to df[new_name] after a groupby().shift()
                        # on a (potentially sorted) copy is generally fine due to pandas' index alignment.
                        # For clarity and safety, ensure the groupby operation for shift is on a DataFrame
                        # that has the same index as the original `df` or can be aligned.
                        # The current code: df.groupby(...).shift() assigns based on original df's index.
                        df[new_lag_col_name] = df.groupby(group_keys, group_keys=False)[
                            col_to_lag
                        ].shift(tr.periods)

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
            else:
                target_cols_for_fill = df.select_dtypes(
                    include=np.number
                ).columns.tolist()
                if (
                    group_by_col_lower in target_cols_for_fill
                ):  # Exclude group_by if numeric
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
    """One-shot helper: parse YAML → load → merge → impute → transform → DataFrame."""
    with open(spec_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = AggregationConfig.model_validate(raw)

    agg = DataAggregator(cfg).load()
    merged_df = agg.merge()
    imputed_df = agg.apply_imputation(merged_df)  # New imputation step
    transformed_df = agg.apply_transformations(
        imputed_df
    )  # Apply transformations on imputed_df

    return transformed_df, cfg
