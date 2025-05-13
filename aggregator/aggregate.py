"""Core data-aggregation logic."""

from __future__ import annotations
import functools
from pathlib import Path
from typing import List

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
    # FillNaGroupedConfig, # Removed
    TransformationConfig,
    # ImputationConfig, # Removed (superseded by GroupedFillMissingConfig)
    CleanNumericConfig,
    GroupedFillMissingConfig,  # Added
)


# ---- helper registry ---------------------------------------------------- #
def _load_local(path: Path) -> pd.DataFrame:
    """
    Load a file, prioritizing `load_and_preprocess` with `DATA_DIR` logic,
    but falling back to direct load with subsequent standardization if needed.
    Ensures date columns are parsed to datetime if possible.
    Normalization and alignment happen later in DataAggregator.load based on config.
    """
    try:
        # Try the custom loader first
        df = load_and_preprocess(path.name)  # Assuming DATA_DIR is set correctly
        df = _standardize_columns(df)
        date_column_name_std = "date"
        # date_format_str = "%Y%m%d" # load_and_preprocess handles its default format

        if date_column_name_std in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[date_column_name_std]):
                # Already datetime, no action needed here for parsing.
                pass
            # load_and_preprocess should have already parsed the date column.
            # If it's still not datetime, it might be an issue with load_and_preprocess
            # or the column wasn't 'date' initially.
            # For robustness, we can add a check here, but the main parsing
            # should be in load_and_preprocess or the fallback.
            elif pd.api.types.is_integer_dtype(df[date_column_name_std]) or (
                pd.api.types.is_object_dtype(df[date_column_name_std])
                and df[date_column_name_std].astype(str).str.match(r"^\d{8}$").all()
            ):
                try:
                    print(
                        f"Info: Attempting to parse integer/string date column '{date_column_name_std}' in '{path.name}' with format %Y%m%d (after load_and_preprocess)."
                    )
                    df[date_column_name_std] = pd.to_datetime(
                        df[date_column_name_std], format="%Y%m%d"
                    )
                except ValueError as e_fmt:
                    print(
                        f"Warning: Failed to parse '{date_column_name_std}' in '{path.name}' with format %Y%m%d: {e_fmt}. Attempting inference."
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
                        f"Info: Date column '{date_column_name_std}' in '{path.name}' (after load_and_preprocess) is not standard. Attempting general date inference."
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
        df_fallback = pd.read_csv(path, low_memory=False)
        df_fallback = _standardize_columns(df_fallback)
        date_column_name_std = "date"
        date_format_str = "%Y%m%d"  # For fallback, assume this common format

        if date_column_name_std in df_fallback.columns:
            if pd.api.types.is_datetime64_any_dtype(df_fallback[date_column_name_std]):
                pass
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
            else:  # Other types, attempt general inference
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
        self._source_original_columns: dict[str, List[str]] = {}

    def load(self) -> "DataAggregator":
        """Read every data source into memory and apply date handling."""
        for src_cfg in self.cfg.sources:
            loader = CONNECTOR_REGISTRY[src_cfg.connector]
            df = loader(src_cfg.path)

            if not isinstance(df.index, pd.RangeIndex):
                df = df.reset_index()

            df.columns = df.columns.str.lower()

            if "date" in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df["date"]):
                    print(
                        f"Warning: Column 'date' in source '{src_cfg.name}' is not datetime after load. Attempting robust conversion."
                    )
                    try:
                        df["date"] = pd.to_datetime(
                            df["date"], format="%Y%m%d", errors="raise"
                        )
                    except (ValueError, TypeError):
                        try:
                            df["date"] = pd.to_datetime(df["date"], errors="raise")
                        except (ValueError, TypeError) as e:
                            raise ValueError(
                                f"Critical: Failed to convert 'date' column to datetime for source '{src_cfg.name}': {e}. Cannot proceed with date handling."
                            )

                if src_cfg.date_handling:
                    print(
                        f"Applying date handling for source '{src_cfg.name}': {src_cfg.date_handling.model_dump_json(exclude_none=True)}"
                    )
                    if src_cfg.date_handling.frequency == "monthly":
                        df["date"] = (
                            df["date"].dt.to_period("M").dt.to_timestamp(how="end")
                        )
                        print(
                            f"  Source '{src_cfg.name}' dates aligned to month-end. Min: {df['date'].min() if not df['date'].empty else 'N/A'}, Max: {df['date'].max() if not df['date'].empty else 'N/A'}"
                        )
                else:
                    if pd.api.types.is_datetime64_any_dtype(df["date"]):
                        df["date"] = df["date"].dt.normalize()
                        print(
                            f"Info: 'date' column in source '{src_cfg.name}' (no specific date_handling) normalized to midnight. Min: {df['date'].min() if not df['date'].empty else 'N/A'}, Max: {df['date'].max() if not df['date'].empty else 'N/A'}"
                        )

            join_keys_lower = {key.lower() for key in src_cfg.join_on}
            for col in df.columns:
                if col in join_keys_lower or col == "date":
                    continue
                try:
                    if pd.api.types.is_numeric_dtype(
                        df[col]
                    ) or pd.api.types.is_bool_dtype(df[col]):
                        continue
                    if df[col].dtype == "object":
                        temp_numeric_col = pd.to_numeric(df[col], errors="coerce")
                        num_non_numeric_strings = df[col][
                            temp_numeric_col.isna() & df[col].notna()
                        ].nunique()
                        if num_non_numeric_strings > 0:
                            print(
                                f"Warning: Source '{src_cfg.name}', column '{col}' contains {num_non_numeric_strings} unique non-numeric string(s) "
                                f"(e.g., {df[col][temp_numeric_col.isna() & df[col].notna()].unique()[:3]}). "
                                f"Consider using 'clean_numeric' transformation if this column should be numeric."
                            )
                except Exception as e:
                    print(
                        f"Debug: Could not perform numeric check on source '{src_cfg.name}', column '{col}': {e}"
                    )

            self._source_original_columns[src_cfg.name] = list(df.columns)
            join_on_lower = [col.lower() for col in src_cfg.join_on]
            missing_keys = [col for col in join_on_lower if col not in df.columns]
            if missing_keys:
                raise KeyError(
                    f"Source '{src_cfg.name}' ({src_cfg.path}) is missing join key(s): {missing_keys}. "
                    f"Available columns after lowercasing and reset_index: {list(df.columns)[:10]}..."
                )
            self._frames[src_cfg.name] = df
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

    # apply_imputation method is removed

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
                    if not all(key in df.columns for key in sort_keys):
                        print(
                            f"Warning: Not all sort keys ({sort_keys}) found for grouped lag on '{col_to_lag}'. Applying simple shift."
                        )
                        df[new_lag_col_name] = df[col_to_lag].shift(tr.periods)
                    else:
                        # For grouped lags, it's crucial that data is sorted by the time dimension (date)
                        # within each group before applying the shift.
                        # Assigning the result of shift on a sorted version ensures correctness.
                        # Pandas' groupby().shift() itself respects the order within groups if the DataFrame
                        # is already sorted. If not, sorting first is essential.
                        df[new_lag_col_name] = (
                            df.sort_values(by=sort_keys)
                            .groupby(group_keys, group_keys=False)[col_to_lag]
                            .shift(tr.periods)
                        )
                        # The above assignment relies on pandas aligning the shifted series (which has a potentially
                        # reordered index from sort_values) back to the original df's index. This is standard.
                else:
                    print(
                        f"Applying simple lag on '{col_to_lag}' for {tr.periods} periods."
                    )
                    df[new_lag_col_name] = df[col_to_lag].shift(tr.periods)

        # FillNaGroupedConfig handler is removed

        elif isinstance(tr, CleanNumericConfig):
            print(f"Applying 'clean_numeric' transformation for columns: {tr.columns}")
            for col_name_orig in tr.columns:
                col_name = col_name_orig.lower()
                if col_name not in df.columns:
                    print(
                        f"Warning: Column '{col_name}' for clean_numeric not found. Skipping."
                    )
                    continue
                print(f"  Cleaning column: '{col_name}'")
                original_dtype = df[col_name].dtype

                # Calculate initial stats
                total_count = len(df[col_name])
                nan_count_before = df[col_name].isnull().sum()

                # Attempt to convert to numeric to identify non-numeric strings and actual numbers
                # This series is temporary, for inspection.
                temp_numeric_series = pd.to_numeric(df[col_name], errors="coerce")
                numeric_count = temp_numeric_series.notnull().sum()
                # Non-numeric strings are those that are not NaN in original but become NaN after coerce
                # and were not originally NaN.
                non_numeric_string_mask = (
                    temp_numeric_series.isnull() & df[col_name].notnull()
                )
                non_numeric_string_count = non_numeric_string_mask.sum()
                print(f"    Stats for '{col_name}' (before cleaning):")
                print(f"      Total entries: {total_count}")
                print(f"      Original NaN count: {nan_count_before}")
                print(f"      Convertible to numeric count: {numeric_count}")
                print(
                    f"      Non-convertible string count (will become NaN): {non_numeric_string_count}"
                )
                if non_numeric_string_count == 0 and pd.api.types.is_numeric_dtype(
                    original_dtype
                ):
                    print(
                        f"    Column '{col_name}' is already numeric and has no non-convertible strings. Skipping actual conversion."
                    )
                elif (
                    non_numeric_string_count == 0
                    and not pd.api.types.is_numeric_dtype(original_dtype)
                    and numeric_count + nan_count_before == total_count
                ):
                    print(
                        f"    Column '{col_name}' contains only numbers and NaNs but might be object type. Forcing to numeric."
                    )
                    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
                    print(
                        f"    Column '{col_name}' converted to {df[col_name].dtype}. New NaN count: {df[col_name].isnull().sum()}"
                    )
                elif tr.action == "to_nan":
                    if non_numeric_string_count > 0:
                        print(
                            f"    Action 'to_nan': Converting {non_numeric_string_count} non-numeric string(s) in '{col_name}' to NaN."
                        )
                    else:
                        print(
                            f"    Action 'to_nan': Column '{col_name}' has no non-convertible strings to change to NaN, but ensuring it is numeric type."
                        )
                    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
                    nan_count_after = df[col_name].isnull().sum()
                    print(
                        f"    Column '{col_name}' converted to {df[col_name].dtype}. New NaN count: {nan_count_after} (was {nan_count_before}, {non_numeric_string_count} new NaNs created from strings)."
                    )
                else:
                    print(
                        f"    Warning: Unknown action '{tr.action}' for clean_numeric on column '{col_name}'. Skipping action."
                    )

        elif isinstance(
            tr, GroupedFillMissingConfig
        ):  # New handler for GroupedFillMissingConfig
            print(
                f"Applying 'grouped_fill_missing' (method: {tr.method}) on columns grouped by '{tr.group_by_column}'."
            )

            group_by_col_lower = tr.group_by_column.lower()
            if group_by_col_lower not in df.columns:
                raise ValueError(
                    f"GroupedFillMissingConfig: group_by_column '{group_by_col_lower}' not found in DataFrame columns: {df.columns.tolist()}."
                )

            cols_to_fill: List[str] = []
            if tr.columns:  # User specified columns
                print(f"  Targeting user-specified columns: {tr.columns}")
                temp_target_cols = [col.lower() for col in tr.columns]
                for col in temp_target_cols:
                    if col not in df.columns:
                        print(
                            f"  Warning: Specified column '{col}' for grouped_fill_missing not found. Skipping."
                        )
                        continue
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        # This check is important. If a column isn't numeric (e.g., after clean_numeric failed or wasn't run),
                        # mean/median will fail or produce unexpected results.
                        print(
                            f"  Warning: Specified column '{col}' for grouped_fill_missing is not numeric (current dtype: {df[col].dtype}). Skipping this column. Ensure 'clean_numeric' was applied if needed."
                        )
                        continue
                    if col == group_by_col_lower:
                        print(
                            f"  Warning: Specified column '{col}' is the group_by_column. Skipping."
                        )
                        continue
                    cols_to_fill.append(col)
            else:  # Auto-identify numeric columns
                print(
                    "  Auto-identifying numeric columns for grouped_fill_missing (excluding group_by_column)."
                )
                all_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                cols_to_fill = [
                    col for col in all_numeric_cols if col != group_by_col_lower
                ]

            if not cols_to_fill:
                print(
                    "  No suitable numeric columns found for grouped_fill_missing. Skipping this transformation step."
                )
                return df
            print(f"  Columns to be processed by grouped_fill_missing: {cols_to_fill}")

            # Pre-check for missing value thresholds
            print(
                f"  Performing pre-check for missing value thresholds on columns, grouped by '{group_by_col_lower}'..."
            )
            for col_to_check in cols_to_fill:

                def check_group_missing_ratio(group_series: pd.Series):
                    group_identifier = group_series.name
                    if group_series.empty:
                        return
                    missing_count = group_series.isnull().sum()
                    total_count = len(group_series)
                    if total_count == 0:
                        return
                    missing_ratio = missing_count / total_count
                    group_id_str = group_identifier
                    if isinstance(group_identifier, pd.Timestamp):
                        group_id_str = group_identifier.strftime("%Y-%m-%d")
                    if missing_ratio > tr.missing_threshold_error:
                        raise ValueError(
                            f"Error (GroupedFillMissing): Column '{col_to_check}' in group '{group_by_col_lower}={group_id_str}' has {missing_ratio * 100:.2f}% missing values, "
                            f"exceeding error threshold of {tr.missing_threshold_error * 100:.2f}%."
                        )
                    if missing_ratio > tr.missing_threshold_warning:
                        print(
                            f"Warning (GroupedFillMissing): Column '{col_to_check}' in group '{group_by_col_lower}={group_id_str}' has {missing_ratio * 100:.2f}% missing values, "
                            f"exceeding warning threshold of {tr.missing_threshold_warning * 100:.2f}%."
                        )

                df.groupby(group_by_col_lower, group_keys=True)[col_to_check].apply(
                    check_group_missing_ratio
                )

            print(f"  Pre-check complete. Applying fill method: '{tr.method}'...")
            for col_to_impute in cols_to_fill:
                if tr.method == "mean":
                    fill_value_func = lambda x: x.fillna(x.mean())
                elif tr.method == "median":
                    fill_value_func = lambda x: x.fillna(x.median())
                else:  # Should be caught by Pydantic
                    raise ValueError(
                        f"Internal Error: Invalid fill method '{tr.method}'."
                    )

                df[col_to_impute] = df.groupby(group_by_col_lower, group_keys=False)[
                    col_to_impute
                ].transform(fill_value_func)
            print(
                "  Grouped_fill_missing process completed for this transformation step."
            )
        return df


def aggregate_from_yaml(
    spec_path: str | Path,
) -> tuple[pd.DataFrame, AggregationConfig]:
    """One-shot helper: parse YAML → load → merge → transform → DataFrame."""
    with open(spec_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = AggregationConfig.model_validate(raw)

    agg = DataAggregator(cfg).load()
    merged_df = agg.merge()

    transformed_df = agg.apply_transformations(merged_df)

    return transformed_df, cfg
