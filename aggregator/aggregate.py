"""Core data-aggregation logic."""

from __future__ import annotations
import functools
from pathlib import Path
from typing import List, Dict, Optional
import warnings

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
    TransformationConfig,
    CleanNumericConfig,
    GroupedFillMissingConfig,
    ExpandCartesianConfig,
    SourceConfig,
)


# ---- helper registry ---------------------------------------------------- #
# _load_local method remains unchanged
def _load_local(path: Path) -> pd.DataFrame:
    try:
        df = load_and_preprocess(path.name)
        df = _standardize_columns(df)
        date_column_name_std = "date"
        if date_column_name_std in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[date_column_name_std]):
                pass
            elif pd.api.types.is_integer_dtype(df[date_column_name_std]) or (
                pd.api.types.is_object_dtype(df[date_column_name_std])
                and df[date_column_name_std].astype(str).str.match(r"^\d{8}$").all()
            ):
                try:
                    df[date_column_name_std] = pd.to_datetime(
                        df[date_column_name_std], format="%Y%m%d"
                    )
                except ValueError:
                    try:
                        df[date_column_name_std] = pd.to_datetime(
                            df[date_column_name_std]
                        )
                    except ValueError as e:
                        print(f"Error parsing date in '{path.name}': {e}")
            else:
                try:
                    df[date_column_name_std] = pd.to_datetime(df[date_column_name_std])
                except ValueError as e:
                    print(f"Error parsing date in '{path.name}': {e}")
        return df
    except FileNotFoundError:
        df_fallback = pd.read_csv(path, low_memory=False)
        df_fallback = _standardize_columns(df_fallback)
        date_column_name_std = "date"
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
                    df_fallback[date_column_name_std] = pd.to_datetime(
                        df_fallback[date_column_name_std], format="%Y%m%d"
                    )
                except ValueError:
                    try:
                        df_fallback[date_column_name_std] = pd.to_datetime(
                            df_fallback[date_column_name_std]
                        )
                    except ValueError as e:
                        print(f"Error parsing date in fallback '{path}': {e}")
            else:
                try:
                    df_fallback[date_column_name_std] = pd.to_datetime(
                        df_fallback[date_column_name_std]
                    )
                except ValueError as e:
                    print(f"Error parsing date in fallback '{path}': {e}")
        return df_fallback


CONNECTOR_REGISTRY = {"local": _load_local}

# ---- public API ----------------------------------------------------------- #


class DataAggregator:
    """Transforms multiple raw data sources into a single modelling table."""

    def __init__(self, cfg: AggregationConfig) -> None:
        self.cfg = cfg
        self._frames: dict[str, pd.DataFrame] = {}
        # Stores final (suffixed) column names for each source
        self._source_final_columns: dict[str, List[str]] = {}
        # Stores mapping from original column name to suffixed name for each source
        self._source_col_rename_map: dict[str, Dict[str, str]] = {}

    def load(self) -> "DataAggregator":
        """Read every data source, apply automatic suffixing based on level, and date handling."""
        # To check for duplicate original column names within the same level
        seen_original_cols_by_level: Dict[str, Dict[str, str]] = {
            "firm": {},
            "macro": {},
        }

        for src_cfg in self.cfg.sources:
            loader = CONNECTOR_REGISTRY[src_cfg.connector]
            df = loader(src_cfg.path)

            if not isinstance(df.index, pd.RangeIndex):
                df = df.reset_index()

            original_cols_before_lower = list(
                df.columns
            )  # For reference before lowercasing
            df.columns = df.columns.str.lower()
            original_cols_after_lower = list(df.columns)

            current_join_keys_lower = {key.lower() for key in src_cfg.join_on}

            # Check for duplicate original column names within the same level BEFORE suffixing
            for original_col_name in original_cols_after_lower:
                if original_col_name in current_join_keys_lower:
                    continue  # Skip join keys for this check

                level_seen_cols = seen_original_cols_by_level[src_cfg.level]
                if (
                    original_col_name in level_seen_cols
                    and level_seen_cols[original_col_name] != src_cfg.name
                ):
                    raise ValueError(
                        f"Duplicate original column name '{original_col_name}' found in multiple sources of level '{src_cfg.level}'. "
                        f"Source '{src_cfg.name}' and source '{level_seen_cols[original_col_name]}' both have this column. "
                        "Please ensure unique column names (excluding join keys) within the same data level across different files, or rename them in the source files."
                    )
                level_seen_cols[original_col_name] = src_cfg.name

            # Automatic suffixing based on level
            auto_suffix = f"_{src_cfg.level}"  # e.g., _firm, _macro
            print(
                f"Automatically applying suffix '{auto_suffix}' to non-join-key columns of source '{src_cfg.name}' (level: {src_cfg.level})."
            )

            rename_map: Dict[str, str] = {}
            new_columns_for_df = {}
            for col in df.columns:
                if col in current_join_keys_lower:
                    new_columns_for_df[col] = col
                    rename_map[col] = col  # Original maps to itself
                else:
                    new_name = f"{col}{auto_suffix}"
                    if (
                        new_name in current_join_keys_lower
                    ):  # Highly unlikely if suffixes are _firm/_macro
                        raise ValueError(
                            f"Error in source '{src_cfg.name}': Auto-suffix '{auto_suffix}' on column '{col}' creates '{new_name}', clashing with a join key."
                        )
                    if (
                        new_name in new_columns_for_df.values()
                    ):  # Intra-source clash after suffixing
                        raise ValueError(
                            f"Error in source '{src_cfg.name}': Auto-suffix '{auto_suffix}' creates duplicate column name '{new_name}'. Original columns: {original_cols_before_lower}"
                        )
                    new_columns_for_df[col] = new_name
                    rename_map[col] = new_name  # Original maps to suffixed

            df = df.rename(columns=new_columns_for_df)
            self._source_col_rename_map[src_cfg.name] = rename_map
            print(
                f"  Columns for '{src_cfg.name}' after auto-suffixing: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}"
            )

            date_col_to_handle = "date"
            if (
                date_col_to_handle in df.columns
            ):  # 'date' is a join key, so it won't be suffixed
                if not pd.api.types.is_datetime64_any_dtype(df[date_col_to_handle]):
                    print(
                        f"Warning: Column '{date_col_to_handle}' in source '{src_cfg.name}' is not datetime. Attempting conversion."
                    )
                    try:
                        df[date_col_to_handle] = pd.to_datetime(
                            df[date_col_to_handle], format="%Y%m%d", errors="raise"
                        )
                    except (ValueError, TypeError):
                        try:
                            df[date_col_to_handle] = pd.to_datetime(
                                df[date_col_to_handle], errors="raise"
                            )
                        except (ValueError, TypeError) as e:
                            raise ValueError(
                                f"Critical: Failed to convert '{date_col_to_handle}' for '{src_cfg.name}': {e}."
                            )

                if (
                    src_cfg.date_handling
                    and src_cfg.date_handling.frequency == "monthly"
                    and pd.api.types.is_datetime64_any_dtype(df[date_col_to_handle])
                ):
                    print(
                        f"Applying monthly date handling for source '{src_cfg.name}'."
                    )
                    df[date_col_to_handle] = (
                        df[date_col_to_handle]
                        .dt.to_period("M")
                        .dt.to_timestamp(how="end")
                        .dt.normalize()
                    )
                    print(
                        f"  Source '{src_cfg.name}' dates aligned to month-end (normalized)."
                    )
                elif pd.api.types.is_datetime64_any_dtype(df[date_col_to_handle]):
                    df[date_col_to_handle] = df[date_col_to_handle].dt.normalize()
                    print(
                        f"Info: '{date_col_to_handle}' column in source '{src_cfg.name}' normalized."
                    )

            self._source_final_columns[src_cfg.name] = list(df.columns)

            # Check for missing join keys (these are original, unsuffixed names from YAML)
            missing_keys = [
                key_col
                for key_col in current_join_keys_lower
                if key_col not in df.columns
            ]
            if missing_keys:
                raise KeyError(
                    f"Source '{src_cfg.name}' missing join key(s): {missing_keys}. Available columns after suffixing: {list(df.columns)[:10]}..."
                )
            self._frames[src_cfg.name] = df

        # Inter-source duplicate check for *final suffixed non-join columns* is implicitly handled
        # because if ep_firm and ep_macro exist, they are unique.
        # The critical check was for original names within the same level.
        print("Source loading and automatic suffixing complete.")
        return self

    def merge(self) -> pd.DataFrame:  # Unchanged
        if not self._frames:
            raise RuntimeError("Call load() first.")
        firm_sources = [s for s in self.cfg.sources if s.level == "firm"]
        macro_sources = [s for s in self.cfg.sources if s.level == "macro"]
        base_df = None
        if not firm_sources:
            if not macro_sources:
                raise ValueError("No sources configured.")
            print("Warning: No firm-level sources. Merging macro sources only.")
            base_df = self._frames[macro_sources[0].name].copy()
            for i in range(1, len(macro_sources)):
                cfg = macro_sources[i]
                right_df = self._frames[cfg.name]
                keys = [k.lower() for k in cfg.join_on]
                base_df = base_df.merge(right_df, how="left", on=keys)
            return base_df
        primary_cfg = next((s for s in firm_sources if s.is_primary_firm_base), None)
        if not primary_cfg:
            if len(firm_sources) == 1:
                primary_cfg = firm_sources[0]
            else:
                raise ValueError(
                    "Multiple firm sources; one must be primary_firm_base: True."
                )
        print(f"Primary firm source: {primary_cfg.name}")
        base_df = self._frames[primary_cfg.name].copy()
        for cfg in firm_sources:
            if cfg.name == primary_cfg.name:
                continue
            right_df = self._frames[cfg.name]
            keys = [k.lower() for k in cfg.join_on]
            base_df = base_df.merge(right_df, how="left", on=keys)
        for cfg in macro_sources:
            right_df = self._frames[cfg.name]
            keys = [k.lower() for k in cfg.join_on]
            base_df = base_df.merge(right_df, how="left", on=keys)
        return base_df

    def apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.cfg.transformations:
            return df
        # Pass self to _apply_one as it's now an instance method
        return functools.reduce(self._apply_one, self.cfg.transformations, df)

    # Made _apply_one an instance method
    def _apply_one(self, df: pd.DataFrame, tr: TransformationConfig) -> pd.DataFrame:
        if isinstance(tr, OneHotConfig):
            # User provides base column name, we find its suffixed version
            target_col_base = tr.column.lower()
            target_col_actual = self._find_actual_column_name(
                df, target_col_base, self.cfg.sources, self._source_col_rename_map
            )
            if not target_col_actual:
                print(
                    f"Warning: Column '{tr.column}' for one-hot encoding not found (after considering suffixes). Skipping."
                )
                return df

            print(f"Applying OneHot to '{target_col_actual}' (base: '{tr.column}')")
            dummies = pd.get_dummies(df[target_col_actual], prefix=tr.prefix, dtype=int)
            df = pd.concat([df, dummies], axis=1)
            if tr.drop_original:
                df = df.drop(columns=target_col_actual)

        elif isinstance(tr, LagConfig):
            processed_lag_cols = []
            for base_col in tr.columns:
                actual_col = self._find_actual_column_name(
                    df, base_col.lower(), self.cfg.sources, self._source_col_rename_map
                )
                if actual_col:
                    processed_lag_cols.append(actual_col)
                else:
                    print(f"Warning: Column '{base_col}' for lag not found. Skipping.")

            if not processed_lag_cols:
                return df

            group_keys = []
            if "permno" in df.columns:
                group_keys.append("permno")
            for col_to_lag_actual in processed_lag_cols:
                new_lag_col_name = (
                    f"{col_to_lag_actual}_lag{tr.periods}"  # Lag suffixed name
                )
                if group_keys:
                    sort_keys = (
                        group_keys + ["date"] if "date" in df.columns else group_keys
                    )
                    if not all(key in df.columns for key in sort_keys):
                        df[new_lag_col_name] = df[col_to_lag_actual].shift(tr.periods)
                    else:
                        df[new_lag_col_name] = (
                            df.sort_values(by=sort_keys)
                            .groupby(group_keys, group_keys=False)[col_to_lag_actual]
                            .shift(tr.periods)
                        )
                else:
                    df[new_lag_col_name] = df[col_to_lag_actual].shift(tr.periods)

        elif isinstance(tr, CleanNumericConfig):
            print(f"Applying 'clean_numeric' transformation.")
            for col_name_base in tr.columns:
                actual_col = self._find_actual_column_name(
                    df,
                    col_name_base.lower(),
                    self.cfg.sources,
                    self._source_col_rename_map,
                )
                if not actual_col:
                    print(
                        f"Warning: Column '{col_name_base}' for clean_numeric not found. Skipping."
                    )
                    continue

                print(f"  Cleaning column: '{actual_col}' (base: '{col_name_base}')")
                # ... (rest of CleanNumericConfig logic using actual_col) ...
                original_dtype = df[actual_col].dtype
                total_count = len(df[actual_col])
                nan_count_before = df[actual_col].isnull().sum()
                temp_numeric_series = pd.to_numeric(df[actual_col], errors="coerce")
                numeric_count = temp_numeric_series.notnull().sum()
                non_numeric_string_mask = (
                    temp_numeric_series.isnull() & df[actual_col].notnull()
                )
                non_numeric_string_count = non_numeric_string_mask.sum()
                print(
                    f"    Stats for '{actual_col}' (before cleaning): Total entries: {total_count}, Original NaN: {nan_count_before}, Numeric: {numeric_count}, Non-convertible strings: {non_numeric_string_count}"
                )
                if non_numeric_string_count == 0 and pd.api.types.is_numeric_dtype(
                    original_dtype
                ):
                    print(
                        f"    Column '{actual_col}' is already numeric. Skipping conversion."
                    )
                elif (
                    non_numeric_string_count == 0
                    and not pd.api.types.is_numeric_dtype(original_dtype)
                    and numeric_count + nan_count_before == total_count
                ):
                    print(
                        f"    Column '{actual_col}' contains only numbers/NaNs but is object type. Forcing to numeric."
                    )
                    df[actual_col] = pd.to_numeric(df[actual_col], errors="coerce")
                    print(
                        f"    Column '{actual_col}' converted to {df[actual_col].dtype}. New NaN count: {df[actual_col].isnull().sum()}"
                    )
                elif tr.action == "to_nan":
                    print(
                        f"    Action 'to_nan': Converting non-numeric strings in '{actual_col}' to NaN and ensuring numeric type."
                    )
                    df[actual_col] = pd.to_numeric(
                        df[actual_col], errors="coerce"
                    )  # This is the actual conversion
                    nan_count_after = df[actual_col].isnull().sum()
                    print(
                        f"    Column '{actual_col}' converted to {df[actual_col].dtype}. New NaN count: {nan_count_after}."
                    )
                else:
                    print(
                        f"    Warning: Unknown action '{tr.action}' for clean_numeric on column '{actual_col}'. Skipping action."
                    )

        elif isinstance(tr, GroupedFillMissingConfig):
            group_by_col_lower = (
                tr.group_by_column.lower()
            )  # group_by_column is usually a join key, not suffixed
            if group_by_col_lower not in df.columns:
                raise ValueError(
                    f"GroupedFillMissing: group_by_column '{group_by_col_lower}' not found."
                )

            cols_to_fill_actual: List[str] = []
            if tr.columns:  # User specified base column names
                print(
                    f"  Targeting user-specified columns for GroupedFillMissing: {tr.columns}"
                )
                for base_col in tr.columns:
                    actual_col = self._find_actual_column_name(
                        df,
                        base_col.lower(),
                        self.cfg.sources,
                        self._source_col_rename_map,
                    )
                    if not actual_col:
                        print(
                            f"  Warning: Base column '{base_col}' for grouped_fill_missing not found. Skipping."
                        )
                        continue
                    if not pd.api.types.is_numeric_dtype(df[actual_col]):
                        print(
                            f"  Warning: Column '{actual_col}' (base: {base_col}) is not numeric. Skipping."
                        )
                        continue
                    if actual_col == group_by_col_lower:
                        print(
                            f"  Warning: Column '{actual_col}' is group_by. Skipping."
                        )
                        continue
                    cols_to_fill_actual.append(actual_col)
            else:  # Auto-identify all numeric columns (these will be suffixed already)
                print(
                    "  Auto-identifying numeric columns for grouped_fill_missing (excluding group_by_column)."
                )
                all_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                cols_to_fill_actual = [
                    c for c in all_numeric_cols if c != group_by_col_lower
                ]

            if not cols_to_fill_actual:
                print(
                    "  No suitable numeric columns found for GroupedFillMissing. Skipping."
                )
                return df
            print(
                f"  Actual columns to be processed by GroupedFillMissing: {cols_to_fill_actual}"
            )

            for col_to_check_actual in cols_to_fill_actual:
                df.groupby(group_by_col_lower, group_keys=True)[
                    col_to_check_actual
                ].apply(
                    lambda g: self._check_group_missing_ratio_static(
                        g,
                        col_to_check_actual,
                        group_by_col_lower,
                        tr.missing_threshold_warning,
                        tr.missing_threshold_error,
                    )
                )
            for col_to_impute_actual in cols_to_fill_actual:
                fill_fn = (
                    (lambda x: x.fillna(x.mean()))
                    if tr.method == "mean"
                    else (lambda x: x.fillna(x.median()))
                )
                df[col_to_impute_actual] = df.groupby(
                    group_by_col_lower, group_keys=False
                )[col_to_impute_actual].transform(fill_fn)
            print("  Grouped_fill_missing process completed.")

        elif isinstance(tr, ExpandCartesianConfig):
            print(
                f"Applying 'expand_cartesian' transformation (infer_suffix: {tr.infer_suffix})."
            )

            # Use self._find_actual_column_name for infer_suffix logic
            processed_macro_cols_actual = [
                self._find_actual_column_name(
                    df,
                    m.lower(),
                    self.cfg.sources,
                    self._source_col_rename_map,
                    "macro" if tr.infer_suffix else None,
                )
                for m in tr.macro_columns
            ]
            processed_firm_cols_actual = [
                self._find_actual_column_name(
                    df,
                    f.lower(),
                    self.cfg.sources,
                    self._source_col_rename_map,
                    "firm" if tr.infer_suffix else None,
                )
                for f in tr.firm_columns
            ]

            # Filter out None values (columns not found)
            valid_macro_cols = [
                col for col in processed_macro_cols_actual if col is not None
            ]
            valid_firm_cols = [
                col for col in processed_firm_cols_actual if col is not None
            ]

            new_interaction_cols_created = []
            for m_col_actual in valid_macro_cols:
                if not pd.api.types.is_numeric_dtype(df[m_col_actual]):
                    print(
                        f"  Warning (ExpandCartesian): Macro column '{m_col_actual}' is not numeric. Skipping."
                    )
                    continue
                for f_col_actual in valid_firm_cols:
                    if not pd.api.types.is_numeric_dtype(df[f_col_actual]):
                        print(
                            f"  Warning (ExpandCartesian): Firm column '{f_col_actual}' is not numeric. Skipping."
                        )
                        continue

                    new_col_name = f"{m_col_actual}_x_{f_col_actual}"
                    if new_col_name in df.columns:
                        print(
                            f"  Warning (ExpandCartesian): Interaction column '{new_col_name}' already exists. Overwriting."
                        )

                    print(
                        f"  Creating interaction: {m_col_actual} * {f_col_actual} -> {new_col_name}"
                    )
                    df[new_col_name] = df[m_col_actual].astype(float) * df[
                        f_col_actual
                    ].astype(float)
                    new_interaction_cols_created.append(new_col_name)

            if new_interaction_cols_created:
                print(
                    f"  Created {len(new_interaction_cols_created)} interaction columns."
                )
            else:
                print("  No interaction columns were created by 'expand_cartesian'.")
        return df

    @staticmethod
    def _check_group_missing_ratio_static(
        group_series: pd.Series,
        col_name: str,
        group_by_col: str,
        warn_thresh: float,
        err_thresh: float,
    ):
        # ... (same as your last version)
        group_identifier = group_series.name
        missing_ratio = (
            group_series.isnull().sum() / len(group_series)
            if len(group_series) > 0
            else 0
        )
        group_id_str = (
            group_identifier.strftime("%Y-%m-%d")
            if isinstance(group_identifier, pd.Timestamp)
            else str(group_identifier)
        )
        if missing_ratio > err_thresh:
            raise ValueError(
                f"Error (GroupedFillMissing): Column '{col_name}' in group '{group_by_col}={group_id_str}' has {missing_ratio * 100:.2f}% missing, exceeding error threshold {err_thresh * 100:.2f}%."
            )
        if missing_ratio > warn_thresh:
            msg = f"Warning (GroupedFillMissing): Column '{col_name}' in group '{group_by_col}={group_id_str}' has {missing_ratio * 100:.2f}% missing, exceeding warning threshold {warn_thresh * 100:.2f}%."
            warnings.warn(msg, UserWarning)
            print(msg)

    def _find_actual_column_name(
        self,
        df: pd.DataFrame,
        base_col_name: str,
        all_sources_cfg: List[SourceConfig],
        source_rename_maps: Dict[str, Dict[str, str]],
        preferred_level_for_inference: Optional[str] = None,
    ) -> Optional[str]:
        """
        Finds the actual (potentially suffixed) column name in the DataFrame.
        If preferred_level_for_inference is given (e.g. "macro", "firm"), it prioritizes that suffix.
        """
        base_col_l = base_col_name.lower()

        # 1. If preferred_level_for_inference is given (typically when infer_suffix=true for expand_cartesian)
        if preferred_level_for_inference:
            auto_suffix = f"_{preferred_level_for_inference}"
            suffixed_name_attempt = f"{base_col_l}{auto_suffix}"
            if suffixed_name_attempt in df.columns:
                return suffixed_name_attempt
            # Fallback: if user provided a name that already IS suffixed (e.g. "dp_macro")
            # and that exists, use it. This handles cases where infer_suffix=true but user is already specific.
            if base_col_l in df.columns:
                # Check if this base_col_l was indeed a result of suffixing from the preferred_level
                for src_name, rename_map in source_rename_maps.items():
                    src_cfg = next(
                        (s for s in all_sources_cfg if s.name == src_name), None
                    )
                    if src_cfg and src_cfg.level == preferred_level_for_inference:
                        for original, suffixed in rename_map.items():
                            if (
                                suffixed == base_col_l
                            ):  # User provided the already suffixed name
                                return base_col_l
                # If not from preferred level, but exists, it might be an unsuffixed join key or from another level.
                # This case is tricky. For now, if suffixed version not found, and base_col_l exists, we'll return it.
                # A more robust system might require explicit naming if ambiguity is high.
                # print(f"  Info: Suffixed version for '{base_col_l}' with preferred_level '{preferred_level_for_inference}' not found. Checking for direct match '{base_col_l}'.")

        # 2. General check: if base_col_name itself exists (could be a join key or already suffixed by user)
        if base_col_l in df.columns:
            return base_col_l

        # 3. If not found yet, and we were NOT specifically inferring for a level,
        #    try to find if it was suffixed by *any* source. This is for transformations
        #    like clean_numeric, lag, etc., where user gives a base name.
        if not preferred_level_for_inference:
            for src_name, rename_map in source_rename_maps.items():
                # Check if base_col_l was an original name in this source
                if base_col_l in rename_map:
                    actual_name = rename_map[base_col_l]
                    if actual_name in df.columns:
                        return actual_name

        return None


def aggregate_from_yaml(
    spec_path: str | Path,
) -> tuple[pd.DataFrame, AggregationConfig]:
    with open(spec_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = AggregationConfig.model_validate(raw)
    agg = DataAggregator(cfg).load()
    merged_df = agg.merge()
    transformed_df = agg.apply_transformations(merged_df)
    return transformed_df, cfg
