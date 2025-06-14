"""Data management layer for paper data."""

from pathlib import Path
import polars as pl
import logging

from paper_data.config_parser import load_config
from paper_data.ingestion.local import CSVLoader
from paper_data.wrangling.augmenter import (
    merge_datasets,
    lag_columns,
    create_macro_firm_interactions,
    create_macro_firm_interactions_lazy,
    create_dummies,
)
from paper_data.wrangling.cleaner import (
    impute_monthly,
    scale_to_range,
)

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages data ingestion, wrangling, and export based on a YAML configuration.
    """

    def __init__(self, config_path: str | Path):
        """
        Initializes the DataManager with a path to the data configuration file.

        Args:
            config_path: The path to the data configuration YAML file.
        """
        self.config = load_config(config_path)
        self.datasets: dict[str, pl.DataFrame] = {}
        self.lazy_datasets: dict[str, pl.LazyFrame] = {}
        self._project_root: Path | None = None  # To be set by the run method
        self._ingestion_metadata: dict[
            str, dict
        ] = {}  # Stores metadata like date_column, id_column

    def _resolve_data_path(self, relative_path: str) -> Path:
        """Resolves a relative data path against the project's raw data directory."""
        if self._project_root is None:
            raise ValueError("Project root must be set before resolving data paths.")
        # Assumes raw data is in 'data/raw' relative to project root
        return self._project_root / "data" / "raw" / relative_path

    def _ingest_data(self):
        """
        Ingests data into Polars DataFrames based on the 'ingestion' section of the config.
        Applies transformations like column lowercasing and date parsing.
        """
        for dataset_config in self.config.get("ingestion", []):
            name = dataset_config["name"]
            path = dataset_config["path"]
            data_format = dataset_config["format"]
            date_column_config = dataset_config.get("date_column", {})
            to_lowercase_cols = dataset_config.get("to_lowercase_cols", False)

            if not date_column_config:
                raise ValueError(
                    f"Dataset '{name}' in ingestion config is missing 'date_column'."
                )

            date_col_name = list(date_column_config.keys())[0]
            date_col_format = date_column_config[date_col_name]

            # Prioritize 'firm_id_column', then 'id_column', then fall back to heuristics.
            id_col = dataset_config.get("firm_id_column") or dataset_config.get(
                "id_column"
            )

            if not id_col:
                if "firm" in name:
                    id_col = "permno"
                    logger.info(
                        f"For dataset '{name}', 'id_col' was inferred as 'permno'. Consider adding 'firm_id_column' to the config for clarity."
                    )
                else:
                    # For non-firm datasets like macro, the date is the identifier.
                    id_col = date_col_name
                    logger.info(
                        f"Dataset '{name}' does not have a specific ID column defined. Using date column '{date_col_name}' as a fallback."
                    )

            if data_format == "csv":
                full_path = self._resolve_data_path(path)
                # Pass date_col and id_col to CSVLoader constructor
                loader = CSVLoader(
                    path=full_path, date_col=date_col_name, id_col=id_col
                )
                # Pass date_format to get_data method
                df = loader.get_data(date_format=date_col_format)

                # Apply transformations
                if to_lowercase_cols:
                    df = df.rename({col: col.lower() for col in df.columns})

                self.datasets[name] = df
                # Store date column name and id column name for later wrangling steps
                self._ingestion_metadata[name] = {
                    "date_column": date_col_name,
                    "id_column": id_col,
                }
            else:
                raise NotImplementedError(
                    f"Ingestion format '{data_format}' not supported yet."
                )

    def _wrangle_data(self):
        """
        Performs data wrangling operations based on the 'wrangling_pipeline' section.
        Delegates 'merge' operation to augmenter.py.
        """
        for i, operation_config in enumerate(self.config.get("wrangling_pipeline", [])):
            operation_type = operation_config["operation"]
            logger.info(f"--- Wrangling Step {i + 1}: {operation_type} ---")

            dataset_name = operation_config.get(
                "dataset"
            )  # Get dataset name for operations that need it

            # Retrieve date column name and id column name from ingestion metadata
            date_col = None
            id_col = None
            if dataset_name and dataset_name in self._ingestion_metadata:
                date_col = self._ingestion_metadata[dataset_name]["date_column"]
                id_col = self._ingestion_metadata[dataset_name]["id_column"]
            # For merge operations, the date column might be in left_dataset, ensure consistency
            elif operation_type == "merge":
                left_dataset_name = operation_config["left_dataset"]
                if left_dataset_name in self._ingestion_metadata:
                    date_col = self._ingestion_metadata[left_dataset_name][
                        "date_column"
                    ]
                    id_col = self._ingestion_metadata[left_dataset_name]["id_column"]

            if operation_type == "monthly_imputation":
                numeric_columns = operation_config.get("numeric_columns", [])
                categorical_columns = operation_config.get("categorical_columns", [])
                output_name = operation_config["output_name"]
                fallback_to_zero = operation_config.get("fallback_to_zero", False)

                if dataset_name not in self.datasets:
                    raise ValueError(
                        f"Dataset '{dataset_name}' not found for monthly_imputation operation."
                    )
                if not date_col:
                    raise ValueError(
                        f"Date column information for dataset '{dataset_name}' not found. Cannot perform monthly_imputation without date column info."
                    )

                df_to_impute = self.datasets[dataset_name]

                logger.info(f"  Input Dataset: '{dataset_name}'")
                logger.info(f"  Numeric Columns: {numeric_columns}")
                logger.info(f"  Categorical Columns: {categorical_columns}")
                logger.info(f"  Fallback to Zero: {fallback_to_zero}")
                logger.info(f"  Output Dataset: '{output_name}'")

                imputed_df = impute_monthly(
                    df_to_impute,
                    date_col,
                    numeric_columns,
                    categorical_columns,
                    fallback_to_zero=fallback_to_zero,
                )
                self.datasets[output_name] = imputed_df
                # Update metadata for the new dataset
                self._ingestion_metadata[output_name] = {
                    "date_column": date_col,
                    "id_column": id_col,
                }
                logger.info(
                    f"  -> Monthly imputation complete. New dataset '{output_name}' shape: {imputed_df.shape}"
                )

            elif operation_type == "scale_to_range":
                cols_to_scale = operation_config.get("cols_to_scale", [])
                range_config = operation_config.get("range", {})
                output_name = operation_config["output_name"]

                if dataset_name not in self.datasets:
                    raise ValueError(
                        f"Dataset '{dataset_name}' not found for scale_to_range operation."
                    )
                if not date_col:
                    raise ValueError(
                        f"Date column information for dataset '{dataset_name}' not found. Cannot perform scale_to_range without date column info."
                    )
                if not cols_to_scale:
                    raise ValueError(
                        f"Operation 'scale_to_range' for dataset '{dataset_name}' is missing 'cols_to_scale'."
                    )
                if "min" not in range_config or "max" not in range_config:
                    raise ValueError(
                        f"Operation 'scale_to_range' for dataset '{dataset_name}' is missing 'range' configuration (min/max)."
                    )

                df_to_scale = self.datasets[dataset_name]
                target_min = float(range_config["min"])
                target_max = float(range_config["max"])

                logger.info(f"  Input Dataset: '{dataset_name}'")
                logger.info(f"  Columns to Scale: {cols_to_scale}")
                logger.info(f"  Target Range: [{target_min}, {target_max}]")
                logger.info(f"  Output Dataset: '{output_name}'")

                scaled_df = scale_to_range(
                    df_to_scale, cols_to_scale, date_col, target_min, target_max
                )
                self.datasets[output_name] = scaled_df
                # Update metadata for the new dataset
                self._ingestion_metadata[output_name] = {
                    "date_column": date_col,
                    "id_column": id_col,
                }
                logger.info(
                    f"  -> Scaling complete. New dataset '{output_name}' shape: {scaled_df.shape}"
                )

            elif operation_type == "merge":
                left_dataset_name = operation_config["left_dataset"]
                right_dataset_name = operation_config["right_dataset"]
                on_cols = operation_config["on"]
                how = operation_config["how"]
                output_name = operation_config["output_name"]

                if left_dataset_name not in self.datasets:
                    raise ValueError(
                        f"Left dataset '{left_dataset_name}' not found for merge operation."
                    )
                if right_dataset_name not in self.datasets:
                    raise ValueError(
                        f"Right dataset '{right_dataset_name}' not found for merge operation."
                    )

                left_df = self.datasets[left_dataset_name]
                right_df = self.datasets[right_dataset_name]

                logger.info(
                    f"  Left Dataset: '{left_dataset_name}' (Shape: {left_df.shape})"
                )
                logger.info(
                    f"  Right Dataset: '{right_dataset_name}' (Shape: {right_df.shape})"
                )
                logger.info(f"  Join Keys (on): {on_cols}")
                logger.info(f"  Join Type (how): '{how}'")
                logger.info(f"  Output Dataset: '{output_name}'")

                merged_df = merge_datasets(left_df, right_df, on_cols, how)
                self.datasets[output_name] = merged_df
                # Inherit date_column and id_column metadata for the merged dataset from the left dataset
                if left_dataset_name in self._ingestion_metadata:
                    self._ingestion_metadata[output_name] = self._ingestion_metadata[
                        left_dataset_name
                    ]
                logger.info(
                    f"  -> Merge complete. New dataset '{output_name}' shape: {merged_df.shape}"
                )

            elif operation_type == "lag":
                periods = operation_config["periods"]
                columns_to_lag_config = operation_config["columns_to_lag"]
                drop_original_cols_after_lag = operation_config.get(
                    "drop_original_cols_after_lag", False
                )
                restore_names = operation_config.get("restore_names", False)
                drop_generated_nans = operation_config.get("drop_generated_nans", False)
                output_name = operation_config["output_name"]

                if dataset_name not in self.datasets:
                    raise ValueError(
                        f"Dataset '{dataset_name}' not found for lag operation."
                    )
                if not date_col:
                    raise ValueError(
                        f"Date column information for dataset '{dataset_name}' not found. Cannot perform lag without date column info."
                    )

                if periods < 1:
                    raise ValueError(
                        f"Lag operation currently only supports periods greater or equal to 1. Found '{periods}'."
                    )

                if restore_names and not drop_original_cols_after_lag:
                    raise ValueError(
                        "Configuration Error for 'lag' operation: "
                        "If 'restore_names' is true, 'drop_original_cols_after_lag' must also be true "
                        "to avoid column name conflicts."
                    )

                df_to_lag = self.datasets[dataset_name]
                all_cols = df_to_lag.columns

                lag_method = columns_to_lag_config[0]["method"]
                specified_cols = columns_to_lag_config[1]["columns"]

                cols_to_lag = []
                if lag_method == "all_except":
                    cols_to_lag = [col for col in all_cols if col not in specified_cols]
                else:
                    raise ValueError(
                        f"Unsupported lag method: '{lag_method}'. Only 'all_except' is currently supported."
                    )

                logger.info(f"  Input Dataset: '{dataset_name}'")
                logger.info(f"  Periods: {periods}")
                logger.info(f"  Columns to Lag: {cols_to_lag}")
                logger.info(f"  Drop Original Columns: {drop_original_cols_after_lag}")
                logger.info(f"  Restore Names: {restore_names}")
                logger.info(f"  Drop Generated NaNs: {drop_generated_nans}")
                logger.info(f"  Output Dataset: '{output_name}'")

                # For time-series data (like macro factors), the id_col might be the same as date_col.
                # In this case, we must treat it as a simple time-series lag (id_col=None),
                # not a panel lag grouped by each unique date.
                id_col_for_lag = id_col
                if id_col == date_col:
                    logger.info(
                        f"Identifier column ('{id_col}') is the same as the date column ('{date_col}'). "
                        "Treating as a time-series lag (no panel grouping)."
                    )
                    id_col_for_lag = None

                lagged_df = lag_columns(
                    df_to_lag,
                    date_col,
                    id_col_for_lag,  # Use the corrected id_col
                    cols_to_lag,
                    periods,
                    drop_original_cols_after_lag,
                    restore_names,
                    drop_generated_nans,
                )
                self.datasets[output_name] = lagged_df
                # Update metadata for the new dataset
                self._ingestion_metadata[output_name] = {
                    "date_column": date_col,
                    "id_column": id_col,
                }
                logger.info(
                    f"  -> Lag operation complete. New dataset '{output_name}' shape: {lagged_df.shape}"
                )

            elif operation_type == "dummy_generation":
                column_to_dummy = operation_config.get("column_to_dummy")
                drop_original_col = operation_config.get("drop_original_col", False)
                output_name = operation_config["output_name"]

                if dataset_name not in self.datasets:
                    raise ValueError(
                        f"Dataset '{dataset_name}' not found for dummy_generation operation."
                    )
                if not column_to_dummy:
                    raise ValueError(
                        f"Operation 'dummy_generation' for dataset '{dataset_name}' is missing 'column_to_dummy'."
                    )

                df_to_dummy = self.datasets[dataset_name]

                logger.info(f"  Input Dataset: '{dataset_name}'")
                logger.info(f"  Column to Dummy: '{column_to_dummy}'")
                logger.info(f"  Drop Original Column: {drop_original_col}")
                logger.info(f"  Output Dataset: '{output_name}'")

                dummied_df = create_dummies(
                    df_to_dummy, column_to_dummy, drop_original_col
                )

                self.datasets[output_name] = dummied_df
                # Inherit metadata from the input dataset
                if dataset_name in self._ingestion_metadata:
                    self._ingestion_metadata[output_name] = self._ingestion_metadata[
                        dataset_name
                    ]

                logger.info(
                    f"  -> Dummy generation complete. New dataset '{output_name}' shape: {dummied_df.shape}"
                )

            elif operation_type == "create_macro_interactions":
                macro_columns = operation_config.get("macro_columns", [])
                firm_columns = operation_config.get("firm_columns", [])
                drop_macro_columns = operation_config.get("drop_macro_columns", False)
                output_name = operation_config["output_name"]
                use_lazy_engine = operation_config.get("use_lazy_engine", False)

                if dataset_name not in self.datasets:
                    raise ValueError(
                        f"Dataset '{dataset_name}' not found for create_macro_interactions operation."
                    )
                if not macro_columns:
                    raise ValueError(
                        f"Operation 'create_macro_interactions' for dataset '{dataset_name}' is missing 'macro_columns'."
                    )
                if not firm_columns:
                    raise ValueError(
                        f"Operation 'create_macro_interactions' for dataset '{dataset_name}' is missing 'firm_columns'."
                    )

                df_to_interact = self.datasets[dataset_name]

                logger.info(f"  Input Dataset: '{dataset_name}'")
                logger.info(f"  Macro Columns: {macro_columns}")
                logger.info(f"  Firm Columns: {firm_columns}")
                logger.info(f"  Drop Macro Columns: {drop_macro_columns}")
                logger.info(f"  Output Dataset: '{output_name}'")
                logger.info(f"  Use Lazy Engine: {use_lazy_engine}")

                # Inherit metadata for the new dataset, regardless of engine
                if dataset_name in self._ingestion_metadata:
                    self._ingestion_metadata[output_name] = self._ingestion_metadata[
                        dataset_name
                    ]

                if use_lazy_engine:
                    logger.info(
                        "Using lazy engine. Computation will be deferred until export."
                    )
                    ldf = df_to_interact.lazy()
                    interactions_ldf = create_macro_firm_interactions_lazy(
                        ldf, macro_columns, firm_columns, drop_macro_columns
                    )
                    self.lazy_datasets[output_name] = interactions_ldf
                    logger.info(
                        f"  -> Lazy macro-firm interaction plan created for '{output_name}'."
                    )
                else:
                    logger.info("Using eager engine. Processing data in memory.")
                    interactions_df = create_macro_firm_interactions(
                        df_to_interact,
                        macro_columns,
                        firm_columns,
                        drop_macro_columns,
                    )
                    self.datasets[output_name] = interactions_df
                    logger.info(
                        f"  -> Eager macro-firm interaction creation complete. New dataset '{output_name}' shape: {interactions_df.shape}"
                    )

            else:
                raise NotImplementedError(
                    f"Wrangling operation '{operation_type}' not supported yet."
                )

    def _export_parquet_partitioned_by_year(
        self,
        df_to_export: pl.DataFrame,
        output_dir: Path,
        output_filename_base: str,
        dataset_name: str,
    ):
        """
        Exports an eager Polars DataFrame to Parquet, partitioned by year.
        Creates files named 'output_filename_base_YYYY.parquet'.
        """
        date_col_for_export = self._ingestion_metadata.get(dataset_name, {}).get(
            "date_column"
        )
        if (
            not date_col_for_export
            or date_col_for_export not in df_to_export.columns
            or not isinstance(
                df_to_export[date_col_for_export].dtype, (pl.Date, pl.Datetime)
            )
        ):
            raise ValueError(
                f"Cannot partition by 'year'. Dataset '{dataset_name}' does not have a recognized date column of type Date or Datetime."
            )

        temp_year_col = "__temp_year__"
        df_for_partitioning = df_to_export.with_columns(
            pl.col(date_col_for_export).dt.year().alias(temp_year_col)
        )

        unique_years = df_for_partitioning[temp_year_col].unique().sort().to_list()
        logger.info(
            f"Exporting '{dataset_name}' by year to separate files: {unique_years}"
        )

        for year in unique_years:
            df_year = df_for_partitioning.filter(pl.col(temp_year_col) == year)
            file_path = output_dir / f"{output_filename_base}_{year}.parquet"
            df_year_to_write = df_year.drop(temp_year_col)
            df_year_to_write.write_parquet(file_path)
            logger.info(
                f"  Exported data for year {year} to '{file_path}'. Final shape for year: {df_year_to_write.shape}"
            )

    def _export_lazy_parquet_partitioned_by_year(
        self,
        ldf_to_export: pl.LazyFrame,
        output_dir: Path,
        output_filename_base: str,
        dataset_name: str,
    ):
        """
        Exports a lazy Polars DataFrame to Parquet, partitioned by year,
        by streaming the output year by year to avoid high memory usage.
        """
        date_col_for_export = self._ingestion_metadata.get(dataset_name, {}).get(
            "date_column"
        )
        if (
            not date_col_for_export
            or date_col_for_export not in ldf_to_export.collect_schema().names()
        ):
            raise ValueError(
                f"Cannot partition by 'year'. Lazy dataset '{dataset_name}' is missing date column '{date_col_for_export}'."
            )

        # Get unique years without collecting the full dataset
        unique_years_df = ldf_to_export.select(
            pl.col(date_col_for_export).dt.year().alias("year")
        ).unique()
        unique_years = unique_years_df.collect()["year"].sort().to_list()

        logger.info(
            f"Streaming lazy export of '{dataset_name}' by year: {unique_years}"
        )

        for year in unique_years:
            file_path = output_dir / f"{output_filename_base}_{year}.parquet"
            ldf_year = ldf_to_export.filter(
                pl.col(date_col_for_export).dt.year() == year
            )

            logger.info(f"  Executing query and writing data for year {year}...")
            ldf_year.sink_parquet(file_path)
            written_df = pl.read_parquet(file_path)
            logger.info(
                f"  -> Exported data for year {year} to '{file_path}'. Final shape for year: {written_df.shape}"
            )

    def _export_data(self):
        """
        Exports processed dataframes based on the 'export' section of the config.
        Supports both eager (in-memory) and lazy (streaming) datasets.
        """
        if self._project_root is None:
            raise ValueError("Project root must be set before exporting data.")

        output_dir = self._project_root / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        for export_config in self.config.get("export", []):
            dataset_name = export_config["dataset_name"]
            output_filename_base = export_config["output_filename_base"]
            data_format = export_config["format"]
            partition_by = export_config.get("partition_by")

            if dataset_name in self.lazy_datasets:
                ldf_to_export = self.lazy_datasets[dataset_name]
                logger.info(f"Found lazy dataset '{dataset_name}' for export.")

                if data_format == "parquet":
                    if partition_by == "year":
                        self._export_lazy_parquet_partitioned_by_year(
                            ldf_to_export,
                            output_dir,
                            output_filename_base,
                            dataset_name,
                        )
                    elif partition_by is None or partition_by == "none":
                        output_path = output_dir / f"{output_filename_base}.parquet"
                        logger.info(
                            f"Executing query and streaming lazy export of '{dataset_name}' to '{output_path}'."
                        )
                        ldf_to_export.sink_parquet(output_path)
                        written_df = pl.read_parquet(output_path)
                        logger.info(
                            f"  -> Streaming export complete. Final shape: {written_df.shape}"
                        )
                    else:
                        raise NotImplementedError(
                            f"Partitioning by '{partition_by}' not supported for lazy export yet."
                        )
                else:
                    raise NotImplementedError(
                        f"Export format '{data_format}' not supported for lazy datasets yet."
                    )
                continue

            if dataset_name not in self.datasets:
                raise ValueError(
                    f"Dataset '{dataset_name}' not found for export (and not a pending lazy dataset)."
                )

            df_to_export = self.datasets[dataset_name]
            logger.info(f"Found eager dataset '{dataset_name}' for export.")

            if data_format == "parquet":
                if partition_by == "year":
                    self._export_parquet_partitioned_by_year(
                        df_to_export, output_dir, output_filename_base, dataset_name
                    )
                elif partition_by is None or partition_by == "none":
                    output_path = output_dir / f"{output_filename_base}.parquet"
                    df_to_export.write_parquet(output_path)
                    logger.info(
                        f"Exported '{dataset_name}' to '{output_path}'. Final shape: {df_to_export.shape}"
                    )
                else:
                    raise NotImplementedError(
                        f"Partitioning by '{partition_by}' not supported yet."
                    )
            else:
                raise NotImplementedError(
                    f"Export format '{data_format}' not supported yet."
                )

    def run(self, project_root: str | Path) -> dict[str, pl.DataFrame]:
        """
        Executes the data pipeline: ingestion, wrangling, and export.

        Args:
            project_root: The root directory of the PAPER project (e.g., 'PAPER/ThesisExample').

        Returns:
            A dictionary of the final processed Polars DataFrames.
            Note: Lazily processed datasets are not returned as they are streamed to disk, not held in memory.
        """
        self._project_root = Path(project_root).expanduser()
        logger.info(f"Running data pipeline for project: {self._project_root}")

        logger.info("--- Ingesting Data ---")
        self._ingest_data()
        for name, df in self.datasets.items():
            logger.info(f"Dataset '{name}' ingested. Shape: {df.shape}")

        logger.info("--- Wrangling Data ---")
        self._wrangle_data()
        for name, df in self.datasets.items():
            logger.info(f"Dataset '{name}' after wrangling. Shape: {df.shape}")
        for name in self.lazy_datasets:
            logger.info(f"Lazy dataset '{name}' plan has been defined.")

        logger.info("--- Exporting Data ---")
        self._export_data()
        logger.info("Data pipeline completed successfully.")

        return self.datasets
