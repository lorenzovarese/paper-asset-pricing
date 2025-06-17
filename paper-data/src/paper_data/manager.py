from pathlib import Path
import polars as pl
import logging
import hashlib
from typing import Dict

from paper_data.config_parser import (
    DataConfig,
    CSVConfig,
    GoogleSheetConfig,
    WRDSConfig,
    ImputationConfig,
    ScaleConfig,
    MergeConfig,
    LagConfig,
    DummyConfig,
    InteractionConfig,
)
from paper_data.ingestion import (
    CSVLoader,
    GoogleSheetConnector,
    WRDSConnector,
)
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


def _generate_cache_hash(text: str) -> str:
    """Generates a truncated SHA256 hash for creating unique, short filenames."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class DataManager:
    """
    Manages data ingestion, wrangling, and export based on a validated configuration object.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.datasets: Dict[str, pl.DataFrame] = {}
        self.lazy_datasets: Dict[str, pl.LazyFrame] = {}
        self._project_root: Path | None = None
        self._ingestion_metadata: Dict[str, Dict] = {}

    def _resolve_data_path(self, relative_path: str) -> Path:
        if self._project_root is None:
            raise ValueError("Project root must be set before resolving data paths.")
        return self._project_root / "data" / "raw" / relative_path

    def _ingest_data(self):
        if self._project_root is None:
            raise ValueError("Project root must be set before ingesting data.")

        raw_data_dir = self._project_root / "data" / "raw"
        raw_data_dir.mkdir(parents=True, exist_ok=True)

        for d_config in self.config.ingestion:
            id_col = d_config.id_column or d_config.date_col_name
            df: pl.DataFrame | None = None

            if isinstance(d_config, CSVConfig):
                full_path = self._resolve_data_path(d_config.path)
                loader = CSVLoader(
                    path=full_path, date_col=d_config.date_col_name, id_col=id_col
                )
                df = loader.get_data(date_format=d_config.date_col_format)
            elif isinstance(d_config, GoogleSheetConfig):
                cache_hash = _generate_cache_hash(d_config.url)
                cache_path = raw_data_dir / f"{d_config.name}_{cache_hash}.csv"
                connector = GoogleSheetConnector(
                    url=d_config.url, cache_path=cache_path
                )
                df_raw = connector.get_data()
                df = self._post_process_loaded_df(
                    df_raw,
                    d_config.date_col_name,
                    id_col,
                    d_config.date_col_format,
                    d_config.name,
                )
            elif isinstance(d_config, WRDSConfig):
                cache_hash = _generate_cache_hash(d_config.query)
                cache_path = raw_data_dir / f"{d_config.name}_{cache_hash}.csv"
                creds = self.config.wrds_credentials or {}
                connector = WRDSConnector(
                    query=d_config.query,
                    cache_path=cache_path,
                    user=getattr(creds, "user", None),
                    password=getattr(creds, "password", None),
                )
                df_raw = connector.get_data()
                df = self._post_process_loaded_df(
                    df_raw,
                    d_config.date_col_name,
                    id_col,
                    d_config.date_col_format,
                    d_config.name,
                )

            if df is not None:
                if d_config.to_lowercase_cols:
                    df = df.rename({col: col.lower() for col in df.columns})
                self.datasets[d_config.name] = df
                self._ingestion_metadata[d_config.name] = {
                    "date_column": d_config.date_col_name,
                    "id_column": id_col,
                }

    def _post_process_loaded_df(
        self,
        df: pl.DataFrame,
        date_col: str,
        id_col: str,
        date_format: str | None,
        dataset_name: str,
    ) -> pl.DataFrame:
        missing = {col for col in [date_col, id_col] if col not in df.columns}
        if missing:
            raise ValueError(
                f"Missing required columns: {', '.join(missing)} for dataset '{dataset_name}'."
            )
        if date_format:
            if df[date_col].dtype.is_numeric():
                df = df.with_columns(pl.col(date_col).cast(pl.Utf8))
            df = df.with_columns(
                pl.col(date_col).str.strptime(pl.Date, format=date_format, strict=False)
            )
            if df[date_col].is_null().any():
                raise ValueError(
                    f"Failed to parse date values in column '{date_col}' for dataset '{dataset_name}'."
                )
        return df

    def _wrangle_data(self):
        for i, op in enumerate(self.config.wrangling_pipeline):
            logger.info(f"--- Wrangling Step {i + 1}: {op.operation} ---")

            dataset_name = getattr(op, "dataset", getattr(op, "left_dataset", None))
            meta = (
                self._ingestion_metadata.get(dataset_name, {}) if dataset_name else {}
            )
            date_col, id_col = meta.get("date_column"), meta.get("id_column")

            if isinstance(op, ImputationConfig):
                if not date_col:
                    raise ValueError(
                        f"Could not find date column metadata for dataset '{op.dataset}' in imputation step."
                    )
                df_to_impute = self.datasets[op.dataset]
                imputed_df = impute_monthly(
                    df_to_impute,
                    date_col,
                    op.numeric_columns,
                    op.categorical_columns,
                    op.fallback_to_zero,
                )
                self.datasets[op.output_name] = imputed_df
                self._ingestion_metadata[op.output_name] = meta

            elif isinstance(op, ScaleConfig):
                if not date_col:
                    raise ValueError(
                        f"Could not find date column metadata for dataset '{op.dataset}' in scaling step."
                    )
                df_to_scale = self.datasets[op.dataset]
                scaled_df = scale_to_range(
                    df_to_scale, op.cols_to_scale, date_col, op.range.min, op.range.max
                )
                self.datasets[op.output_name] = scaled_df
                self._ingestion_metadata[op.output_name] = meta

            elif isinstance(op, MergeConfig):
                left_df = self.datasets[op.left_dataset]
                right_df = self.datasets[op.right_dataset]
                merged_df = merge_datasets(left_df, right_df, op.on, op.how.value)
                self.datasets[op.output_name] = merged_df
                self._ingestion_metadata[op.output_name] = self._ingestion_metadata[
                    op.left_dataset
                ]

            elif isinstance(op, LagConfig):
                if not date_col:
                    raise ValueError(
                        f"Could not find date column metadata for dataset '{op.dataset}' in lagging step."
                    )
                df_to_lag = self.datasets[op.dataset]
                lag_method = op.columns_to_lag.method.value
                specified_cols = op.columns_to_lag.columns

                cols_to_lag = (
                    [col for col in df_to_lag.columns if col not in specified_cols]
                    if lag_method == "all_except"
                    else specified_cols
                )

                id_col_for_lag = id_col if id_col != date_col else None
                lagged_df = lag_columns(
                    df_to_lag,
                    date_col,
                    id_col_for_lag,
                    cols_to_lag,
                    op.periods,
                    op.drop_original_cols_after_lag,
                    op.restore_names,
                    op.drop_generated_nans,
                )
                self.datasets[op.output_name] = lagged_df
                self._ingestion_metadata[op.output_name] = meta

            elif isinstance(op, DummyConfig):
                dummied_df = create_dummies(
                    self.datasets[op.dataset], op.column_to_dummy, op.drop_original_col
                )
                self.datasets[op.output_name] = dummied_df
                self._ingestion_metadata[op.output_name] = meta

            elif isinstance(op, InteractionConfig):
                self._ingestion_metadata[op.output_name] = meta
                if op.use_lazy_engine:
                    ldf = self.datasets[op.dataset].lazy()
                    self.lazy_datasets[op.output_name] = (
                        create_macro_firm_interactions_lazy(
                            ldf,
                            op.macro_columns,
                            op.firm_columns,
                            op.drop_macro_columns,
                        )
                    )
                else:
                    self.datasets[op.output_name] = create_macro_firm_interactions(
                        self.datasets[op.dataset],
                        op.macro_columns,
                        op.firm_columns,
                        op.drop_macro_columns,
                    )
            else:
                raise NotImplementedError(
                    f"Wrangling operation '{op.operation}' not supported."
                )

    def _export_data(self):
        if self._project_root is None:
            raise ValueError("Project root must be set before exporting data.")
        output_dir = self._project_root / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        for exp_config in self.config.export:
            if exp_config.dataset_name in self.lazy_datasets:
                ldf = self.lazy_datasets[exp_config.dataset_name]
                if exp_config.partition_by and exp_config.partition_by.value == "year":
                    self._export_lazy_parquet_partitioned_by_year(
                        ldf,
                        output_dir,
                        exp_config.output_filename_base,
                        exp_config.dataset_name,
                    )
                else:
                    path = output_dir / f"{exp_config.output_filename_base}.parquet"
                    ldf.sink_parquet(path)
            elif exp_config.dataset_name in self.datasets:
                df = self.datasets[exp_config.dataset_name]
                if exp_config.partition_by and exp_config.partition_by.value == "year":
                    self._export_parquet_partitioned_by_year(
                        df,
                        output_dir,
                        exp_config.output_filename_base,
                        exp_config.dataset_name,
                    )
                else:
                    path = output_dir / f"{exp_config.output_filename_base}.parquet"
                    df.write_parquet(path)
            else:
                raise ValueError(
                    f"Dataset '{exp_config.dataset_name}' not found for export."
                )

    def _export_parquet_partitioned_by_year(
        self, df, output_dir, base_name, dataset_name
    ):
        date_col = self._ingestion_metadata[dataset_name]["date_column"]
        for year, df_year in df.group_by(df[date_col].dt.year()):
            path = output_dir / f"{base_name}_{year}.parquet"
            df_year.write_parquet(path)

    def _export_lazy_parquet_partitioned_by_year(
        self, ldf, output_dir, base_name, dataset_name
    ):
        date_col = self._ingestion_metadata[dataset_name]["date_column"]
        years = (
            ldf.select(pl.col(date_col).dt.year().unique())
            .collect()
            .to_series()
            .to_list()
        )
        for year in years:
            path = output_dir / f"{base_name}_{year}.parquet"
            ldf.filter(pl.col(date_col).dt.year() == year).sink_parquet(path)

    def run(self, project_root: str | Path) -> Dict[str, pl.DataFrame]:
        self._project_root = Path(project_root).expanduser()
        logger.info(f"Running data pipeline for project: {self._project_root}")
        self._ingest_data()
        self._wrangle_data()
        self._export_data()
        logger.info("Data pipeline completed successfully.")
        return self.datasets
