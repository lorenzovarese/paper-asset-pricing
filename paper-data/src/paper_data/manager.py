"""Data management layer for paper data."""

from pathlib import Path
import polars as pl

from paper_data.config_parser import load_config
from paper_data.ingestion.local import CSVLoader  # Only local CSV ingestion for now
from paper_data.wrangling.augmenter import (
    merge_datasets,
)


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
        self._project_root: Path | None = None  # To be set by the run method

    def _resolve_data_path(self, relative_path: str) -> Path:
        """Resolves a relative data path against the project's raw data directory."""
        if self._project_root is None:
            raise ValueError("Project root must be set before resolving data paths.")
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

            # Determine id_col based on dataset name, as it's required by CSVLoader
            id_col = None
            if name == "firm":
                id_col = "permco"
            elif name == "macro":
                # For macro data, 'date' can serve as a unique identifier per row
                # in the absence of a specific entity ID like 'permco'.
                id_col = "date"
                print(
                    f"Warning: For dataset '{name}', 'id_col' is not specified in config. Using '{id_col}' as a fallback. Ensure this is appropriate for your data."
                )
            else:
                raise ValueError(
                    f"Dataset '{name}' requires an 'id_col' but it's not specified in config and no default is available."
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
            else:
                raise NotImplementedError(
                    f"Ingestion format '{data_format}' not supported yet."
                )

    def _wrangle_data(self):
        """
        Performs data wrangling operations based on the 'wrangling_pipeline' section.
        Delegates 'merge' operation to augmenter.py.
        """
        for operation_config in self.config.get("wrangling_pipeline", []):
            operation_type = operation_config["operation"]

            if operation_type == "merge":
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

                # Call the merge function from augmenter.py
                merged_df = merge_datasets(left_df, right_df, on_cols, how)
                self.datasets[output_name] = merged_df
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
        Exports a Polars DataFrame to Parquet, partitioned by year.
        Creates 'year=YYYY' subdirectories.
        """
        # Ensure 'date' column exists and is of Date type for year extraction
        if "date" not in df_to_export.columns or not isinstance(
            df_to_export["date"].dtype, pl.Date
        ):
            raise ValueError(
                f"Cannot partition by 'year'. Dataset '{dataset_name}' does not have a 'date' column of type Date."
            )

        # Add a 'year' column for partitioning if not already present
        if "year" not in df_to_export.columns:
            df_to_export = df_to_export.with_columns(
                pl.col("date").dt.year().alias("year")
            )

        unique_years = df_to_export["year"].unique().sort().to_list()
        print(f"Partitioning '{dataset_name}' by year: {unique_years}")

        for year in unique_years:
            year_dir = output_dir / f"year={year}"
            year_dir.mkdir(parents=True, exist_ok=True)

            # Filter data for the current year
            df_year = df_to_export.filter(pl.col("year") == year)

            # Define the output path for the current year's data
            file_path = year_dir / f"{output_filename_base}.parquet"

            # Write the filtered DataFrame to the specific year directory
            df_year.write_parquet(file_path)
            print(f"  Exported data for year {year} to '{file_path}'.")

    def _export_data(self):
        """
        Exports processed dataframes based on the 'export' section of the config.
        Currently supports 'parquet' format with 'year' or 'null' partitioning.
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

            if dataset_name not in self.datasets:
                raise ValueError(f"Dataset '{dataset_name}' not found for export.")

            df_to_export = self.datasets[dataset_name]

            if data_format == "parquet":
                if partition_by == "year":
                    self._export_parquet_partitioned_by_year(
                        df_to_export, output_dir, output_filename_base, dataset_name
                    )
                elif partition_by is None or partition_by == "null":
                    output_path = output_dir / f"{output_filename_base}.parquet"
                    df_to_export.write_parquet(output_path)
                    print(f"Exported '{dataset_name}' to '{output_path}'.")
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
        """
        self._project_root = Path(project_root).expanduser()
        print(f"Running data pipeline for project: {self._project_root}")

        print("--- Ingesting Data ---")
        self._ingest_data()
        for name, df in self.datasets.items():
            print(f"Dataset '{name}' ingested. Shape: {df.shape}")

        print("\n--- Wrangling Data ---")
        self._wrangle_data()
        for name, df in self.datasets.items():
            print(f"Dataset '{name}' after wrangling. Shape: {df.shape}")

        print("\n--- Exporting Data ---")
        self._export_data()
        print("Data pipeline completed successfully.")

        return self.datasets
