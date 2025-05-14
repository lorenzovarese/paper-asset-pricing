import typer
from pathlib import Path
import yaml
import polars as pl
from typing_extensions import Annotated
from typing import Optional
import logging

from paper_data.config import (
    PaperDataConfig,
    LocalConnectorParams,
    HTTPConnectorParams,
    GoogleDriveConnectorParams,
    HuggingFaceConnectorParams,
    WRDSConnectorParams,
    ConnectorConfig,
    SourceConfig,
    GlobalSettings,
    RenameDateColumnConfig,
    ParseDateConfig,
    CleanNumericColumnConfig,
    ImputeConstantConfig,
    ImputeCrossSectionMedianConfig,
    ImputeCrossSectionMeanConfig,
    ImputeCrossSectionModeConfig,
)
from paper_data.ingestion.base import BaseConnector
from paper_data.ingestion.local import LocalConnector
from paper_data.ingestion.http import HTTPConnector
from paper_data.ingestion.google_drive import GoogleDriveConnector
from paper_data.ingestion.huggingface_ds import HuggingFaceConnector
from paper_data.ingestion.wrds_conn import WRDSConnector
from paper_data.schema.firm import firm_schema
from paper_data.schema.macro import macro_schema as validate_macro_schema
from paper_data.wrangling.cleaner.cleaner import (
    RawDataset,
    FirmCleaner,
    MacroCleaner,
    BaseCleaner,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("paper-data")

app = typer.Typer(
    name="paper-data",
    help="P.A.P.E.R Data: Ingest, clean, and preprocess data for asset pricing research.",
    add_completion=False,
    no_args_is_help=True,
)


def _load_data_config(config_path: Path) -> PaperDataConfig:
    logger.info(f"Loading configuration from: {config_path}")
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        typer.secho(
            f"Error: Config file not found at {config_path}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
        # Use Pydantic V2's model_validate
        config = PaperDataConfig.model_validate(raw_config)
        logger.info("Configuration loaded and validated successfully.")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML in {config_path}: {e}", exc_info=True)
        typer.secho(
            f"Error parsing YAML in {config_path}: {e}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Error validating config {config_path}: {e}", exc_info=True)
        typer.secho(
            f"Error validating config {config_path}: {e}", fg=typer.colors.RED, err=True
        )
        # Print more detailed Pydantic validation errors if available
        if hasattr(e, "errors") and callable(e.errors):
            try:
                typer.secho(
                    "Pydantic validation errors:", fg=typer.colors.YELLOW, err=True
                )
                for error in e.errors():  # type: ignore
                    typer.secho(
                        f"  Loc: {error['loc']}, Msg: {error['msg']}, Type: {error['type']}",
                        fg=typer.colors.YELLOW,
                        err=True,
                    )
            except Exception:  # Fallback if errors() format is unexpected
                pass
        raise typer.Exit(code=1)


def _get_connector(connector_cfg: ConnectorConfig, project_root: Path) -> BaseConnector:
    conn_params = (
        connector_cfg.params
    )  # This is already the specific Pydantic model instance
    if connector_cfg.type == "local":
        assert isinstance(conn_params, LocalConnectorParams)
        local_path = project_root / conn_params.path
        if not local_path.exists():
            msg = f"Local path {local_path} for connector '{connector_cfg.type}' does not exist."
            logger.error(msg)
            raise FileNotFoundError(msg)
        return LocalConnector(path=local_path, member_name=conn_params.member_name)
    elif connector_cfg.type == "http":
        assert isinstance(conn_params, HTTPConnectorParams)
        return HTTPConnector(url=conn_params.url, timeout=conn_params.timeout)
    elif connector_cfg.type == "gdrive":
        assert isinstance(conn_params, GoogleDriveConnectorParams)
        return GoogleDriveConnector(drive_url=conn_params.drive_url)
    elif connector_cfg.type == "huggingface":
        assert isinstance(conn_params, HuggingFaceConnectorParams)
        return HuggingFaceConnector(
            repo_id=conn_params.repo_id,
            split=conn_params.split,
            **conn_params.load_kwargs,
        )
    elif connector_cfg.type == "wrds":
        assert isinstance(conn_params, WRDSConnectorParams)
        return WRDSConnector(query=conn_params.query, max_rows=conn_params.max_rows)
    else:
        # This case should be caught by Pydantic validation of ConnectorConfig.type
        msg = f"Internal Error: Unsupported connector type '{connector_cfg.type}' reached _get_connector."
        logger.error(msg)
        raise ValueError(msg)


def _apply_transformations(
    df: pl.DataFrame, source_cfg: SourceConfig, global_cfg_settings: GlobalSettings
) -> pl.DataFrame:
    if not source_cfg.transformations:
        logger.info(f"No transformations to apply for source '{source_cfg.name}'.")
        return df

    logger.info(
        f"Applying {len(source_cfg.transformations)} transformations for source '{source_cfg.name}'..."
    )
    raw_dataset = RawDataset(df.clone(), objective=source_cfg.objective_for_cleaner)

    cleaner: BaseCleaner
    if source_cfg.objective_for_cleaner == "firm":
        date_col = source_cfg.date_col_for_firm_cleaner
        id_col = source_cfg.id_col_for_firm_cleaner
        cleaner = FirmCleaner(raw_dataset, date_col=date_col, id_col=id_col)
    elif source_cfg.objective_for_cleaner == "macro":
        date_col = global_cfg_settings.default_date_col_for_cleaners
        cleaner = MacroCleaner(raw_dataset, date_col=date_col)
    else:
        # Should be caught by Pydantic validation of SourceConfig.objective_for_cleaner
        raise ValueError(
            f"Internal Error: Invalid objective_for_cleaner '{source_cfg.objective_for_cleaner}'."
        )

    for i, trans_cfg in enumerate(
        source_cfg.transformations
    ):  # trans_cfg is already the specific model instance
        logger.info(
            f"  Step {i + 1}/{len(source_cfg.transformations)}: Applying transformation: {trans_cfg.type}"
        )

        try:
            if trans_cfg.type == "normalize_columns":
                cleaner.normalize_columns()
            elif trans_cfg.type == "rename_date_column":
                assert isinstance(trans_cfg, RenameDateColumnConfig)
                cleaner.rename_date_column(
                    candidates=trans_cfg.candidates, target=trans_cfg.target
                )
            elif trans_cfg.type == "parse_date":
                assert isinstance(trans_cfg, ParseDateConfig)
                cleaner.parse_date(
                    date_col=trans_cfg.date_col,
                    date_format=trans_cfg.date_format,
                    monthly_option=trans_cfg.monthly_option,
                )
            elif trans_cfg.type == "clean_numeric_column":
                assert isinstance(trans_cfg, CleanNumericColumnConfig)
                cleaner.clean_numeric_column(col=trans_cfg.col)
            elif trans_cfg.type == "impute_constant":
                assert isinstance(trans_cfg, ImputeConstantConfig)
                cleaner.impute_constant(cols=trans_cfg.cols, value=trans_cfg.value)
            elif trans_cfg.type == "impute_cross_section_median":
                assert isinstance(trans_cfg, ImputeCrossSectionMedianConfig)
                if isinstance(cleaner, FirmCleaner):
                    cleaner.impute_cross_section_median(cols=trans_cfg.cols)
                else:
                    logger.warning(
                        f"Transformation '{trans_cfg.type}' is for 'firm' objective. Current: '{source_cfg.objective_for_cleaner}'. Skipping."
                    )
            elif trans_cfg.type == "impute_cross_section_mean":
                assert isinstance(trans_cfg, ImputeCrossSectionMeanConfig)
                if isinstance(cleaner, FirmCleaner):
                    cleaner.impute_cross_section_mean(cols=trans_cfg.cols)
                else:
                    logger.warning(
                        f"Transformation '{trans_cfg.type}' is for 'firm' objective. Current: '{source_cfg.objective_for_cleaner}'. Skipping."
                    )
            elif trans_cfg.type == "impute_cross_section_mode":
                assert isinstance(trans_cfg, ImputeCrossSectionModeConfig)
                if isinstance(cleaner, FirmCleaner):
                    cleaner.impute_cross_section_mode(cols=trans_cfg.cols)
                else:
                    logger.warning(
                        f"Transformation '{trans_cfg.type}' is for 'firm' objective. Current: '{source_cfg.objective_for_cleaner}'. Skipping."
                    )
            # No 'else' needed here as Pydantic's discriminated union validation in SourceConfig
            # should ensure trans_cfg is one of the known types.
        except Exception as e:
            logger.error(
                f"Error during transformation '{trans_cfg.type}' for source '{source_cfg.name}': {e}",
                exc_info=True,
            )
            raise
    return cleaner.df


@app.command()
def process(
    config_file: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to the data-config.yaml file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    project_root: Annotated[
        Path,
        typer.Option(
            "--project-root",
            help="Path to the P.A.P.E.R project root directory.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ],
):
    """Process data sources as defined in the data-config.yaml file."""
    typer.secho(
        f"--- P.A.P.E.R. Data Processing ---", fg=typer.colors.BRIGHT_BLUE, bold=True
    )
    logger.info(
        f"Starting paper-data processing. Project root: {project_root}, Config: {config_file}"
    )

    cfg = _load_data_config(config_file)

    global_output_dir = project_root / cfg.global_settings.output_dir
    global_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Global output directory: {global_output_dir}")
    typer.secho(
        f"Global output directory: {global_output_dir.relative_to(project_root)}",
        fg=typer.colors.BLUE,
    )

    for source_idx, source_cfg in enumerate(cfg.sources):
        typer.secho(
            f"\n[{source_idx + 1}/{len(cfg.sources)}] Processing source: {source_cfg.name}",
            fg=typer.colors.MAGENTA,
            bold=True,
        )
        logger.info(f"Processing source: {source_cfg.name}")

        current_df: Optional[pl.DataFrame] = None
        try:
            # 1. Fetch Data
            logger.info(
                f"  1. Fetching data using '{source_cfg.connector.type}' connector..."
            )
            connector = _get_connector(source_cfg.connector, project_root)
            current_df = connector.get_data()
            logger.info(
                f"  Fetched data. Shape: {current_df.shape}, Columns: {current_df.columns[:5]}..."
            )  # Log first 5 cols
            typer.secho(
                f"  Fetched data. Shape: {current_df.shape}", fg=typer.colors.GREEN
            )

            # 2. Schema Validation
            if source_cfg.schema_validation:
                logger.info(
                    f"  2. Validating schema: {source_cfg.schema_validation.type}..."
                )
                if source_cfg.schema_validation.type == "firm":
                    current_df = firm_schema(current_df)
                elif source_cfg.schema_validation.type == "macro":
                    current_df = validate_macro_schema(current_df)
                logger.info(f"  Schema validation successful.")
                typer.secho(f"  Schema validation successful.", fg=typer.colors.GREEN)
            else:
                logger.info(
                    f"  2. Schema validation skipped for source '{source_cfg.name}'."
                )

            # 3. Transformations
            logger.info(f"  3. Applying transformations...")
            current_df = _apply_transformations(
                current_df, source_cfg, cfg.global_settings
            )
            logger.info(
                f"  Transformations applied. New shape: {current_df.shape}, Columns: {current_df.columns[:5]}..."
            )
            typer.secho(
                f"  Transformations applied. New shape: {current_df.shape}",
                fg=typer.colors.GREEN,
            )

            # 4. Save Output
            output_path = global_output_dir / source_cfg.output.filename
            logger.info(
                f"  4. Saving processed data to: {output_path} (format: {source_cfg.output.format})"
            )

            if source_cfg.output.format == "parquet":
                current_df.write_parquet(output_path)
            elif source_cfg.output.format == "csv":
                current_df.write_csv(output_path)
            elif source_cfg.output.format == "feather":
                current_df.write_ipc(output_path)
            elif source_cfg.output.format == "json":
                current_df.write_json(output_path, row_oriented=True)
            # Pydantic validation of OutputConfig.format should catch unsupported formats

            logger.info(
                f"  Successfully saved processed data for source '{source_cfg.name}'."
            )
            typer.secho(
                f"  Successfully saved: {output_path.relative_to(project_root)}",
                fg=typer.colors.GREEN,
            )

        except FileNotFoundError as e:
            logger.error(
                f"File not found error for source '{source_cfg.name}': {e}",
                exc_info=True,
            )
            typer.secho(
                f"  Error (File Not Found) for source '{source_cfg.name}': {e}. Skipping this source.",
                fg=typer.colors.RED,
                err=True,
            )
            continue
        except Exception as e:
            logger.error(
                f"Failed to process source '{source_cfg.name}': {e}", exc_info=True
            )
            typer.secho(
                f"  Error processing source '{source_cfg.name}': {e}. Skipping this source.",
                fg=typer.colors.RED,
                err=True,
            )
            if hasattr(e, "errors") and callable(e.errors):  # For Pydantic errors
                try:
                    typer.secho(
                        "  Pydantic validation errors:",
                        fg=typer.colors.YELLOW,
                        err=True,
                    )
                    for error in e.errors():  # type: ignore
                        typer.secho(
                            f"    Loc: {error['loc']}, Msg: {error['msg']}, Type: {error['type']}",
                            fg=typer.colors.YELLOW,
                            err=True,
                        )
                except Exception:
                    pass
            continue

    logger.info("paper-data processing finished.")
    typer.secho(
        "\n--- P.A.P.E.R. Data Processing Finished ---",
        fg=typer.colors.BRIGHT_BLUE,
        bold=True,
    )


if __name__ == "__main__":
    app()
