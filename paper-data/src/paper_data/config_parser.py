from pathlib import Path
import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from typing import Annotated, List, Literal, Optional, Union, Dict
from enum import Enum


# --- Enums for controlled vocabularies ---
class IngestionFormat(str, Enum):
    CSV = "csv"
    GOOGLE_SHEET = "google_sheet"
    WRDS = "wrds"


class MergeStrategy(str, Enum):
    LEFT = "left"
    INNER = "inner"
    OUTER = "outer"
    FULL = "full"


class LagMethod(str, Enum):
    ALL_EXCEPT = "all_except"


class PartitionStrategy(str, Enum):
    YEAR = "year"
    NONE = "none"


# --- Ingestion Configuration Models ---
class BaseIngestionConfig(BaseModel):
    name: str
    format: IngestionFormat
    date_column: Dict[str, str] = Field(
        ..., description="A single-entry dict mapping date column name to its format."
    )
    id_column: Optional[str] = Field(
        None,
        description="The firm/entity identifier column. Defaults to date_column for time-series data.",
    )
    to_lowercase_cols: bool = False

    @field_validator("date_column")
    def _validate_date_column(cls, v: Dict[str, str]) -> Dict[str, str]:
        if len(v) != 1:
            raise ValueError("'date_column' must contain exactly one key-value pair.")
        return v

    @property
    def date_col_name(self) -> str:
        return list(self.date_column.keys())[0]

    @property
    def date_col_format(self) -> str:
        return list(self.date_column.values())[0]


class CSVConfig(BaseIngestionConfig):
    format: Literal[IngestionFormat.CSV]
    path: str


class GoogleSheetConfig(BaseIngestionConfig):
    format: Literal[IngestionFormat.GOOGLE_SHEET]
    url: str


class WRDSConfig(BaseIngestionConfig):
    format: Literal[IngestionFormat.WRDS]
    query: str


AnyIngestionConfig = Union[CSVConfig, GoogleSheetConfig, WRDSConfig]


# --- Wrangling Pipeline Operation Models ---
class BaseOperation(BaseModel):
    operation: str
    output_name: str


class ImputationConfig(BaseOperation):
    operation: Literal["monthly_imputation"]
    dataset: str
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    fallback_to_zero: bool = False


class ScaleRangeConfig(BaseModel):
    min: float
    max: float


class ScaleConfig(BaseOperation):
    operation: Literal["scale_to_range"]
    dataset: str
    cols_to_scale: List[str]
    range: ScaleRangeConfig


class MergeConfig(BaseOperation):
    operation: Literal["merge"]
    left_dataset: str
    right_dataset: str
    on: List[str]
    how: MergeStrategy


class LagColumnsConfig(BaseModel):
    method: LagMethod
    columns: List[str]


class LagConfig(BaseOperation):
    operation: Literal["lag"]
    dataset: str
    periods: int
    columns_to_lag: LagColumnsConfig
    drop_original_cols_after_lag: bool = False
    restore_names: bool = False
    drop_generated_nans: bool = False

    @model_validator(mode="after")
    def _validate_restore_names(self) -> "LagConfig":
        if self.restore_names and not self.drop_original_cols_after_lag:
            raise ValueError(
                "If 'restore_names' is true, 'drop_original_cols_after_lag' must also be true."
            )
        return self


class DummyConfig(BaseOperation):
    operation: Literal["dummy_generation"]
    dataset: str
    column_to_dummy: str
    drop_original_col: bool = False


class InteractionConfig(BaseOperation):
    operation: Literal["create_macro_interactions"]
    dataset: str
    macro_columns: List[str]
    firm_columns: List[str]
    drop_macro_columns: bool = False
    use_lazy_engine: bool = False


AnyWranglingOperation = Union[
    ImputationConfig,
    ScaleConfig,
    MergeConfig,
    LagConfig,
    DummyConfig,
    InteractionConfig,
]


# --- Export Configuration Models ---
class ExportConfig(BaseModel):
    dataset_name: str
    output_filename_base: str
    format: Literal["parquet"]
    partition_by: Optional[PartitionStrategy] = PartitionStrategy.NONE


# --- Top-Level Configuration Schema ---
class WRDSCredentials(BaseModel):
    user: Optional[str] = None
    password: Optional[str] = None


class DataConfig(BaseModel):
    ingestion: List[AnyIngestionConfig]
    # MODIFIED: Added the discriminator field
    wrangling_pipeline: List[
        Annotated[AnyWranglingOperation, Field(discriminator="operation")]
    ] = Field(default_factory=list)
    export: List[ExportConfig]
    wrds_credentials: Optional[WRDSCredentials] = None

    @model_validator(mode="after")
    def _validate_dataset_flow(self) -> "DataConfig":
        """Ensures that all datasets used in wrangling/export are defined."""
        known_datasets = {d.name for d in self.ingestion}

        for op in self.wrangling_pipeline:
            if isinstance(op, MergeConfig):
                if op.left_dataset not in known_datasets:
                    raise ValueError(
                        f"In merge operation, left_dataset '{op.left_dataset}' is not defined. "
                        f"Available datasets: {sorted(list(known_datasets))}"
                    )
                if op.right_dataset not in known_datasets:
                    raise ValueError(
                        f"In merge operation, right_dataset '{op.right_dataset}' is not defined. "
                        f"Available datasets: {sorted(list(known_datasets))}"
                    )
            else:
                if op.dataset not in known_datasets:
                    raise ValueError(
                        f"In operation '{op.operation}', input dataset '{op.dataset}' is not defined. "
                        f"Available datasets: {sorted(list(known_datasets))}"
                    )

            known_datasets.add(op.output_name)

        for exp in self.export:
            if exp.dataset_name not in known_datasets:
                raise ValueError(
                    f"In export config, dataset '{exp.dataset_name}' is not defined. "
                    f"Available datasets: {sorted(list(known_datasets))}"
                )
        return self


def load_config(config_path: Union[str, Path]) -> DataConfig:
    """
    Loads and validates the data configuration YAML file using Pydantic models.
    """
    config_path = Path(config_path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as file:
        try:
            raw_config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(
                f"Error parsing YAML file {config_path}: {exc}"
            ) from exc

    if not isinstance(raw_config, dict):
        raise ValueError(f"Configuration file '{config_path}' is empty or invalid.")

    try:
        return DataConfig.model_validate(raw_config)
    except ValidationError as e:
        raise ValueError(
            f"Configuration schema validation failed for {config_path}:\n{e}"
        ) from e
