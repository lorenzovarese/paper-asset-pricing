# paper-data/src/paper_data/config.py
from typing import List, Dict, Any, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from pathlib import (
    Path,
)  # Added for completeness, though not directly used in this file


# --- Connector Parameter Models ---
class BaseConnectorParams(BaseModel):
    class Config:
        extra = "forbid"


class HTTPConnectorParams(BaseConnectorParams):
    url: str
    timeout: int = 30


class LocalConnectorParams(BaseConnectorParams):
    path: str
    member_name: Optional[str] = None


class GoogleDriveConnectorParams(BaseConnectorParams):
    drive_url: str


class HuggingFaceConnectorParams(BaseConnectorParams):
    repo_id: str
    split: Optional[str] = None
    load_kwargs: Dict[str, Any] = Field(default_factory=dict)


class WRDSConnectorParams(BaseConnectorParams):
    query: str
    max_rows: Optional[int] = None


ConnectorParamsUnion = Union[
    HTTPConnectorParams,
    LocalConnectorParams,
    GoogleDriveConnectorParams,
    HuggingFaceConnectorParams,
    WRDSConnectorParams,
]


class ConnectorConfig(BaseModel):
    type: Literal["http", "local", "gdrive", "huggingface", "wrds"]
    params: ConnectorParamsUnion

    @field_validator("params", mode="before")
    @classmethod
    def _parse_params_based_on_type(
        cls, v_params: Any, info: ValidationInfo
    ) -> ConnectorParamsUnion:
        if not isinstance(v_params, dict):
            raise TypeError(
                f"Connector 'params' must be a dictionary, got {type(v_params).__name__}."
            )

        connector_type = info.data.get("type")

        if not connector_type:
            raise ValueError(
                "Internal error: Connector 'type' not available for 'params' validation."
            )

        try:
            if connector_type == "http":
                return HTTPConnectorParams.model_validate(v_params)
            elif connector_type == "local":
                return LocalConnectorParams.model_validate(v_params)
            elif connector_type == "gdrive":
                return GoogleDriveConnectorParams.model_validate(v_params)
            elif connector_type == "huggingface":
                return HuggingFaceConnectorParams.model_validate(v_params)
            elif connector_type == "wrds":
                return WRDSConnectorParams.model_validate(v_params)
            else:
                raise ValueError(
                    f"Unknown or unsupported connector type: {connector_type}"
                )
        except Exception as e:
            raise ValueError(
                f"Invalid params for connector type '{connector_type}': {e}"
            ) from e


# --- Schema Validation Model ---
class SchemaValidationConfig(BaseModel):
    type: Literal["firm", "macro"]

    class Config:
        extra = "forbid"


# --- Transformation Step Models ---
class BaseTransformationConfig(BaseModel):
    type: str

    class Config:
        extra = "forbid"


class NormalizeColumnsConfig(BaseTransformationConfig):
    type: Literal["normalize_columns"] = "normalize_columns"


class RenameDateColumnConfig(BaseTransformationConfig):
    type: Literal["rename_date_column"] = "rename_date_column"
    candidates: List[str] = Field(default_factory=lambda: ["date", "yyyymm", "time"])
    target: str = "date"


class ParseDateConfig(BaseTransformationConfig):
    type: Literal["parse_date"] = "parse_date"
    date_col: str = "date"
    date_format: Optional[str] = None
    monthly_option: Optional[Literal["start", "end"]] = None


class CleanNumericColumnConfig(BaseTransformationConfig):
    type: Literal["clean_numeric_column"] = "clean_numeric_column"
    col: str


class ImputeConstantConfig(BaseTransformationConfig):
    type: Literal["impute_constant"] = "impute_constant"
    cols: List[str]
    value: Any


class ImputeCrossSectionMedianConfig(BaseTransformationConfig):
    type: Literal["impute_cross_section_median"] = "impute_cross_section_median"
    cols: List[str]


class ImputeCrossSectionMeanConfig(BaseTransformationConfig):
    type: Literal["impute_cross_section_mean"] = "impute_cross_section_mean"
    cols: List[str]


class ImputeCrossSectionModeConfig(BaseTransformationConfig):
    type: Literal["impute_cross_section_mode"] = "impute_cross_section_mode"
    cols: List[str]


TransformationStepUnion = Union[
    NormalizeColumnsConfig,
    RenameDateColumnConfig,
    ParseDateConfig,
    CleanNumericColumnConfig,
    ImputeConstantConfig,
    ImputeCrossSectionMedianConfig,
    ImputeCrossSectionMeanConfig,
    ImputeCrossSectionModeConfig,
]


# --- Output Model ---
class OutputConfig(BaseModel):
    filename: str
    format: Literal["parquet", "csv", "feather", "json"] = "parquet"

    class Config:
        extra = "forbid"


# --- Source Model ---
class SourceConfig(BaseModel):
    name: str
    connector: ConnectorConfig
    schema_validation: Optional[SchemaValidationConfig] = None
    objective_for_cleaner: Literal["firm", "macro"] = "firm"
    date_col_for_firm_cleaner: str = "date"
    id_col_for_firm_cleaner: str = "company_id"
    transformations: List[TransformationStepUnion] = Field(default_factory=list)
    output: OutputConfig

    class Config:
        extra = "forbid"

    @field_validator("transformations", mode="before")
    @classmethod
    def _parse_transformations(
        cls, v_transformations: Any
    ) -> List[TransformationStepUnion]:
        if not isinstance(v_transformations, list):
            raise TypeError(
                f"'transformations' must be a list, got {type(v_transformations).__name__}."
            )

        processed_transformations: List[TransformationStepUnion] = []

        model_map: Dict[str, type[BaseTransformationConfig]] = {
            "normalize_columns": NormalizeColumnsConfig,
            "rename_date_column": RenameDateColumnConfig,
            "parse_date": ParseDateConfig,
            "clean_numeric_column": CleanNumericColumnConfig,
            "impute_constant": ImputeConstantConfig,
            "impute_cross_section_median": ImputeCrossSectionMedianConfig,
            "impute_cross_section_mean": ImputeCrossSectionMeanConfig,
            "impute_cross_section_mode": ImputeCrossSectionModeConfig,
        }

        for i, trans_config_raw in enumerate(v_transformations):
            if not isinstance(trans_config_raw, dict):
                raise TypeError(
                    f"Transformation at index {i} must be a dictionary, got {type(trans_config_raw).__name__}."
                )

            trans_type_str = trans_config_raw.get("type")
            if not trans_type_str:
                raise ValueError(
                    f"Transformation at index {i} is missing 'type' field."
                )
            if not isinstance(trans_type_str, str):
                raise TypeError(
                    f"Transformation 'type' at index {i} must be a string, got {type(trans_type_str).__name__}."
                )

            model_to_use = model_map.get(trans_type_str)
            if not model_to_use:
                raise ValueError(
                    f"Transformation at index {i} has an unsupported type: '{trans_type_str}'. Supported types are: {list(model_map.keys())}"
                )

            try:
                validated_trans = model_to_use.model_validate(trans_config_raw)
                processed_transformations.append(validated_trans)
            except Exception as e:
                raise ValueError(
                    f"Error validating transformation at index {i} (type: {trans_type_str}): {e}"
                ) from e

        return processed_transformations


# --- Global Settings Model ---
class GlobalSettings(BaseModel):
    default_date_format: Optional[str] = None
    output_dir: str = "data/processed"
    default_date_col_for_cleaners: str = "date"
    default_id_col_for_firm_cleaner: str = "company_id"

    class Config:
        extra = "forbid"


# --- Main Data Configuration Model ---
class PaperDataConfig(BaseModel):
    project_name: Optional[str] = None
    global_settings: GlobalSettings = Field(default_factory=GlobalSettings)
    sources: List[SourceConfig]

    class Config:
        extra = "forbid"
