from pathlib import Path
import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import (
    List,
    Literal,
    Optional,
    Union,
    Dict,
    Any,
)
from enum import Enum

# --- Pydantic Models for Configuration Schema ---


class SplittedType(str, Enum):
    """Defines valid options for data splitting."""

    YEAR = "year"
    NONE = "none"


class EvaluationImplementation(str, Enum):
    """Defines valid options for evaluation implementation."""

    ROLLING_WINDOW = "rolling window"


class InputDataConfig(BaseModel):
    """Schema for the 'input_data' section of the models config."""

    dataset_name: str
    splitted: SplittedType
    date_column: str
    id_column: str
    risk_free_rate_col: str


class EvaluationConfig(BaseModel):
    """Schema for the 'evaluation' section of the models config."""

    implementation: EvaluationImplementation
    train_month: int = Field(
        ..., gt=0, description="Number of months for the training window."
    )
    validation_month: int = Field(
        0,
        ge=0,
        description="Number of months for the validation window (currently unused).",
    )
    testing_month: int = Field(
        ..., gt=0, description="Number of months for the testing window."
    )
    step_month: int = Field(
        ..., gt=0, description="Number of months to step forward for the next window."
    )
    metrics: List[str] = Field(
        default_factory=list, description="List of metrics to compute."
    )


# Base model config for common fields
class BaseModelConfig(BaseModel):
    """Base schema for any model configuration."""

    name: str
    type: str  # This will be further validated by Literal types in derived classes
    save_model_checkpoints: bool = False
    save_prediction_results: bool = False


# Specific model configurations
class FamaFrench3FactorModelConfig(BaseModelConfig):
    """Schema for the 'fama_french_3_factor' model."""

    type: Literal["fama_french_3_factor"]  # Ensures 'type' is exactly this string
    target_return_col: str
    factor_columns: List[str]


class SimpleLinearRegressionModelConfig(BaseModelConfig):
    """Schema for the 'linear_regression' model."""

    type: Literal["linear_regression"]  # Ensures 'type' is exactly this string
    target_column: str
    feature_columns: List[str]
    random_state: Optional[int] = None


class ModelsConfig(BaseModel):
    """Overall schema for the models configuration file."""

    input_data: InputDataConfig
    evaluation: EvaluationConfig
    models: List[Union[FamaFrench3FactorModelConfig, SimpleLinearRegressionModelConfig]]

    @field_validator("models", mode="before")
    @classmethod
    def check_model_type_and_dispatch(
        cls, v: List[Dict[str, Any]]
    ) -> List[BaseModelConfig]:
        """
        Dynamically dispatches model configurations based on their 'type' field.
        This allows Pydantic to validate against the correct specific model schema.
        """
        # Explicitly type the list to hold BaseModelConfig instances
        dispatched_models: List[BaseModelConfig] = []
        for model_dict in v:
            if not isinstance(model_dict, dict):
                raise ValueError(
                    f"Model configuration must be a dictionary, got: {type(model_dict)}"
                )

            model_type = model_dict.get("type")
            if model_type == "fama_french_3_factor":
                dispatched_models.append(FamaFrench3FactorModelConfig(**model_dict))
            elif model_type == "linear_regression":
                dispatched_models.append(
                    SimpleLinearRegressionModelConfig(**model_dict)
                )
            else:
                raise ValueError(f"Unknown or unsupported model type: {model_type}")
        return dispatched_models


# --- Configuration Loading Function ---


def load_config(config_path: str | Path) -> ModelsConfig:
    """
    Loads and parses a YAML configuration file, validating it against the schema.

    Args:
        config_path: The path to the YAML configuration file.

    Returns:
        A ModelsConfig object representing the parsed and validated configuration.

    Raises:
        FileNotFoundError: If the config_path does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
        ValueError: If the configuration does not conform to the defined schema.
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

    try:
        # Validate the raw dictionary against the Pydantic schema
        config = ModelsConfig(**raw_config)
    except ValidationError as e:
        raise ValueError(
            f"Configuration schema validation failed for {config_path}:\n{e}"
        ) from e
    return config
