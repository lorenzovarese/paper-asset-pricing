from pathlib import Path
import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from typing import List, Literal, Optional, Union, Dict, Any
from enum import Enum

# --- Pydantic Models for Configuration Schema ---


class SplittedType(str, Enum):
    YEAR = "year"
    NONE = "none"


class EvaluationImplementation(str, Enum):
    ROLLING_WINDOW = "rolling window"


class ObjectiveFunction(str, Enum):
    L2 = "l2"
    HUBER = "huber"


class WeightingScheme(str, Enum):
    NONE = "none"
    INV_N_STOCKS = "inv_n_stocks"
    MKT_CAP = "mkt_cap"


class InputDataConfig(BaseModel):
    dataset_name: str
    splitted: SplittedType
    date_column: str
    id_column: str
    risk_free_rate_col: Optional[str] = None


class EvaluationConfig(BaseModel):
    implementation: EvaluationImplementation
    train_month: int = Field(..., gt=0)
    validation_month: int = Field(0, ge=0)
    testing_month: int = Field(..., gt=0)
    step_month: int = Field(..., gt=0)
    metrics: List[str] = Field(default_factory=list)


# --- Model-specific Configurations ---


class BaseModelConfig(BaseModel):
    name: str
    type: str
    target_column: str
    feature_columns: List[str]
    objective_function: ObjectiveFunction = ObjectiveFunction.L2
    save_model_checkpoints: bool = False
    save_prediction_results: bool = False
    random_state: Optional[int] = None


class OLSConfig(BaseModelConfig):
    type: Literal["ols"]
    weighting_scheme: WeightingScheme = Field(
        WeightingScheme.NONE,
        description="Sample weighting scheme to use during training.",
    )
    market_cap_column: Optional[str] = Field(
        None, description="Required column name for market cap weighting."
    )
    huber_epsilon_quantile: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Quantile for adaptively setting Huber loss epsilon (ξ).",
    )

    @model_validator(mode="after")
    def _validate_config(self) -> "OLSConfig":
        if (
            self.weighting_scheme == WeightingScheme.MKT_CAP
            and self.market_cap_column is None
        ):
            raise ValueError(
                "'market_cap_column' must be provided when 'weighting_scheme' is 'mkt_cap'."
            )
        if (
            self.huber_epsilon_quantile is not None
            and self.objective_function != "huber"
        ):
            raise ValueError(
                "'huber_epsilon_quantile' can only be set when 'objective_function' is 'huber'."
            )
        return self


class ElasticNetConfig(BaseModelConfig):
    type: Literal["enet"]
    alpha: Union[float, List[float]] = Field(
        1.0, description="Regularization strength or list of strengths for tuning"
    )
    l1_ratio: Union[float, List[float]] = Field(
        0.5, description="ElasticNet mixing parameter or list of parameters for tuning"
    )

    @property
    def requires_tuning(self) -> bool:
        return isinstance(self.alpha, list) or isinstance(self.l1_ratio, list)


class PCRConfig(BaseModelConfig):
    type: Literal["pcr"]
    n_components: int = Field(
        ..., gt=0, description="Number of principal components to keep"
    )


class PLSConfig(BaseModelConfig):
    type: Literal["pls"]
    n_components: int = Field(
        ..., gt=0, description="Number of partial least squares components"
    )


# --- Main Configuration Schema ---

AnyModel = Union[OLSConfig, ElasticNetConfig, PCRConfig, PLSConfig]


class ModelsConfig(BaseModel):
    input_data: InputDataConfig
    evaluation: EvaluationConfig
    models: List[AnyModel]

    @field_validator("models", mode="before")
    @classmethod
    def dispatch_model_configs(cls, v: List[Dict[str, Any]]) -> List[AnyModel]:
        dispatched: List[AnyModel] = []
        for model_config in v:
            model_type = model_config.get("type")
            if model_type == "ols":
                dispatched.append(OLSConfig(**model_config))
            elif model_type == "enet":
                dispatched.append(ElasticNetConfig(**model_config))
            elif model_type == "pcr":
                dispatched.append(PCRConfig(**model_config))
            elif model_type == "pls":
                dispatched.append(PLSConfig(**model_config))
            else:
                raise ValueError(f"Unsupported model type: '{model_type}'")
        return dispatched

    @model_validator(mode="after")
    def check_validation_set_for_tuning(self) -> "ModelsConfig":
        """Ensure validation_month is set if any model requires tuning."""
        for model in self.models:
            if isinstance(model, ElasticNetConfig) and model.requires_tuning:
                if self.evaluation.validation_month <= 0:
                    raise ValueError(
                        f"Model '{model.name}' requires hyperparameter tuning, "
                        "but 'evaluation.validation_month' is not set to a value greater than 0."
                    )
        return self


def load_config(config_path: Union[str, Path]) -> ModelsConfig:
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
        return ModelsConfig(**raw_config)
    except ValidationError as e:
        raise ValueError(
            f"Configuration schema validation failed for {config_path}:\n{e}"
        ) from e
