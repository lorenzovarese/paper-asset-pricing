from pathlib import Path
import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from typing import List, Literal, Optional, Tuple, Union, Dict, Any
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


class FeatureSelectionConfig(BaseModel):
    """Defines the 'all_except' feature selection method."""

    method: Literal["all_except"]
    columns: List[str] = Field(
        ..., description="List of columns to exclude from features."
    )


# --- Model-specific Configurations ---


class BaseModelConfig(BaseModel):
    name: str
    type: str
    target_column: str
    features: Union[List[str], FeatureSelectionConfig] = Field(
        ...,
        description="Defines the feature set for the model, either by explicit list or by exclusion.",
    )
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

    @property
    def requires_tuning(self) -> bool:
        return False


class ElasticNetConfig(BaseModelConfig):
    type: Literal["enet"]
    alpha: Union[float, List[float]] = Field(
        1.0, description="Regularization strength or list of strengths for tuning"
    )
    l1_ratio: Union[float, List[float]] = Field(
        0.5, description="ElasticNet mixing parameter or list of parameters for tuning"
    )
    max_iter: int = Field(
        1000, description="Maximum number of iterations for the solver."
    )

    @property
    def requires_tuning(self) -> bool:
        return isinstance(self.alpha, list) or isinstance(self.l1_ratio, list)


class PCRConfig(BaseModelConfig):
    type: Literal["pcr"]
    n_components: Union[int, List[int]] = Field(
        ..., description="Number of principal components or list for tuning"
    )

    @property
    def requires_tuning(self) -> bool:
        return isinstance(self.n_components, list)


class PLSConfig(BaseModelConfig):
    type: Literal["pls"]
    n_components: Union[int, List[int]] = Field(
        ..., description="Number of partial least squares components or list for tuning"
    )

    @property
    def requires_tuning(self) -> bool:
        return isinstance(self.n_components, list)


class GLMConfig(BaseModelConfig):
    type: Literal["glm"]
    n_knots: int = Field(
        3,
        description="Number of knots for the spline transformer. Default is 3 as per Gu et al.",
    )
    alpha: Union[float, List[float]] = Field(
        ..., description="Group lasso regularization strength (λ) or list for tuning."
    )

    @model_validator(mode="after")
    def _validate_config(self) -> "GLMConfig":
        if self.objective_function == "huber":
            raise NotImplementedError(
                "The 'glm' model with Group Lasso does not support Huber loss in this implementation. Please use 'l2'."
            )
        return self

    @property
    def requires_tuning(self) -> bool:
        return isinstance(self.alpha, list)


class RandomForestConfig(BaseModelConfig):
    type: Literal["rf"]
    n_estimators: int = Field(300, description="Number of trees in the forest (B).")
    max_depth: Union[int, List[int]] = Field(
        ..., description="Maximum depth of the tree (L) or list for tuning."
    )
    max_features: Union[int, float, str, List[Union[int, float, str]]] = Field(
        "sqrt",
        description="Number of features to consider for each split (e.g., 'sqrt', 0.5, 50) or list for tuning.",
    )

    @property
    def requires_tuning(self) -> bool:
        return isinstance(self.max_depth, list) or isinstance(self.max_features, list)


class GBRTConfig(BaseModelConfig):
    type: Literal["gbrt"]
    n_estimators: Union[int, List[int]] = Field(
        ..., description="Number of boosting stages to perform (B) or list for tuning."
    )
    max_depth: Union[int, List[int]] = Field(
        ...,
        description="Maximum depth of the individual regression estimators (L) or list for tuning.",
    )
    learning_rate: Union[float, List[float]] = Field(
        ...,
        description="Learning rate shrinks the contribution of each tree (ν) or list for tuning.",
    )
    use_hist_implementation: bool = Field(
        False,
        description="If true, use the faster HistGradientBoostingRegressor implementation.",
    )

    @property
    def requires_tuning(self) -> bool:
        return (
            isinstance(self.n_estimators, list)
            or isinstance(self.max_depth, list)
            or isinstance(self.learning_rate, list)
        )


class NNConfig(BaseModelConfig):
    type: Literal["nn"]
    hidden_layer_sizes: Tuple[int, ...] = Field(
        ..., description="A tuple defining the number of neurons in each hidden layer."
    )
    alpha: Union[float, List[float]] = Field(
        ..., description="L1 penalty (λ) or list for tuning."
    )
    learning_rate: Union[float, List[float]] = Field(
        ..., description="Learning rate for the Adam optimizer or list for tuning."
    )
    batch_size: int = Field(10000, description="Size of mini-batches for SGD.")
    epochs: int = Field(100, description="Maximum number of training epochs.")
    patience: int = Field(
        5, description="Epochs to wait for improvement before early stopping."
    )
    n_ensembles: int = Field(
        10, description="Number of models to train in the ensemble."
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        "auto",
        description="Device to use for training, e.g., 'auto', 'cpu', 'cuda', 'mps'.",
    )
    num_workers: int = Field(
        0,
        description="Number of subprocesses to use for data loading. 0 means data is loaded in the main process.",
    )

    @property
    def requires_tuning(self) -> bool:
        # Tuning is required if alpha or learning_rate are lists
        return isinstance(self.alpha, list) or isinstance(self.learning_rate, list)


# --- Main Configuration Schema ---

AnyModel = Union[
    OLSConfig,
    ElasticNetConfig,
    PCRConfig,
    PLSConfig,
    GLMConfig,
    RandomForestConfig,
    GBRTConfig,
    NNConfig,
]


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
            elif model_type == "glm":
                dispatched.append(GLMConfig(**model_config))
            elif model_type == "rf":
                dispatched.append(RandomForestConfig(**model_config))
            elif model_type == "gbrt":
                dispatched.append(GBRTConfig(**model_config))
            elif model_type == "nn":
                dispatched.append(NNConfig(**model_config))
            else:
                raise ValueError(f"Unsupported model type: '{model_type}'")
        return dispatched

    @model_validator(mode="after")
    def check_validation_set_for_tuning(self) -> "ModelsConfig":
        """Ensure validation_month is set if any model requires tuning."""
        for model in self.models:
            if model.requires_tuning or isinstance(model, NNConfig):
                if self.evaluation.validation_month <= 0:
                    raise ValueError(
                        f"Model '{model.name}' requires a validation set (for tuning or early stopping), "
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

    # Check if the loaded config is a dictionary. This handles empty files.
    if not isinstance(raw_config, dict):
        raise ValueError(
            f"Configuration file '{config_path}' is empty or does not contain a valid YAML mapping (dictionary)."
        )

    try:
        return ModelsConfig(**raw_config)
    except ValidationError as e:
        raise ValueError(
            f"Configuration schema validation failed for {config_path}:\n{e}"
        ) from e
