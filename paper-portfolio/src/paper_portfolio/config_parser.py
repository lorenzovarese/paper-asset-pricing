from pathlib import Path
import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import List, Literal, Union, Optional

# --- Individual Data Source Configurations ---


class PredictionFilesConfig(BaseModel):
    """Configuration for loading pre-computed prediction files."""

    date_column: str = "date"
    id_column: str = "permno"


class CompanyValuesConfig(BaseModel):
    """Configuration for the value-weighting data."""

    file_name: str
    date_column: str = "date"
    id_column: str = "permno"
    date_format: str = "%Y-%m-%d"
    value_weight_col: str = "market_value"


class RiskFreeDatasetConfig(BaseModel):
    """Configuration for the risk-free rate data."""

    file_name: str
    date_column: str = "date"
    return_column: str = "rf"
    date_format: str = "%Y-%m-%d"


class MarketBenchmarkConfig(BaseModel):
    """Configuration for a market index benchmark."""

    name: str = Field(
        ..., description="Display name for the benchmark (e.g., 'S&P 500')."
    )
    file_name: str
    date_column: str = "date"
    return_column: str = "ret"
    date_format: str = "%Y-%m-%d"


# --- Main Input Data Configuration ---


class InputDataConfig(BaseModel):
    """Defines all data inputs for the portfolio analysis."""

    prediction_model_names: List[str]
    prediction_extraction_method: Literal["precomputed_prediction_files"]
    precomputed_prediction_files: PredictionFilesConfig
    company_value_weights: Optional[CompanyValuesConfig] = None
    risk_free_dataset: RiskFreeDatasetConfig
    market_benchmark: Optional[MarketBenchmarkConfig] = None


# --- Strategy and Top-Level Configuration ---


class PortfolioStrategyConfig(BaseModel):
    """Defines a single long-short portfolio strategy."""

    name: str
    weighting_scheme: Literal["equal", "value"]
    long_quantiles: List[float] = Field(..., min_length=2, max_length=2)
    short_quantiles: List[float] = Field(..., min_length=2, max_length=2)


class PortfolioConfig(BaseModel):
    """The root configuration model for the entire portfolio pipeline."""

    input_data: InputDataConfig
    portfolio_strategies: List[PortfolioStrategyConfig]
    metrics: List[Literal["sharpe_ratio", "expected_shortfall", "cumulative_return"]]
    cross_sectional_analysis: Optional[bool] = False

    @model_validator(mode="after")
    def check_value_weighting_dependencies(self) -> "PortfolioConfig":
        """
        Ensures that if any strategy uses value weighting, the corresponding
        dataset is defined in the input_data section.
        """
        uses_value_weighting = any(
            s.weighting_scheme == "value" for s in self.portfolio_strategies
        )
        if uses_value_weighting and self.input_data.company_value_weights is None:
            raise ValueError(
                "A strategy uses 'weighting_scheme: value', but 'company_value_weights' "
                "is not defined in the 'input_data' section of the configuration."
            )
        return self


def load_config(config_path: Union[str, Path]) -> PortfolioConfig:
    """Loads and validates the portfolio configuration YAML file."""
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
        raise ValueError(
            f"Configuration file '{config_path}' is empty or does not contain a valid YAML mapping (dictionary)."
        )

    try:
        return PortfolioConfig(**raw_config)
    except ValidationError as e:
        raise ValueError(
            f"Portfolio configuration validation failed for {config_path}:\n{e}"
        ) from e
