from pathlib import Path
import yaml
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal, Union, Optional


class PortfolioStrategyConfig(BaseModel):
    name: str
    weighting_scheme: Literal["equal", "value"]
    long_quantiles: List[float] = Field(..., min_length=2, max_length=2)
    short_quantiles: List[float] = Field(..., min_length=2, max_length=2)


class InputDataConfig(BaseModel):
    prediction_model_names: List[str]
    processed_dataset_name: str
    date_column: str = "date"
    id_column: str = "permno"
    risk_free_rate_col: str = "rf"
    value_weight_col: str = "bm"


class MarketBenchmarkConfig(BaseModel):
    """Configuration for a market index benchmark."""

    name: str = Field(
        ..., description="Display name for the benchmark (e.g., 'S&P 500')."
    )
    file_name: str = Field(
        ...,
        description="The name of the CSV file in the 'portfolios/indexes/' directory.",
    )
    date_column: str = Field(
        "date", description="The name of the date column in the CSV file."
    )
    return_column: str = Field(
        "ret", description="The name of the return column in the CSV file."
    )
    date_format: str = Field(
        "%Y-%m-%d", description="The date format string for parsing the date column."
    )


class PortfolioConfig(BaseModel):
    input_data: InputDataConfig
    strategies: List[PortfolioStrategyConfig]
    metrics: List[Literal["sharpe_ratio", "expected_shortfall", "cumulative_return"]]
    market_benchmark: Optional[MarketBenchmarkConfig] = None


def load_config(config_path: Union[str, Path]) -> PortfolioConfig:
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
