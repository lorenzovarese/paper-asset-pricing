from pathlib import Path
import yaml
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal, Union


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


class PortfolioConfig(BaseModel):
    input_data: InputDataConfig
    strategies: List[PortfolioStrategyConfig]
    metrics: List[Literal["sharpe_ratio", "expected_shortfall", "cumulative_return"]]


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

    try:
        return PortfolioConfig(**raw_config)
    except ValidationError as e:
        raise ValueError(
            f"Portfolio configuration validation failed for {config_path}:\n{e}"
        ) from e
