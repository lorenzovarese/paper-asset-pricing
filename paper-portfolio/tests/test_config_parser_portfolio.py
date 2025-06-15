import pytest
import yaml
from paper_portfolio.config_parser import (  # type: ignore
    load_config,
    PortfolioConfig,
)


def make_minimal_cfg(tmp_path):
    cfg = {
        "input_data": {
            "prediction_model_names": ["m1"],
            "processed_dataset_name": "pfx",
        },
        "strategies": [
            {
                "name": "strat1",
                "weighting_scheme": "equal",
                "long_quantiles": [0.0, 0.5],
                "short_quantiles": [0.5, 1.0],
            }
        ],
        "metrics": ["sharpe_ratio"],
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))
    return p


def test_load_valid_config(tmp_path):
    p = make_minimal_cfg(tmp_path)
    result = load_config(p)
    assert isinstance(result, PortfolioConfig)
    assert result.input_data.processed_dataset_name == "pfx"
    assert result.strategies[0].name == "strat1"


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_empty_file(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(ValueError):
        load_config(p)


def test_invalid_yaml(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("::: not yaml :::")
    with pytest.raises(yaml.YAMLError):
        load_config(p)


def test_validation_error_quantiles(tmp_path):
    # quantiles wrong length
    cfg = {
        "input_data": {
            "prediction_model_names": ["m1"],
            "processed_dataset_name": "pfx",
        },
        "strategies": [
            {
                "name": "bad",
                "weighting_scheme": "equal",
                "long_quantiles": [0.2],  # should be length 2
                "short_quantiles": [0.8, 1.0],
            }
        ],
        "metrics": ["sharpe_ratio"],
    }
    p = tmp_path / "badq.yaml"
    p.write_text(yaml.dump(cfg))
    with pytest.raises(ValueError):
        load_config(p)
