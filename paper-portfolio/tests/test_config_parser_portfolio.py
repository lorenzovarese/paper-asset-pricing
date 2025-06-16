import pytest
import yaml
from paper_portfolio.config_parser import (  # type: ignore[import]
    load_config,
    PortfolioConfig,
)


def make_minimal_cfg_dict():
    """Returns a dictionary for a minimal valid configuration."""
    return {
        "input_data": {
            "prediction_model_names": ["m1"],
            "prediction_extraction_method": "precomputed_prediction_files",
            "precomputed_prediction_files": {
                "date_column": "date",
                "id_column": "permno",
            },
            "risk_free_dataset": {
                "file_name": "rf.csv",
                "date_column": "date",
                "return_column": "rf",
                "date_format": "%Y-%m-%d",
            },
        },
        "portfolio_strategies": [
            {
                "name": "strat1",
                "weighting_scheme": "equal",
                "long_quantiles": [0.9, 1.0],
                "short_quantiles": [0.0, 0.1],
            }
        ],
        "metrics": ["sharpe_ratio"],
    }


def test_load_valid_config(tmp_path):
    """Tests that a valid, minimal configuration loads correctly."""
    cfg = make_minimal_cfg_dict()
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))

    result = load_config(p)
    assert isinstance(result, PortfolioConfig)
    assert result.input_data.prediction_model_names == ["m1"]
    assert result.portfolio_strategies[0].name == "strat1"
    assert result.input_data.risk_free_dataset.file_name == "rf.csv"


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_empty_file(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(
        ValueError, match="empty or does not contain a valid YAML mapping"
    ):
        load_config(p)


def test_invalid_yaml(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("key: [invalid yaml")
    with pytest.raises(yaml.YAMLError):
        load_config(p)


def test_validation_error_missing_field(tmp_path):
    """Tests that Pydantic catches a missing required field."""
    cfg = make_minimal_cfg_dict()
    del cfg["metrics"]  # Remove a required field
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(cfg))
    with pytest.raises(ValueError, match="validation failed"):
        load_config(p)


def test_validation_error_bad_quantiles(tmp_path):
    """Tests that Pydantic catches incorrectly formatted quantiles."""
    cfg = make_minimal_cfg_dict()
    cfg["portfolio_strategies"][0]["long_quantiles"] = [0.8]  # Should have length 2
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(cfg))
    with pytest.raises(ValueError, match="validation failed"):
        load_config(p)


def test_value_weighting_dependency_check(tmp_path):
    """Tests that the validator catches value weighting without the required dataset."""
    cfg = make_minimal_cfg_dict()
    # Add a value-weighted strategy
    cfg["portfolio_strategies"].append(
        {
            "name": "strat2_value",
            "weighting_scheme": "value",
            "long_quantiles": [0.9, 1.0],
            "short_quantiles": [0.0, 0.1],
        }
    )

    # This should fail because company_value_weights is not defined
    p = tmp_path / "bad_value.yaml"
    p.write_text(yaml.dump(cfg))
    with pytest.raises(ValueError, match="company_value_weights"):
        load_config(p)

    # Now, add the required dataset definition
    cfg["input_data"]["company_value_weights"] = {
        "file_name": "weights.csv",
        "value_weight_col": "mktcap",
    }
    p_good = tmp_path / "good_value.yaml"
    p_good.write_text(yaml.dump(cfg))
    # This should now pass without raising an error
    assert isinstance(load_config(p_good), PortfolioConfig)
