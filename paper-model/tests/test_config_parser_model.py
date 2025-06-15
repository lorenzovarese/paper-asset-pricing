import pytest
import yaml
from paper_model.config_parser import (  # type: ignore
    load_config,
    ModelsConfig,
    OLSConfig,
)


def make_minimal_config(tmp_path, override: dict = None):  # type: ignore
    cfg = {
        "input_data": {
            "dataset_name": "macro",
            "splitted": "none",
            "date_column": "date",
            "id_column": "id",
        },
        "evaluation": {
            "implementation": "rolling window",
            "train_month": 12,
            "validation_month": 1,
            "testing_month": 3,
            "step_month": 1,
            "metrics": ["mse", "r2_oos"],
        },
        "models": [
            {
                "name": "model1",
                "type": "ols",
                "target_column": "y",
                "features": ["x1", "x2"],
                "weighting_scheme": "none",
            }
        ],
    }
    if override:
        cfg.update(override)
    path = tmp_path / "models-config.yaml"
    path.write_text(yaml.dump(cfg))
    return path


def test_load_valid_config(tmp_path):
    cfg_path = make_minimal_config(tmp_path)
    cfg = load_config(cfg_path)
    assert isinstance(cfg, ModelsConfig)
    # input_data parsed correctly
    assert cfg.input_data.dataset_name == "macro"
    # our single model is an OLSConfig
    model = cfg.models[0]
    assert isinstance(model, OLSConfig)
    assert model.type == "ols"
    # no tuning required by default
    assert model.requires_tuning is False


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "no-such.yaml")


def test_unsupported_model_type(tmp_path):
    # Inject a model with invalid type
    bad = make_minimal_config(
        tmp_path,
        override={
            "models": [
                {
                    "name": "bad",
                    "type": "foo",
                    "target_column": "y",
                    "features": ["x"],
                    "weighting_scheme": "none",
                }
            ]
        },
    )
    with pytest.raises(ValueError):
        load_config(bad)


def test_ols_requires_market_cap(tmp_path):
    # weighting_scheme=mkt_cap but no market_cap_column → should fail
    override = {
        "models": [
            {
                "name": "mcap",
                "type": "ols",
                "target_column": "y",
                "features": ["x"],
                "weighting_scheme": "mkt_cap",
            }
        ]
    }
    bad = make_minimal_config(tmp_path, override=override)
    with pytest.raises(ValueError):
        load_config(bad)
