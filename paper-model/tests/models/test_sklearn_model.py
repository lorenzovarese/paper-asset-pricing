import pytest
import polars as pl
import numpy as np
from unittest.mock import patch, MagicMock

from paper_model.models.sklearn_model import SklearnModel  # type: ignore
from sklearn.linear_model import LinearRegression, ElasticNet, HuberRegressor  # type: ignore
from sklearn.ensemble import (  # type: ignore
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.cross_decomposition import PLSRegression  # type: ignore

# --- Fixtures ---


@pytest.fixture
def sample_data():
    """Provides sample training, validation, and prediction data."""
    train_df = pl.DataFrame(
        {
            "date": ["2020-01-31"] * 10,
            "permno": range(10),
            "feat1": np.random.rand(10),
            "feat2": np.random.rand(10),
            "mkt_cap": np.random.rand(10) * 1000,
            "ret": np.random.rand(10),
        }
    )
    val_df = pl.DataFrame(
        {
            "date": ["2020-02-29"] * 5,
            "permno": range(5),
            "feat1": np.random.rand(5),
            "feat2": np.random.rand(5),
            "mkt_cap": np.random.rand(5) * 1000,
            "ret": np.random.rand(5),
        }
    )
    return train_df, val_df


@pytest.fixture
def base_config():
    """Provides a base configuration dictionary."""
    return {
        "name": "test_model",
        "target_column": "ret",
        "feature_columns": ["feat1", "feat2"],
        "date_column": "date",
        "id_column": "permno",
    }


# --- Tests for SklearnModel ---


@pytest.mark.parametrize(
    "model_type, expected_class",
    [
        ("ols", LinearRegression),
        ("enet", ElasticNet),
        ("rf", RandomForestRegressor),
        ("gbrt", HistGradientBoostingRegressor),
        ("gbrt_no_hist", GradientBoostingRegressor),
        ("pcr", HuberRegressor),  # PCR with huber objective
        ("pls", PLSRegression),
    ],
)
def test_model_creation_no_tuning(model_type, expected_class, base_config, sample_data):
    """Tests that the correct sklearn model is created for non-tuning scenarios."""
    train_df, _ = sample_data
    config = base_config.copy()

    # Special handling for gbrt without hist
    use_hist = True
    if model_type == "gbrt_no_hist":
        model_type = "gbrt"
        use_hist = False

    config.update(
        {
            "type": model_type,
            "objective_function": "huber" if model_type == "pcr" else "l2",
            "alpha": 0.1,
            "l1_ratio": 0.5,
            "n_components": 2,
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "use_hist_implementation": use_hist,
            "max_features": "sqrt",
        }
    )

    model = SklearnModel(name="test", config=config)
    model.train(train_df)

    assert model.model is not None

    if model_type in ["ols", "enet", "pls"]:
        final_estimator = model.model.named_steps["model"]
    elif model_type == "pcr":
        final_estimator = model.model.named_steps["pcr_pipeline"].named_steps[
            "regressor"
        ]
    else:
        final_estimator = model.model

    assert isinstance(final_estimator, expected_class)


@pytest.mark.parametrize(
    "model_type, tuning_params",
    [
        ("enet", {"alpha": [0.1, 1.0]}),
        ("pcr", {"n_components": [1, 2]}),
        ("pls", {"n_components": [1, 2]}),
        ("glm", {"alpha": [0.1, 0.2], "n_knots": 2}),
        ("rf", {"max_depth": [2, 3]}),
        ("gbrt", {"learning_rate": [0.05, 0.1]}),
    ],
)
@patch("paper_model.models.sklearn_model.GridSearchCV")
def test_model_tuning_scenarios(
    mock_grid_search_class, model_type, tuning_params, base_config, sample_data
):
    """Tests that GridSearchCV is used for all tuning-enabled models."""
    train_df, val_df = sample_data

    mock_instance = MagicMock()
    mock_instance.best_params_ = {"mock_param": 1}
    mock_instance.best_estimator_ = MagicMock()
    mock_grid_search_class.return_value = mock_instance

    config = base_config.copy()
    config.update({"type": model_type, **tuning_params})

    # MODIFIED: Add all required default params for GBRT
    config.setdefault("n_estimators", 10)
    config.setdefault("max_depth", 3)
    config.setdefault("learning_rate", 0.1)
    config.setdefault("max_features", "sqrt")

    model = SklearnModel(name=f"test_{model_type}_tuning", config=config)
    model.train(train_df, val_df)

    mock_grid_search_class.assert_called_once()
    mock_instance.fit.assert_called_once()
    assert model.model is not None


def test_predict_logic(base_config, sample_data):
    """Tests the predict method on a trained model."""
    train_df, _ = sample_data
    config = base_config.copy()
    config["type"] = "ols"

    model = SklearnModel(name="test_ols", config=config)
    model.train(train_df)

    pred_df = pl.DataFrame(
        {
            "date": ["2020-03-31"] * 3,
            "permno": range(3),
            "feat1": [0.1, 0.2, 0.3],
            "feat2": [0.4, 0.5, 0.6],
        }
    )

    predictions = model.predict(pred_df)

    assert isinstance(predictions, pl.Series)
    assert len(predictions) == len(pred_df)
    assert not predictions.is_null().any()


def test_predict_with_missing_features_in_data(base_config, sample_data):
    """Tests that predict returns nulls for rows with missing features."""
    train_df, _ = sample_data
    config = base_config.copy()
    config["type"] = "ols"

    model = SklearnModel(name="test_ols", config=config)
    model.train(train_df)

    pred_df_with_nulls = pl.DataFrame(
        {
            "date": ["2020-03-31"] * 2,
            "permno": [1, 2],
            "feat1": [0.1, None],
            "feat2": [0.4, 0.5],
        }
    )

    predictions = model.predict(pred_df_with_nulls)

    assert len(predictions) == 2
    assert predictions[0] is not None
    assert predictions[1] is None


def test_train_handles_empty_clean_data(base_config, caplog):
    """Tests that training is skipped if no clean data is available."""
    config = base_config.copy()
    config["type"] = "ols"

    train_df_with_nulls = pl.DataFrame(
        {"feat1": [1.0, None], "feat2": [None, 2.0], "ret": [0.1, 0.2]}
    )

    model = SklearnModel(name="test_empty", config=config)
    model.train(train_df_with_nulls)

    assert model.model is None
    assert "Skipped training for this window" in caplog.text


def test_ols_with_market_cap_weighting(base_config, sample_data):
    """Tests the OLS market cap weighting scheme."""
    train_df, _ = sample_data
    config = base_config.copy()
    config.update(
        {"type": "ols", "weighting_scheme": "mkt_cap", "market_cap_column": "mkt_cap"}
    )

    model = SklearnModel(name="test_ols_weighted", config=config)

    with patch.object(Pipeline, "fit") as mock_fit:
        model.train(train_df)
        mock_fit.assert_called_once()
        _, kwargs = mock_fit.call_args
        assert "model__sample_weight" in kwargs
        assert isinstance(kwargs["model__sample_weight"], np.ndarray)
