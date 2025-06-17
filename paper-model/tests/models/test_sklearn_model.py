import pytest
import polars as pl
import numpy as np
from unittest.mock import patch

from paper_model.models.sklearn_model import SklearnModel  # type: ignore
from sklearn.linear_model import LinearRegression, ElasticNet  # type: ignore
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore

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
        (
            "gbrt",
            HistGradientBoostingRegressor,
        ),  # Testing the faster hist implementation
    ],
)
def test_model_creation_no_tuning(model_type, expected_class, base_config, sample_data):
    """Tests that the correct sklearn model is created for non-tuning scenarios."""
    train_df, _ = sample_data
    config = base_config.copy()
    config.update(
        {
            "type": model_type,
            # Add required params for non-tuning cases
            "alpha": 0.1,
            "l1_ratio": 0.5,
            "n_components": 2,
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "use_hist_implementation": True,
            "max_features": "sqrt",
        }
    )

    model = SklearnModel(name="test", config=config)
    model.train(train_df)

    assert model.model is not None

    # For models wrapped in a pipeline, find the actual model step
    if model_type in ["ols", "enet", "pls"]:
        final_estimator = model.model.named_steps["model"]
    elif model_type == "pcr":
        final_estimator = model.model.named_steps["pcr_pipeline"].named_steps[
            "regressor"
        ]
    else:  # rf, gbrt, glm are not wrapped in the same way
        final_estimator = model.model

    assert isinstance(final_estimator, expected_class)


@patch("paper_model.models.sklearn_model.GridSearchCV")
def test_model_creation_with_tuning(mock_grid_search_fit, base_config, sample_data):
    """Tests that GridSearchCV is used when tuning is required."""
    train_df, val_df = sample_data

    config = base_config.copy()
    config.update(
        {
            "type": "enet",
            "alpha": [0.1, 1.0],  # This triggers tuning
            "l1_ratio": 0.5,
        }
    )

    model = SklearnModel(name="test_tuning", config=config)
    # We don't need to mock the return value of the model, just check that fit is called
    model.train(train_df, val_df)

    # The assertion is now on the patched fit method
    mock_grid_search_fit.assert_called_once()


def test_predict_logic(base_config, sample_data):
    """Tests the predict method on a trained model."""
    train_df, _ = sample_data
    config = base_config.copy()
    config["type"] = "ols"

    model = SklearnModel(name="test_ols", config=config)
    model.train(train_df)  # Train a real, simple model

    # Create prediction data
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
            "feat1": [0.1, None],  # One row has a null
            "feat2": [0.4, 0.5],
        }
    )

    predictions = model.predict(pred_df_with_nulls)

    assert len(predictions) == 2
    assert predictions.to_list() == [predictions[0], None]
    assert not predictions.is_null()[0]  # Check the boolean mask at index 0
    assert predictions.is_null()[1]  # Check the boolean mask at index 1


def test_train_handles_empty_clean_data(base_config, caplog):
    """Tests that training is skipped if no clean data is available."""
    config = base_config.copy()
    config["type"] = "ols"

    # Create a dataframe that will be empty after dropping nulls
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

    # Mock the fit method to check the `sample_weight` argument
    with patch.object(Pipeline, "fit") as mock_fit:
        model.train(train_df)
        mock_fit.assert_called_once()
        # Check that 'model__sample_weight' was passed in kwargs
        _, kwargs = mock_fit.call_args
        assert "model__sample_weight" in kwargs
        assert isinstance(kwargs["model__sample_weight"], np.ndarray)
