import pytest
import torch
import polars as pl
from polars.testing import assert_series_equal
import numpy as np
from unittest.mock import patch, MagicMock

from paper_model.models.torch_model import (  # type: ignore
    TorchModel,
    EarlyStopper,
    FeedForwardNN,
)

# --- Fixtures ---


@pytest.fixture
def base_config():
    """Provides a base configuration for a TorchModel."""
    return {
        "name": "test_nn",
        "target_column": "ret",
        "feature_columns": ["feat1", "feat2"],
        "date_column": "date",
        "id_column": "permno",
        "hidden_layer_sizes": (32, 16),
        "alpha": 0.01,
        "learning_rate": 0.001,
        "batch_size": 10,
        "epochs": 3,
        "patience": 2,
        "n_ensembles": 2,
        "random_state": 42,
        "device": "cpu",  # Force CPU for testing
    }


@pytest.fixture
def sample_data():
    """Provides sample training, validation, and prediction data."""
    train_df = pl.DataFrame(
        {
            "date": ["2020-01-31"] * 10,
            "permno": range(10),
            "feat1": np.random.rand(10),
            "feat2": np.random.rand(10),
            "ret": np.random.rand(10),
        }
    )
    val_df = pl.DataFrame(
        {
            "date": ["2020-02-29"] * 5,
            "permno": range(5),
            "feat1": np.random.rand(5),
            "feat2": np.random.rand(5),
            "ret": np.random.rand(5),
        }
    )
    pred_df = pl.DataFrame(
        {
            "date": ["2020-03-31"] * 3,
            "permno": range(3),
            "feat1": np.random.rand(3),
            "feat2": np.random.rand(3),
        }
    )
    return train_df, val_df, pred_df


# --- Unit Tests for Helper Classes ---


def test_early_stopper():
    stopper = EarlyStopper(patience=2, min_delta=0.01)

    assert not stopper.early_stop(10.0)
    assert stopper.min_validation_loss == 10.0
    assert stopper.counter == 0

    assert not stopper.early_stop(9.5)
    assert stopper.min_validation_loss == 9.5
    assert stopper.counter == 0

    assert not stopper.early_stop(9.505)
    assert stopper.counter == 0

    assert not stopper.early_stop(9.6)
    assert stopper.counter == 1
    assert stopper.min_validation_loss == 9.5

    assert stopper.early_stop(9.6)
    assert stopper.counter == 2


def test_feed_forward_nn():
    net = FeedForwardNN(input_size=10, hidden_layer_sizes=(32, 16))
    # Check layer types and sizes
    assert isinstance(net.network[0], torch.nn.Linear)
    assert net.network[0].in_features == 10
    assert net.network[0].out_features == 32
    assert isinstance(net.network[1], torch.nn.BatchNorm1d)
    assert isinstance(net.network[3], torch.nn.Linear)
    assert net.network[3].in_features == 32
    assert net.network[3].out_features == 16
    assert isinstance(net.network[6], torch.nn.Linear)
    assert net.network[6].in_features == 16
    assert net.network[6].out_features == 1

    # Test forward pass
    input_tensor = torch.randn(5, 10)  # batch_size=5, input_size=10
    output = net(input_tensor)
    assert output.shape == (5, 1)


# --- Tests for TorchModel Class ---


def test_torch_model_initialization(base_config):
    model = TorchModel(name="test_nn", config=base_config)
    assert model.name == "test_nn"
    assert model.device.type == "cpu"
    assert model.model is None


def test_train_raises_error_without_validation_data(base_config, sample_data):
    train_df, _, _ = sample_data
    model = TorchModel(name="test_nn", config=base_config)
    with pytest.raises(ValueError, match="requires a validation set"):
        model.train(train_df, validation_data=None)
    with pytest.raises(ValueError, match="requires a validation set"):
        model.train(train_df, validation_data=pl.DataFrame())


def test_train_handles_empty_data(base_config, sample_data, caplog):
    _, val_df, _ = sample_data
    empty_df = pl.DataFrame({"feat1": [], "feat2": [], "ret": []})
    model = TorchModel(name="test_nn", config=base_config)

    model.train(empty_df, val_df)
    assert "Skipped training: not enough clean data" in caplog.text
    assert model.model is None


@patch("paper_model.models.torch_model.TorchModel._train_single_net")
def test_train_non_tuning_path(mock_train_single, base_config, sample_data):
    """Tests the main training path without hyperparameter tuning."""
    train_df, val_df, _ = sample_data

    # Mock the training loop to return a dummy network
    mock_net = MagicMock(spec=FeedForwardNN)
    mock_train_single.return_value = mock_net

    model = TorchModel(name="test_nn", config=base_config)
    model.train(train_df, val_df)

    # Assertions
    assert mock_train_single.call_count == base_config["n_ensembles"]
    assert model.model is not None
    assert len(model.model) == base_config["n_ensembles"]
    assert all(m is mock_net for m in model.model)


@patch("paper_model.models.torch_model.TorchModel._train_single_net")
@patch("paper_model.models.torch_model.r2_out_of_sample")
def test_train_tuning_path(mock_r2, mock_train_single, base_config, sample_data):
    """Tests the hyperparameter tuning logic."""
    train_df, val_df, _ = sample_data

    # Configure mocks
    mock_train_single.return_value = MagicMock(spec=FeedForwardNN)
    # Simulate that the second set of params is better
    mock_r2.side_effect = [0.5, 0.8]

    # Update config to require tuning
    tuning_config = base_config.copy()
    tuning_config["alpha"] = [0.01, 0.1]  # Two alpha values
    tuning_config["learning_rate"] = 0.001  # One learning rate

    model = TorchModel(name="test_nn_tuning", config=tuning_config)

    # Mock the predict method inside the tuning loop
    with patch.object(model, "predict", return_value=pl.Series([0.5] * len(val_df))):
        model.train(train_df, val_df)

    # Assertions
    # Should be called for each combination of params (2 alphas * 1 lr) * n_ensembles
    assert mock_train_single.call_count == 2 * base_config["n_ensembles"]
    assert model.model is not None
    # The final model should be the one trained with the best params
    assert len(model.model) == base_config["n_ensembles"]


def test_predict_without_model(base_config, sample_data, caplog):
    """Tests that predict returns nulls if the model is not trained."""
    _, _, pred_df = sample_data
    model = TorchModel(name="test_nn", config=base_config)

    predictions = model.predict(pred_df)

    assert "Model 'test_nn' is not trained" in caplog.text
    assert predictions.is_null().all()
    assert len(predictions) == len(pred_df)


def test_predict_with_trained_model(base_config, sample_data):
    """Tests the prediction path with a trained (mocked) model."""
    train_df, val_df, pred_df = sample_data

    model = TorchModel(name="test_nn", config=base_config)

    # Manually set a "trained" model
    mock_net = MagicMock(spec=FeedForwardNN)
    # Make the mock return a predictable tensor
    mock_net.return_value = torch.tensor([[0.5], [0.6], [0.7]], dtype=torch.float32)
    model.model = [mock_net] * base_config["n_ensembles"]

    # Fit the scaler
    model.scaler.fit(train_df.select(model.feature_cols).to_numpy())

    predictions = model.predict(pred_df)

    assert not predictions.is_null().any()
    assert len(predictions) == len(pred_df)
    expected = pl.Series(
        "ret_predicted",
        [0.5, 0.6, 0.7],
        dtype=pl.Float32,  # Adjust dtype as needed
    )
    assert_series_equal(predictions, expected, check_names=False)


def test_r2_out_of_sample_logic():
    from paper_model.evaluation.metrics import r2_out_of_sample  # type: ignore

    y_true = np.array([1, 2, 3, 4])
    y_pred_perfect = np.array([1, 2, 3, 4])
    y_pred_good = np.array([1.1, 1.9, 3.2, 4.0])
    y_pred_bad = np.array([10, -5, 12, -8])
    y_pred_null = np.array([0, 0, 0, 0])

    assert r2_out_of_sample(y_true, y_pred_perfect) == 1.0
    assert r2_out_of_sample(y_true, y_pred_good) > 0.9
    assert r2_out_of_sample(y_true, y_pred_bad) < 0

    ss_tot = np.sum(y_true**2)
    expected_r2_null = 1 - (np.sum(y_true**2) / ss_tot)
    assert r2_out_of_sample(y_true, y_pred_null) == pytest.approx(expected_r2_null)

    # Edge case: all true values are zero
    y_true_zero = np.array([0, 0, 0])
    assert r2_out_of_sample(y_true_zero, np.array([0, 0, 0])) == 1.0
    assert r2_out_of_sample(y_true_zero, np.array([0.1, 0, 0])) == -np.inf
