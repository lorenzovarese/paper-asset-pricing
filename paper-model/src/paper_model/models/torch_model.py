import polars as pl
import numpy as np
import logging
from typing import Any, Dict, Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler  # type: ignore

from .base import BaseModel

logger = logging.getLogger(__name__)

# --- Helper Classes ---


class EarlyStopper:
    """Simple early stopping implementation."""

    def __init__(self, patience: int = 5, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class FeedForwardNN(nn.Module):
    """Dynamically creates a feed-forward neural network with batch normalization."""

    def __init__(self, input_size: int, hidden_layer_sizes: Tuple[int, ...]):
        super().__init__()
        layers: List[nn.Module] = []

        in_features = input_size
        for i, out_features in enumerate(hidden_layer_sizes):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            in_features = out_features

        layers.append(nn.Linear(in_features, 1))  # Final output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# --- Main Model Class ---


class TorchModel(BaseModel[List[FeedForwardNN]]):
    """
    A wrapper for PyTorch-based neural network models.
    Handles ensembling, training loop, early stopping, and prediction.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.target_col = config["target_column"]
        self.feature_cols = config["feature_columns"]
        # This now correctly matches the inherited generic type
        self.model: Optional[List[FeedForwardNN]] = None
        self.scaler = StandardScaler()

    def _train_ensemble(
        self,
        train_data: pl.DataFrame,
        validation_data: pl.DataFrame,
        alpha: float,
        learning_rate: float,
    ) -> List[FeedForwardNN]:
        """Trains an ensemble of neural networks."""

        # Prepare data
        X_train_raw = train_data.select(self.feature_cols).to_numpy()
        y_train_np = train_data.select(self.target_col).to_numpy().ravel()
        X_val_raw = validation_data.select(self.feature_cols).to_numpy()
        y_val_np = validation_data.select(self.target_col).to_numpy().ravel()

        # Fit scaler on training data and transform both sets
        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        X_val_scaled = self.scaler.transform(X_val_raw)

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=self.config["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val_tensor, y_val_tensor),
            batch_size=self.config["batch_size"],
        )

        ensemble_models: List[FeedForwardNN] = []
        for i in range(self.config["n_ensembles"]):
            logger.debug(
                f"Training ensemble member {i + 1}/{self.config['n_ensembles']}..."
            )

            # Set seed for this member's initialization
            torch.manual_seed(self.config.get("random_state", 0) + i)

            net = FeedForwardNN(
                input_size=X_train_tensor.shape[1],
                hidden_layer_sizes=self.config["hidden_layer_sizes"],
            )

            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
            loss_fn = nn.MSELoss()  # L2 loss
            early_stopper = EarlyStopper(patience=self.config["patience"])

            for epoch in range(self.config["epochs"]):
                # Training loop
                net.train()
                for X_batch, y_batch in train_loader:
                    y_pred = net(X_batch)
                    l2_loss = loss_fn(y_pred, y_batch)

                    # Add L1 regularization
                    l1_loss = 0
                    for param in net.parameters():
                        l1_loss += torch.linalg.vector_norm(param, ord=1)

                    total_loss = l2_loss + alpha * l1_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                # Validation loop
                net.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_val_batch, y_val_batch in val_loader:
                        y_val_pred = net(X_val_batch)
                        val_loss += loss_fn(y_val_pred, y_val_batch).item()

                avg_val_loss = val_loss / len(val_loader)
                logger.debug(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.6f}")

                if early_stopper.early_stop(avg_val_loss):
                    logger.info(
                        f"Early stopping triggered at epoch {epoch + 1} for ensemble member {i + 1}."
                    )
                    break

            ensemble_models.append(net)

        return ensemble_models

    def train(
        self, train_data: pl.DataFrame, validation_data: Optional[pl.DataFrame] = None
    ) -> None:
        """Orchestrates the training process, including hyperparameter tuning."""
        if validation_data is None or validation_data.is_empty():
            raise ValueError(
                f"Neural network model '{self.name}' requires a validation set for early stopping."
            )

        # Manual grid search if tuning is required
        if self.config.get("requires_tuning", False):
            logger.info(f"Starting hyperparameter tuning for '{self.name}'...")
            best_score = -np.inf
            best_params = {}
            best_ensemble = None

            alphas = (
                self.config["alpha"]
                if isinstance(self.config["alpha"], list)
                else [self.config["alpha"]]
            )
            learning_rates = (
                self.config["learning_rate"]
                if isinstance(self.config["learning_rate"], list)
                else [self.config["learning_rate"]]
            )

            for alpha in alphas:
                for lr in learning_rates:
                    logger.debug(f"Testing NN with alpha={alpha}, learning_rate={lr}")

                    current_ensemble = self._train_ensemble(
                        train_data, validation_data, alpha, lr
                    )

                    y_val_pred_avg = self.predict(
                        validation_data, ensemble=current_ensemble
                    )
                    y_val_true = validation_data.get_column(self.target_col)

                    score = r2_out_of_sample(
                        y_val_true.to_numpy(), y_val_pred_avg.to_numpy()
                    )

                    if score > best_score:
                        best_score = score
                        best_params = {"alpha": alpha, "learning_rate": lr}
                        best_ensemble = current_ensemble

            logger.info(f"Best params for '{self.name}': {best_params}")
            self.model = best_ensemble
        else:
            self.model = self._train_ensemble(
                train_data,
                validation_data,
                alpha=self.config["alpha"],
                learning_rate=self.config["learning_rate"],
            )

    def predict(
        self, data: pl.DataFrame, ensemble: Optional[List[FeedForwardNN]] = None
    ) -> pl.Series:
        """Generates predictions by averaging the ensemble's outputs."""
        models_to_use = ensemble if ensemble is not None else self.model

        if models_to_use is None:
            logger.warning(
                f"Model '{self.name}' is not trained. Cannot generate predictions."
            )
            return pl.Series(
                name=f"{self.target_col}_predicted",
                values=[None] * len(data),
                dtype=pl.Float64,
            )

        X_raw = data.select(self.feature_cols).fill_null(0).to_numpy()
        X_scaled = self.scaler.transform(X_raw)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            all_preds = [net(X_tensor).numpy().ravel() for net in models_to_use]

        avg_preds = np.mean(all_preds, axis=0)

        return pl.Series(name=f"{self.target_col}_predicted", values=avg_preds)

    def evaluate(self, y_true: pl.Series, y_pred: pl.Series) -> Dict[str, Any]:
        raise NotImplementedError("Evaluation logic is handled by the ModelManager.")


def r2_out_of_sample(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true**2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else -np.inf
    return float(1 - ss_res / ss_tot)
