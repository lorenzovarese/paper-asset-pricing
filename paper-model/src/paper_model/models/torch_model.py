import polars as pl
import numpy as np
import logging
from typing import Any, Dict, Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from .base import BaseModel
from ..evaluation.metrics import r2_out_of_sample

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
        # If the new loss is better than the best loss we've seen, reset the counter.
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        # If the new loss is NOT an improvement, check if it's worse by more than min_delta.
        # If it is, increment the counter.
        elif validation_loss > self.min_validation_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        # If the loss is the same or slightly worse (within min_delta), do nothing.
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
        self.feature_cols = config.get("feature_columns", [])
        self.model: Optional[List[FeedForwardNN]] = None
        self.scaler = StandardScaler()

        self.device_str = config.get("device", "auto")
        if self.device_str == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.device_str)

        self.num_workers = config.get("num_workers", 0)
        logger.info(
            f"[{self.name}] Using device: '{self.device}' with {self.num_workers} data loader workers."
        )

    def _train_single_net(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        input_size: int,
        alpha: float,
        learning_rate: float,
        ensemble_member_idx: int,
    ) -> FeedForwardNN:
        """Trains a single neural network, to be used as part of an ensemble."""
        logger.debug(
            f"Training ensemble member {ensemble_member_idx + 1}/{self.config['n_ensembles']}..."
        )
        torch.manual_seed(self.config.get("random_state", 0) + ensemble_member_idx)

        net = FeedForwardNN(
            input_size=input_size,
            hidden_layer_sizes=self.config["hidden_layer_sizes"],
        ).to(self.device)

        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        early_stopper = EarlyStopper(patience=self.config["patience"])

        for epoch in range(self.config["epochs"]):
            net.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                y_pred = net(X_batch)
                l2_loss = loss_fn(y_pred, y_batch)
                l1_loss = sum(
                    torch.linalg.vector_norm(p, ord=1) for p in net.parameters()
                )
                total_loss = l2_loss + alpha * l1_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            net.eval()
            val_loss = 0
            with torch.no_grad():
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = (
                        X_val_batch.to(self.device),
                        y_val_batch.to(self.device),
                    )
                    y_val_pred = net(X_val_batch)
                    val_loss += loss_fn(y_val_pred, y_val_batch).item()
            avg_val_loss = val_loss / len(val_loader)
            if early_stopper.early_stop(avg_val_loss):
                logger.info(
                    f"Early stopping triggered at epoch {epoch + 1} for ensemble member {ensemble_member_idx + 1}."
                )
                break
        return net

    def train(
        self, train_data: pl.DataFrame, validation_data: Optional[pl.DataFrame] = None
    ) -> None:
        """
        Orchestrates the training process, including data preparation and hyperparameter tuning.
        """
        if validation_data is None or validation_data.is_empty():
            raise ValueError(
                f"Neural network model '{self.name}' requires a validation set for early stopping."
            )

        logger.info(f"[{self.name}] Preparing data for training...")
        required_cols = self.feature_cols + [self.target_col]
        clean_train_data = train_data.drop_nulls(subset=required_cols)
        clean_val_data = validation_data.drop_nulls(subset=required_cols)

        if clean_train_data.is_empty() or clean_val_data.is_empty():
            logger.warning(
                f"[{self.name}] Skipped training: not enough clean data in training or validation set."
            )
            return

        X_train_raw = clean_train_data.select(self.feature_cols).to_numpy()
        y_train_np = clean_train_data.select(self.target_col).to_numpy().ravel()
        X_val_raw = clean_val_data.select(self.feature_cols).to_numpy()
        y_val_np = clean_val_data.select(self.target_col).to_numpy().ravel()

        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        X_val_scaled = self.scaler.transform(X_val_raw)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1)

        pin_memory = self.device.type == "cuda"
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            TensorDataset(X_val_tensor, y_val_tensor),
            batch_size=self.config["batch_size"],
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )
        input_size = X_train_tensor.shape[1]

        is_tuning_required = isinstance(self.config["alpha"], list) or isinstance(
            self.config["learning_rate"], list
        )

        if is_tuning_required:
            logger.info(f"[{self.name}] Starting hyperparameter tuning...")
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
                    current_ensemble = [
                        self._train_single_net(
                            train_loader, val_loader, input_size, alpha, lr, i
                        )
                        for i in range(self.config["n_ensembles"])
                    ]

                    y_val_pred_full = self.predict(
                        validation_data, ensemble=current_ensemble
                    )
                    temp_df = pl.DataFrame(
                        {
                            "y_true": validation_data.get_column(self.target_col),
                            "y_pred": y_val_pred_full,
                        }
                    ).drop_nulls()

                    if not temp_df.is_empty():
                        score = r2_out_of_sample(
                            temp_df["y_true"].to_numpy(), temp_df["y_pred"].to_numpy()
                        )
                        if score > best_score:
                            best_score = score
                            best_params = {"alpha": alpha, "learning_rate": lr}
                            best_ensemble = current_ensemble

            logger.info(f"Best params for '{self.name}': {best_params}")
            self.model = best_ensemble
        else:
            alpha = self.config["alpha"]
            learning_rate = self.config["learning_rate"]
            if not isinstance(alpha, (int, float)) or not isinstance(
                learning_rate, (int, float)
            ):
                raise TypeError(
                    "For non-tuned models, 'alpha' and 'learning_rate' must be single float values."
                )
            self.model = [
                self._train_single_net(
                    train_loader, val_loader, input_size, alpha, learning_rate, i
                )
                for i in range(self.config["n_ensembles"])
            ]

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

        data_for_prediction = data.drop_nulls(subset=self.feature_cols)
        if data_for_prediction.is_empty():
            return pl.Series(
                name=f"{self.target_col}_predicted",
                values=[None] * len(data),
                dtype=pl.Float64,
            )

        X_raw = data_for_prediction.select(self.feature_cols).to_numpy()
        X_scaled = self.scaler.transform(X_raw)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            for net in models_to_use:
                net.eval()
                net.to(self.device)
            all_preds = [net(X_tensor).cpu().numpy().ravel() for net in models_to_use]

        avg_preds = np.mean(all_preds, axis=0)

        prediction_df = data_for_prediction.select(
            [self.config["date_column"], self.config["id_column"]]
        ).with_columns(pl.Series(name=f"{self.target_col}_predicted", values=avg_preds))

        full_predictions = data.select(
            [self.config["date_column"], self.config["id_column"]]
        ).join(
            prediction_df,
            on=[self.config["date_column"], self.config["id_column"]],
            how="left",
        )

        return full_predictions.get_column(f"{self.target_col}_predicted")

    def evaluate(self, y_true: pl.Series, y_pred: pl.Series) -> Dict[str, Any]:
        raise NotImplementedError("Evaluation logic is handled by the ModelManager.")
