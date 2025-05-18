from __future__ import annotations
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
import logging
from typing import Any, Dict

from .base import BaseModel
from paper_model.evaluation.metrics import calculate_r_squared, calculate_mse

logger = logging.getLogger(__name__)


class SimpleLinearRegression(BaseModel):
    """
    Implements a simple Linear Regression model for return prediction.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.target_col = config.get("target_column", "return")
        self.feature_cols = config.get("feature_columns", [])
        self.test_size = config.get("test_size", 0.2)
        self.random_state = config.get("random_state", 42)

        if not self.feature_cols:
            raise ValueError(
                "Linear Regression model requires 'feature_columns' to be specified in config."
            )

        self.model: LinearRegression | None = None
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_test: np.ndarray | None = None
        self.predicted_returns: pl.DataFrame | None = None

    def train(self, data: pl.DataFrame) -> None:
        """
        Trains the Linear Regression model.
        Splits data into training and testing sets.
        """
        logger.info(f"Training Simple Linear Regression Model '{self.name}'...")

        required_cols = [self.target_col] + self.feature_cols
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for training: {missing_cols}")

        # Drop rows with any nulls in target or features for training
        clean_data = data.drop_nulls(subset=required_cols)
        if clean_data.is_empty():
            logger.warning("No clean data available for training after dropping nulls.")
            self.model = None
            return

        X = clean_data.select(self.feature_cols).to_numpy()
        y = clean_data.select(self.target_col).to_numpy().flatten()
        identifiers = clean_data.select(
            [
                self.config.get("date_column", "date"),
                self.config.get("id_column", "permco"),
            ]
        )

        if len(X) < 2:
            logger.warning("Not enough samples to train a linear regression model.")
            self.model = None
            return

        # Split data for training and testing
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, identifiers, test_size=self.test_size, random_state=self.random_state
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.ids_test = ids_test  # Store identifiers for test set to create checkpoint

        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        logger.info("Linear Regression Model training complete.")

    def evaluate(self, data: pl.DataFrame) -> Dict[str, Any]:
        """
        Evaluates the trained Linear Regression model on the test set.
        """
        logger.info(f"Evaluating Simple Linear Regression Model '{self.name}'...")
        if self.model is None or self.X_test is None or self.y_test is None:
            logger.warning(
                "Model not trained or test data not available for evaluation."
            )
            return {"r_squared": np.nan, "mse": np.nan}

        y_pred = self.model.predict(self.X_test)

        r_squared = calculate_r_squared(self.y_test, y_pred)
        mse = calculate_mse(self.y_test, y_pred)

        self.evaluation_results = {"r_squared": r_squared, "mean_squared_error": mse}
        logger.info(f"Evaluation complete. R-squared: {r_squared:.4f}, MSE: {mse:.4f}")
        return self.evaluation_results

    def generate_checkpoint(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Generates a checkpoint DataFrame containing predicted returns for the test set.
        """
        logger.info(
            f"Generating checkpoint for Simple Linear Regression Model '{self.name}'..."
        )
        if self.model is None or self.X_test is None or self.ids_test is None:
            logger.warning(
                "Model not trained or test data not available to generate checkpoint."
            )
            return pl.DataFrame()

        predicted_values = self.model.predict(self.X_test)

        checkpoint_df = pl.DataFrame(
            {
                self.config.get("date_column", "date"): self.ids_test.select(
                    self.config.get("date_column", "date")
                ).to_series(),
                self.config.get("id_column", "permco"): self.ids_test.select(
                    self.config.get("id_column", "permco")
                ).to_series(),
                f"{self.target_col}_predicted": predicted_values,
            }
        )
        logger.info(f"Checkpoint generated with shape: {checkpoint_df.shape}")
        return checkpoint_df
