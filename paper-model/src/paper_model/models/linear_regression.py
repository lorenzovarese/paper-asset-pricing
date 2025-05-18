from __future__ import annotations
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]
import logging
from typing import Any, Dict

from .base import BaseModel
from paper_model.evaluation.metrics import (
    r2_out_of_sample,
    mean_squared_error,
    r2_adj_out_of_sample,
)

logger = logging.getLogger(__name__)


class SimpleLinearRegression(BaseModel):
    """
    Implements a simple Linear Regression model for return prediction.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.target_col = config.get("target_column", "return")
        self.feature_cols = config.get("feature_columns", [])
        self.random_state = config.get("random_state", 42)

        if not self.feature_cols:
            raise ValueError(
                "Linear Regression model requires 'feature_columns' to be specified in config."
            )

        self.model: LinearRegression | None = None

    def train(self, data: pl.DataFrame) -> None:
        """
        Trains the Linear Regression model on the provided data.
        """
        logger.info(f"Training Simple Linear Regression Model '{self.name}'...")

        required_cols = [self.target_col] + self.feature_cols
        clean_data = data.drop_nulls(subset=required_cols)

        if clean_data.is_empty():
            logger.warning("No clean data available for training after dropping nulls.")
            self.model = None
            return

        X = clean_data.select(self.feature_cols).to_numpy()
        y = clean_data.select(self.target_col).to_numpy().flatten()

        if len(X) < 2:
            logger.warning("Not enough samples to train a linear regression model.")
            self.model = None
            return

        self.model = LinearRegression()
        self.model.fit(X, y)
        logger.info("Linear Regression Model training complete.")

    def predict(self, data: pl.DataFrame) -> pl.Series:
        """
        Generates predictions using the trained Linear Regression model.
        """
        if self.model is None:
            logger.warning("Model not trained. Cannot generate predictions.")
            return pl.Series(name=f"{self.target_col}_predicted", values=[])

        missing_cols = [col for col in self.feature_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required feature columns for prediction: {missing_cols}"
            )

        # Drop nulls only for the feature columns used for prediction
        # This ensures that we only predict for rows where features are available
        data_for_prediction = data.drop_nulls(subset=self.feature_cols)

        if data_for_prediction.is_empty():
            logger.warning(
                "No valid data for prediction after dropping nulls in features."
            )
            return pl.Series(name=f"{self.target_col}_predicted", values=[])

        X_pred = data_for_prediction.select(self.feature_cols).to_numpy()
        predicted_values = self.model.predict(X_pred)

        # Create a Polars DataFrame with original identifiers and predictions
        # Then join back to the original 'data' to ensure alignment and handle missing predictions
        # for rows that had null features.
        prediction_df = pl.DataFrame(
            {
                self.config.get("date_column", "date"): data_for_prediction.select(
                    self.config.get("date_column", "date")
                ).to_series(),
                self.config.get("id_column", "permco"): data_for_prediction.select(
                    self.config.get("id_column", "permco")
                ).to_series(),
                f"{self.target_col}_predicted": predicted_values,
            }
        )

        # Join back to the original data to ensure the output series has the same length and alignment
        # as the input data's target column. Rows without predictions will have nulls.
        full_predictions = data.select(
            [
                self.config.get("date_column", "date"),
                self.config.get("id_column", "permco"),
            ]
        ).join(
            prediction_df,
            on=[
                self.config.get("date_column", "date"),
                self.config.get("id_column", "permco"),
            ],
            how="left",
        )

        return full_predictions[f"{self.target_col}_predicted"]

    def evaluate(self, y_true: pl.Series, y_pred: pl.Series) -> Dict[str, Any]:
        """
        Evaluates the trained Linear Regression model using provided true and predicted values.
        """
        logger.info(f"Evaluating Simple Linear Regression Model '{self.name}'...")

        # Filter out NaNs from either series to ensure valid comparison
        combined_df = pl.DataFrame({"y_true": y_true, "y_pred": y_pred}).drop_nulls()

        if combined_df.is_empty():
            logger.warning("No valid data points for evaluation after dropping nulls.")
            return {"mse": np.nan, "r2_oos": np.nan, "r2_adj_oos": np.nan}

        y_true_np = combined_df["y_true"].to_numpy()
        y_pred_np = combined_df["y_pred"].to_numpy()

        if len(y_true_np) < 2:
            logger.warning("Not enough data points for meaningful evaluation.")
            return {"mse": np.nan, "r2_oos": np.nan, "r2_adj_oos": np.nan}

        mse = mean_squared_error(y_true_np, y_pred_np)  # type: ignore[arg-type]
        r2_oos = r2_out_of_sample(y_true_np, y_pred_np)

        # For adjusted R2, we need the number of predictors.
        n_predictors = len(self.feature_cols)
        r2_adj_oos = r2_adj_out_of_sample(y_true_np, y_pred_np, n_predictors)

        metrics = {
            "mse": mse,
            "r2_oos": r2_oos,
            "r2_adj_oos": r2_adj_oos,
        }
        logger.info(
            f"Evaluation complete. MSE: {mse:.4f}, R2_oos: {r2_oos:.4f}, R2_adj_oos: {r2_adj_oos:.4f}"
        )
        return metrics
