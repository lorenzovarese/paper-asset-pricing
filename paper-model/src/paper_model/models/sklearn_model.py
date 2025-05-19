# paper-model/src/paper_model/models/sklearn_model.py

import polars as pl
import logging
from typing import Any, Dict
from sklearn.linear_model import LinearRegression, ElasticNet, HuberRegressor  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.cross_decomposition import PLSRegression  # type: ignore

from .base import BaseModel

logger = logging.getLogger(__name__)


class SklearnModel(BaseModel):
    """
    A generic wrapper for scikit-learn compatible models.
    Handles model creation, training, and prediction based on configuration.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.target_col = config["target_column"]
        self.feature_cols = config["feature_columns"]
        self.model = self._create_model()

    def _create_model(self) -> Pipeline:
        """Factory method to create the sklearn model pipeline."""
        model_type = self.config["type"]
        objective = self.config.get("objective_function", "l2")
        random_state = self.config.get("random_state")

        steps = [("scaler", StandardScaler())]

        model_instance: Any
        if model_type == "ols":
            model_instance = (
                HuberRegressor() if objective == "huber" else LinearRegression()
            )
        elif model_type == "enet":
            model_instance = ElasticNet(
                alpha=self.config["alpha"],
                l1_ratio=self.config["l1_ratio"],
                random_state=random_state,
            )
        elif model_type == "pcr":
            model_instance = Pipeline(
                [
                    ("pca", PCA(n_components=self.config["n_components"])),
                    (
                        "regressor",
                        HuberRegressor()
                        if objective == "huber"
                        else LinearRegression(),
                    ),
                ]
            )
        elif model_type == "pls":
            model_instance = PLSRegression(n_components=self.config["n_components"])
        else:
            raise ValueError(f"Unsupported model type for SklearnModel: {model_type}")

        if model_type not in ["pcr"]:  # PCR has its own internal pipeline
            steps.append(("model", model_instance))
            return Pipeline(steps)
        else:
            steps.append(("pcr_pipeline", model_instance))
            return Pipeline(steps)

    def train(self, data: pl.DataFrame) -> None:
        """Trains the sklearn model."""
        logger.info(f"Training {self.config['type']} model '{self.name}'...")

        if not self.model:
            logger.error("Model instance is not initialized. Cannot train.")
            return

        required_cols = [self.target_col] + self.feature_cols
        clean_data = data.drop_nulls(subset=required_cols)

        if clean_data.is_empty():
            logger.warning("No clean data available for training after dropping nulls.")
            return

        X = clean_data.select(self.feature_cols).to_numpy()
        y = clean_data.select(self.target_col).to_numpy().ravel()

        if len(X) < 2:
            logger.warning("Not enough samples to train the model.")
            return

        self.model.fit(X, y)
        logger.info(f"Model '{self.name}' training complete.")

    def predict(self, data: pl.DataFrame) -> pl.Series:
        """Generates predictions using the trained model."""
        if self.model is None or not hasattr(self.model, "predict"):
            logger.warning("Model not trained or invalid. Cannot generate predictions.")
            return pl.Series(name=f"{self.target_col}_predicted", dtype=pl.Float64)

        # Ensure all feature columns are present
        missing_cols = [col for col in self.feature_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required feature columns for prediction: {missing_cols}"
            )

        # Predict only on rows with valid features
        data_for_prediction = data.drop_nulls(subset=self.feature_cols)
        if data_for_prediction.is_empty():
            return pl.Series(name=f"{self.target_col}_predicted", dtype=pl.Float64)

        X_pred = data_for_prediction.select(self.feature_cols).to_numpy()
        predicted_values = self.model.predict(X_pred)

        # Create a DataFrame with predictions and identifiers
        prediction_df = data_for_prediction.select(
            [self.config["date_column"], self.config["id_column"]]
        ).with_columns(
            pl.Series(name=f"{self.target_col}_predicted", values=predicted_values)
        )

        # Join back to the original data to align predictions, filling non-predicted rows with null
        full_predictions = data.select(
            [self.config["date_column"], self.config["id_column"]]
        ).join(
            prediction_df,
            on=[self.config["date_column"], self.config["id_column"]],
            how="left",
        )

        return full_predictions.get_column(f"{self.target_col}_predicted")

    def evaluate(self, y_true: pl.Series, y_pred: pl.Series) -> Dict[str, Any]:
        # This method can be kept in the base class or manager if evaluation is generic
        raise NotImplementedError("Evaluation logic is handled by the ModelManager.")
