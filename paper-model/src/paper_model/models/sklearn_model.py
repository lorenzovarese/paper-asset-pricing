import polars as pl
import numpy as np
import logging
from typing import Any, Dict, Optional

from sklearn.linear_model import (  # type: ignore
    LinearRegression,
    ElasticNet,
    HuberRegressor,
    SGDRegressor,
)
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.cross_decomposition import PLSRegression  # type: ignore
from sklearn.model_selection import GridSearchCV, PredefinedSplit  # type: ignore

from .base import BaseModel

logger = logging.getLogger(__name__)


class SklearnModel(BaseModel):
    """
    A generic wrapper for scikit-learn compatible models.
    Handles model creation, training (with optional hyperparameter tuning), and prediction.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.target_col = config["target_column"]
        self.feature_cols = config["feature_columns"]
        self.model: Optional[Pipeline] = None

    def _create_pipeline(self, model_instance: Any) -> Pipeline:
        """Helper to create a standard pipeline with a scaler."""
        model_type = self.config["type"]
        steps = [("scaler", StandardScaler())]

        if model_type == "pcr":
            steps.append(("pcr_pipeline", model_instance))
        else:
            steps.append(("model", model_instance))

        return Pipeline(steps)

    def train(
        self, train_data: pl.DataFrame, validation_data: Optional[pl.DataFrame] = None
    ) -> None:
        """
        Trains the sklearn model. If the configuration specifies multiple hyperparameters,
        it performs a grid search using the validation set.
        """
        logger.info(f"Training {self.config['type']} model '{self.name}'...")

        model_config = self.config
        model_type = model_config["type"]
        objective = model_config.get("objective_function", "l2")
        random_state = model_config.get("random_state")

        # UPDATED: Generalize the check for any model requiring tuning
        is_tuning_required = (
            (
                model_type == "enet"
                and (
                    isinstance(model_config.get("alpha"), list)
                    or isinstance(model_config.get("l1_ratio"), list)
                )
            )
            or (
                model_type == "pcr"
                and isinstance(model_config.get("n_components"), list)
            )
            or (
                model_type == "pls"
                and isinstance(model_config.get("n_components"), list)
            )
        )

        if is_tuning_required and (
            validation_data is None or validation_data.is_empty()
        ):
            raise ValueError(
                f"Model '{self.name}' requires hyperparameter tuning, but no validation data was provided."
            )

        # --- Prepare data ---
        required_cols = [self.target_col] + self.feature_cols
        ps: Optional[PredefinedSplit] = None

        if is_tuning_required:
            if validation_data is None:
                raise ValueError(
                    "Logically unreachable: Validation data is required for tuning."
                )

            full_train_data = pl.concat([train_data, validation_data])
            clean_data = full_train_data.drop_nulls(subset=required_cols)

            train_indices = np.full(
                train_data.drop_nulls(subset=required_cols).height, -1, dtype=int
            )
            validation_indices = np.zeros(
                validation_data.drop_nulls(subset=required_cols).height, dtype=int
            )
            test_fold = np.concatenate([train_indices, validation_indices])
            ps = PredefinedSplit(test_fold)
        else:
            clean_data = train_data.drop_nulls(subset=required_cols)

        if clean_data.is_empty():
            logger.warning(
                f"No clean data available for training for model '{self.name}'."
            )
            return

        X = clean_data.select(self.feature_cols).to_numpy()
        y = clean_data.select(self.target_col).to_numpy().ravel()

        if len(X) < 2:
            logger.warning(f"Not enough samples to train model '{self.name}'.")
            return

        # --- Model Fitting ---
        if is_tuning_required:
            if ps is None:
                raise RuntimeError("PredefinedSplit was not created for tuning.")

            logger.info(f"Starting hyperparameter tuning for '{self.name}'...")
            param_grid: Dict[str, Any] = {}
            base_model: Any

            # UPDATED: Build param_grid and base_model based on model_type
            if model_type == "enet":
                if isinstance(model_config.get("alpha"), list):
                    param_grid["model__alpha"] = model_config["alpha"]
                if isinstance(model_config.get("l1_ratio"), list):
                    param_grid["model__l1_ratio"] = model_config["l1_ratio"]
                if objective == "huber":
                    base_model = SGDRegressor(
                        loss="huber",
                        penalty="elasticnet",
                        random_state=random_state,
                        max_iter=1000,
                        tol=1e-3,
                    )
                    if not isinstance(model_config.get("l1_ratio"), list):
                        base_model.set_params(l1_ratio=model_config.get("l1_ratio"))
                else:
                    base_model = ElasticNet(random_state=random_state)
                    if not isinstance(model_config.get("alpha"), list):
                        base_model.set_params(alpha=model_config.get("alpha"))
                    if not isinstance(model_config.get("l1_ratio"), list):
                        base_model.set_params(l1_ratio=model_config.get("l1_ratio"))

            elif model_type == "pcr":
                param_grid["pcr_pipeline__pca__n_components"] = model_config[
                    "n_components"
                ]
                regressor = (
                    HuberRegressor() if objective == "huber" else LinearRegression()
                )
                base_model = Pipeline([("pca", PCA()), ("regressor", regressor)])

            elif model_type == "pls":
                param_grid["model__n_components"] = model_config["n_components"]
                base_model = PLSRegression()

            pipeline = self._create_pipeline(base_model)
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=ps,
                scoring="neg_mean_squared_error",
                refit=True,
                n_jobs=-1,
            )
            grid_search.fit(X, y)
            logger.info(f"Best params for '{self.name}': {grid_search.best_params_}")
            self.model = grid_search.best_estimator_

        else:
            # This block for non-tuned models remains largely the same
            model_instance: Any
            fit_params = {}

            if model_type == "ols":
                # OLS logic as before
                weights: Optional[np.ndarray] = None
                weighting_scheme = model_config.get("weighting_scheme")
                if weighting_scheme == "inv_n_stocks":
                    date_col = self.config["date_column"]
                    n_stocks_per_month = (
                        clean_data.group_by(pl.col(date_col).dt.truncate("1mo"))
                        .len()
                        .rename({"len": "n_stocks"})
                    )
                    weighted_data = clean_data.join(
                        n_stocks_per_month, on=pl.col(date_col).dt.truncate("1mo")
                    )
                    weights = (1 / weighted_data.get_column("n_stocks")).to_numpy()
                elif weighting_scheme == "mkt_cap":
                    mkt_cap_col = model_config.get("market_cap_column")
                    if mkt_cap_col not in clean_data.columns:
                        raise ValueError(
                            f"Market cap column '{mkt_cap_col}' not found in data."
                        )
                    weights = clean_data.get_column(mkt_cap_col).to_numpy()
                if weights is not None:
                    fit_params["model__sample_weight"] = weights
                if objective == "huber":
                    huber_epsilon = 1.35
                    huber_quantile = model_config.get("huber_epsilon_quantile")
                    if huber_quantile is not None:
                        temp_ols = LinearRegression()
                        temp_pipeline = Pipeline(
                            [("scaler", StandardScaler()), ("model", temp_ols)]
                        )
                        temp_pipeline.fit(X, y, **fit_params)
                        residuals = y - temp_pipeline.predict(X)
                        huber_epsilon = np.quantile(np.abs(residuals), huber_quantile)
                        logger.info(
                            f"Calculated Huber epsilon (ξ) = {huber_epsilon:.4f}"
                        )
                    model_instance = SGDRegressor(
                        loss="huber",
                        penalty=None,
                        epsilon=huber_epsilon,
                        random_state=random_state,
                    )
                else:
                    model_instance = LinearRegression()

            elif model_type == "enet":
                if objective == "huber":
                    model_instance = SGDRegressor(
                        loss="huber",
                        penalty="elasticnet",
                        alpha=model_config["alpha"],
                        l1_ratio=model_config["l1_ratio"],
                        random_state=random_state,
                    )
                else:
                    model_instance = ElasticNet(
                        alpha=model_config["alpha"],
                        l1_ratio=model_config["l1_ratio"],
                        random_state=random_state,
                    )

            elif model_type == "pcr":
                regressor = (
                    HuberRegressor() if objective == "huber" else LinearRegression()
                )
                model_instance = Pipeline(
                    [
                        ("pca", PCA(n_components=model_config["n_components"])),
                        ("regressor", regressor),
                    ]
                )

            elif model_type == "pls":
                model_instance = PLSRegression(
                    n_components=model_config["n_components"]
                )

            else:
                raise ValueError(
                    f"Unsupported model type for SklearnModel: {model_type}"
                )

            self.model = self._create_pipeline(model_instance)
            self.model.fit(X, y, **fit_params)

        logger.info(f"Model '{self.name}' training complete.")

    def predict(self, data: pl.DataFrame) -> pl.Series:
        """Generates predictions using the trained model."""
        if self.model is None:
            logger.warning(
                f"Model '{self.name}' is not trained. Cannot generate predictions."
            )
            return pl.Series(
                name=f"{self.target_col}_predicted",
                values=[None] * len(data),
                dtype=pl.Float64,
            )

        missing_cols = [col for col in self.feature_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required feature columns for prediction: {missing_cols}"
            )

        data_for_prediction = data.drop_nulls(subset=self.feature_cols)

        if data_for_prediction.is_empty():
            return pl.Series(
                name=f"{self.target_col}_predicted",
                values=[None] * len(data),
                dtype=pl.Float64,
            )

        X_pred = data_for_prediction.select(self.feature_cols).to_numpy()
        predicted_values = self.model.predict(X_pred)

        prediction_df = data_for_prediction.select(
            [self.config["date_column"], self.config["id_column"]]
        ).with_columns(
            pl.Series(name=f"{self.target_col}_predicted", values=predicted_values)
        )

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
