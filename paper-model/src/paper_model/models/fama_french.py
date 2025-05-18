from __future__ import annotations
import polars as pl
import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore[import-untyped]
from statsmodels.regression.linear_model import RegressionResultsWrapper  # type: ignore[import-untyped]
import logging
from typing import Any, Dict

from .base import BaseModel
from paper_model.evaluation.metrics import (
    r2_out_of_sample,
    mean_squared_error,
    r2_adj_out_of_sample,
)

logger = logging.getLogger(__name__)


class FamaFrench3FactorModel(BaseModel):
    """
    Implements a simplified Fama-French 3-Factor Model.

    R_it - R_ft = alpha_i + beta_MKT * (R_Mt - R_ft) + beta_SMB * SMB_t + beta_HML * HML_t + e_it

    This implementation performs a cross-sectional regression for each time period
    or a pooled regression, depending on configuration. For simplicity, we'll
    assume individual regressions per firm.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.target_return_col = config.get("target_return_col", "return")
        self.date_col = config.get("date_column", "date")
        self.id_col = config.get("id_column", "permco")
        self.factor_cols = config.get("factor_columns", ["mkt_rf", "smb", "hml"])
        self.excess_return_col = config.get(
            "excess_return_col", "excess_return"
        )  # R_it - R_ft
        self.risk_free_rate_col = config.get("risk_free_rate_col", "rf")  # R_ft

        self.firm_models: Dict[Any, RegressionResultsWrapper] = {}
        self.firm_betas: Dict[Any, Dict[str, float]] = {}

    def train(self, data: pl.DataFrame) -> None:
        """
        Trains the Fama-French 3-Factor model for each unique firm (permco)
        within the provided training data.
        """
        logger.info(f"Training Fama-French 3-Factor Model '{self.name}'...")

        # Calculate excess returns
        if self.risk_free_rate_col not in data.columns:
            raise ValueError(
                f"Risk-free rate column '{self.risk_free_rate_col}' not found in data."
            )
        if self.target_return_col not in data.columns:
            raise ValueError(
                f"Target return column '{self.target_return_col}' not found in data."
            )

        data_with_excess_return = data.with_columns(
            (pl.col(self.target_return_col) - pl.col(self.risk_free_rate_col)).alias(
                self.excess_return_col
            )
        )

        X_cols = self.factor_cols
        y_col = self.excess_return_col

        missing_cols = [
            col
            for col in X_cols + [y_col]
            if col not in data_with_excess_return.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing required columns for training: {missing_cols}")

        # Convert to Pandas for statsmodels
        pdf = data_with_excess_return.to_pandas()
        # Ensure date column is datetime for proper indexing if needed, though not strictly for OLS
        if self.date_col in pdf.columns:
            pdf[self.date_col] = pd.to_datetime(pdf[self.date_col])
        pdf = pdf.set_index([self.date_col, self.id_col])

        unique_firms = pdf.index.get_level_values(self.id_col).unique()

        self.firm_models = {}  # Clear previous models
        self.firm_betas = {}  # Clear previous betas

        for firm_id in unique_firms:
            firm_data = pdf.loc[(slice(None), firm_id), :].dropna(
                subset=X_cols + [y_col]
            )

            if firm_data.empty:
                logger.debug(
                    f"Skipping firm {firm_id}: No valid data for regression in training window."
                )
                continue
            if (
                len(firm_data) < len(X_cols) + 2
            ):  # Need at least k+1 observations for k features + intercept
                logger.debug(
                    f"Skipping firm {firm_id}: Not enough observations ({len(firm_data)}) for regression with {len(X_cols)} factors."
                )
                continue

            X = sm.add_constant(firm_data[X_cols])
            y = firm_data[y_col]

            try:
                model = sm.OLS(y, X)
                results = model.fit()
                self.firm_models[firm_id] = results
                self.firm_betas[firm_id] = results.params.drop("const").to_dict()
            except Exception as e:
                logger.error(f"Error training FF3 model for firm {firm_id}: {e}")
                continue
        logger.info(f"FF3 Model training complete for {len(self.firm_models)} firms.")

    def predict(self, data: pl.DataFrame) -> pl.Series:
        """
        Generates predictions for the given data using the trained firm-specific models.
        Returns predicted *total* returns.
        """
        if not self.firm_models:
            logger.warning("Model not trained. Cannot generate predictions.")
            return pl.Series(name=f"{self.target_return_col}_predicted", values=[])

        # Calculate excess returns for factors in the prediction data
        if self.risk_free_rate_col not in data.columns:
            raise ValueError(
                f"Risk-free rate column '{self.risk_free_rate_col}' not found in prediction data."
            )

        X_cols = self.factor_cols
        missing_cols = [col for col in X_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required factor columns for prediction: {missing_cols}"
            )

        pdf = data.to_pandas()
        if self.date_col in pdf.columns:
            pdf[self.date_col] = pd.to_datetime(pdf[self.date_col])
        pdf = pdf.set_index([self.date_col, self.id_col])

        all_predictions = []
        for firm_id, firm_model in self.firm_models.items():
            firm_data = pdf.loc[(slice(None), firm_id), :].dropna(subset=X_cols)
            if firm_data.empty:
                continue

            X_pred = sm.add_constant(firm_data[X_cols])
            try:
                # Predict excess returns
                excess_return_predictions = firm_model.predict(X_pred)
                # Add back risk-free rate to get total predicted returns
                total_return_predictions = (
                    excess_return_predictions + firm_data[self.risk_free_rate_col]
                )
                all_predictions.append(
                    pl.DataFrame(
                        {
                            self.date_col: firm_data.index.get_level_values(
                                self.date_col
                            ).to_list(),
                            self.id_col: firm_id,
                            f"{self.target_return_col}_predicted": total_return_predictions.values,
                        }
                    ).with_columns(pl.col(self.date_col).cast(pl.Date))
                )
            except Exception as e:
                logger.warning(
                    f"Could not generate prediction for firm {firm_id} in FF3 model: {e}"
                )
                continue

        if all_predictions:
            # Concatenate all firm predictions and join back to original data structure
            # to ensure predictions align with the original data's date/id structure.
            predicted_df = pl.concat(all_predictions)
            # Join with original data to ensure all original rows are present,
            # and missing predictions are null.
            # This is crucial for evaluation where y_true and y_pred must align.
            full_predictions = data.select(
                [self.date_col, self.id_col, self.target_return_col]
            ).join(predicted_df, on=[self.date_col, self.id_col], how="left")
            return full_predictions[f"{self.target_return_col}_predicted"]
        else:
            logger.warning("No predictions generated for any firm.")
            return pl.Series(name=f"{self.target_return_col}_predicted", values=[])

    def evaluate(self, y_true: pl.Series, y_pred: pl.Series) -> Dict[str, Any]:
        """
        Evaluates the Fama-French 3-Factor model using provided true and predicted values.
        """
        logger.info(f"Evaluating Fama-French 3-Factor Model '{self.name}'...")

        # Ensure y_true and y_pred are aligned and clean
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

        mse = mean_squared_error(y_true_np, y_pred_np)  # type: ignore[call-arg]
        r2_oos = r2_out_of_sample(y_true_np, y_pred_np)

        # For adjusted R2, we need the number of predictors.
        # In FF3, it's typically 3 (MKT-RF, SMB, HML).
        n_predictors = len(self.factor_cols)
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
