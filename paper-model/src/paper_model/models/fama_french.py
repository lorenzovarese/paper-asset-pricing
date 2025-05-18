from __future__ import annotations
import polars as pl
import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore[import-untyped]
from statsmodels.regression.linear_model import OLSResults  # type: ignore[import-untyped]
import logging
from typing import Any, Dict

from .base import BaseModel
from paper_model.evaluation.metrics import calculate_r_squared

logger = logging.getLogger(__name__)


class FamaFrench3FactorModel(BaseModel):
    """
    Implements a simplified Fama-French 3-Factor Model.

    R_it - R_ft = alpha_i + beta_MKT * (R_Mt - R_ft) + beta_SMB * SMB_t + beta_HML * HML_t + e_it

    This implementation performs a cross-sectional regression for each time period
    or a pooled regression, depending on configuration. For simplicity, we'll
    assume a pooled regression for now, or individual regressions per firm.
    The example will focus on individual firm regressions.
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

        self.firm_models: Dict[Any, OLSResults] = {}
        self.firm_betas: Dict[Any, Dict[str, float]] = {}
        self.firm_predicted_returns: pl.DataFrame | None = None

    def train(self, data: pl.DataFrame) -> None:
        """
        Trains the Fama-French 3-Factor model for each unique firm (permco).
        Assumes `data` contains `date_col`, `id_col`, `target_return_col`,
        `risk_free_rate_col`, and `factor_cols`.
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

        # Prepare features (factors) and target
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
        pdf[self.date_col] = pd.to_datetime(pdf[self.date_col])
        pdf = pdf.set_index([self.date_col, self.id_col])

        unique_firms = pdf.index.get_level_values(self.id_col).unique()

        all_predictions = []
        for firm_id in unique_firms:
            firm_data = pdf.loc[(slice(None), firm_id), :].dropna(
                subset=X_cols + [y_col]
            )

            if firm_data.empty:
                logger.warning(
                    f"Skipping firm {firm_id}: No valid data for regression."
                )
                continue
            if (
                len(firm_data) < len(X_cols) + 2
            ):  # Need at least k+1 observations for k features + intercept
                logger.warning(
                    f"Skipping firm {firm_id}: Not enough observations ({len(firm_data)}) for regression with {len(X_cols)} factors."
                )
                continue

            X = sm.add_constant(firm_data[X_cols])
            y = firm_data[y_col]

            try:
                model = sm.OLS(y, X)
                results = model.fit()
                self.firm_models[firm_id] = results  # Store results object
                self.firm_betas[firm_id] = results.params.drop(
                    "const"
                ).to_dict()  # Store betas

                # Generate predictions for this firm
                predictions = results.predict(X)
                firm_predictions_df = pl.DataFrame(
                    {
                        self.date_col: firm_data.index.get_level_values(
                            self.date_col
                        ).to_list(),
                        self.id_col: firm_id,
                        f"{self.target_return_col}_predicted": predictions.values
                        + firm_data[
                            self.risk_free_rate_col
                        ].values,  # Add back risk-free rate
                    }
                ).with_columns(
                    # --- ADD THIS LINE TO EXPLICITLY CAST TO pl.Date ---
                    pl.col(self.date_col).cast(pl.Date)
                )
                all_predictions.append(firm_predictions_df)

            except Exception as e:
                logger.error(f"Error training FF3 model for firm {firm_id}: {e}")
                continue

        if all_predictions:
            self.firm_predicted_returns = pl.concat(all_predictions)
            logger.info(
                f"FF3 Model training complete for {len(self.firm_models)} firms."
            )
        else:
            logger.warning("No firms were successfully trained for FF3 model.")
            self.firm_predicted_returns = pl.DataFrame(
                {
                    self.date_col: [],
                    self.id_col: [],
                    f"{self.target_return_col}_predicted": [],
                }
            ).with_columns(
                [
                    pl.col(self.date_col).cast(pl.Date),
                    pl.col(self.id_col).cast(pl.Int64),
                    pl.col(f"{self.target_return_col}_predicted").cast(pl.Float64),
                ]
            )

    def evaluate(self, data: pl.DataFrame) -> Dict[str, Any]:
        """
        Evaluates the Fama-French 3-Factor model.
        Calculates R-squared for each firm and averages them.
        """
        logger.info(f"Evaluating Fama-French 3-Factor Model '{self.name}'...")
        if not self.firm_models or self.firm_predicted_returns is None:
            logger.warning(
                "Model not trained or no predictions available for evaluation."
            )
            return {"r_squared_avg": np.nan}

        # Merge actual returns with predicted returns
        eval_df = data.select(
            [self.date_col, self.id_col, self.target_return_col]
        ).join(
            self.firm_predicted_returns, on=[self.date_col, self.id_col], how="inner"
        )

        if eval_df.is_empty():
            logger.warning(
                "No overlapping data for evaluation after merging predictions."
            )
            return {"r_squared_avg": np.nan}

        r_squared_values = []
        for firm_id in eval_df[self.id_col].unique().to_list():
            firm_eval_data = eval_df.filter(pl.col(self.id_col) == firm_id)
            actual = firm_eval_data[self.target_return_col]
            predicted = firm_eval_data[f"{self.target_return_col}_predicted"]

            if (
                len(actual) > 1 and not actual.std() == 0
            ):  # Ensure variance for R-squared calculation
                r2 = calculate_r_squared(actual.to_numpy(), predicted.to_numpy())
                r_squared_values.append(r2)
            else:
                logger.warning(
                    f"Skipping R-squared for firm {firm_id}: Insufficient data or zero variance."
                )

        avg_r_squared = np.mean(r_squared_values) if r_squared_values else np.nan
        self.evaluation_results = {"average_r_squared": avg_r_squared}
        logger.info(f"Evaluation complete. Average R-squared: {avg_r_squared:.4f}")
        return self.evaluation_results

    def generate_checkpoint(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Generates a checkpoint DataFrame containing predicted returns.
        """
        logger.info(
            f"Generating checkpoint for Fama-French 3-Factor Model '{self.name}'..."
        )
        if self.firm_predicted_returns is None:
            logger.warning("No predicted returns available to generate checkpoint.")
            return pl.DataFrame()

        # Add factor exposures (betas) to the checkpoint if desired
        # For simplicity, we'll just output predicted returns for now.
        # A more complex checkpoint might include firm_id, date, predicted_return, beta_MKT, beta_SMB, beta_HML
        checkpoint_df = self.firm_predicted_returns.clone()
        logger.info(f"Checkpoint generated with shape: {checkpoint_df.shape}")
        return checkpoint_df
