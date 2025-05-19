import polars as pl
import logging
from pathlib import Path
from typing import Dict, Union, Any

from .config_parser import PortfolioConfig
from .evaluation import metrics, reporter

logger = logging.getLogger(__name__)


class PortfolioManager:
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.reporter: reporter.PortfolioReporter | None = None

    def _load_data(self, project_root: Path) -> Dict[str, pl.DataFrame]:
        """Loads prediction files and merges them with processed data."""
        input_conf = self.config.input_data
        predictions_dir = project_root / "models" / "predictions"
        processed_dir = project_root / "data" / "processed"

        # Load main processed data
        processed_files = list(
            processed_dir.glob(f"{input_conf.processed_dataset_name}_*.parquet")
        )
        if not processed_files:
            raise FileNotFoundError(
                f"No processed data found for pattern '{input_conf.processed_dataset_name}'"
            )

        main_df = pl.concat([pl.read_parquet(f) for f in processed_files])

        merged_data = {}
        for model_name in input_conf.prediction_model_names:
            pred_file = predictions_dir / f"{model_name}_predictions.parquet"
            if not pred_file.exists():
                logger.warning(
                    f"Prediction file for model '{model_name}' not found. Skipping."
                )
                continue

            preds_df = pl.read_parquet(pred_file)

            # Merge predictions with necessary columns from the main dataset
            data_for_model = preds_df.join(
                main_df.select(
                    [
                        input_conf.date_column,
                        input_conf.id_column,
                        input_conf.risk_free_rate_col,
                        input_conf.value_weight_col,
                    ]
                ),
                on=[input_conf.date_column, input_conf.id_column],
                how="left",
            )
            merged_data[model_name] = data_for_model

        return merged_data

    def _calculate_monthly_returns(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculates portfolio returns for each month based on all strategies."""
        all_monthly_returns = []

        for date, monthly_data in data.group_by(
            self.config.input_data.date_column, maintain_order=True
        ):
            monthly_data = monthly_data.drop_nulls(subset=["predicted_ret"])
            if monthly_data.is_empty():
                continue

            for strat in self.config.strategies:
                # --- 1. Validate Quantiles ---
                quantiles = [
                    monthly_data["predicted_ret"].quantile(strat.long_quantiles[0]),
                    monthly_data["predicted_ret"].quantile(strat.long_quantiles[1]),
                    monthly_data["predicted_ret"].quantile(strat.short_quantiles[0]),
                    monthly_data["predicted_ret"].quantile(strat.short_quantiles[1]),
                ]
                if not all(isinstance(q, (float, int)) for q in quantiles):
                    logger.warning(
                        f"Could not compute all quantiles for date {date} and strategy {strat.name}. Skipping."
                    )
                    continue

                q_low_long, q_high_long, q_low_short, q_high_short = quantiles

                # --- 2. Construct Portfolios ---
                long_portfolio = monthly_data.filter(
                    (pl.col("predicted_ret") >= q_low_long)
                    & (pl.col("predicted_ret") <= q_high_long)
                )
                short_portfolio = monthly_data.filter(
                    (pl.col("predicted_ret") >= q_low_short)
                    & (pl.col("predicted_ret") <= q_high_short)
                )

                if long_portfolio.is_empty() or short_portfolio.is_empty():
                    continue

                # --- 3. Calculate Returns with Type Safety ---
                long_return: Union[float, int, None] = None
                short_return: Union[float, int, None] = None

                if strat.weighting_scheme == "equal":
                    # Here, weights are scalars (float)
                    equal_long_weight = 1.0 / len(long_portfolio)
                    equal_short_weight = 1.0 / len(short_portfolio)
                    long_return = (
                        long_portfolio["actual_ret"] * equal_long_weight
                    ).sum()
                    short_return = (
                        short_portfolio["actual_ret"] * equal_short_weight
                    ).sum()

                elif strat.weighting_scheme == "value":
                    # Here, weights are Series
                    value_col = self.config.input_data.value_weight_col
                    long_sum = long_portfolio[value_col].sum()
                    short_sum = short_portfolio[value_col].sum()

                    if (
                        isinstance(long_sum, (int, float))
                        and long_sum > 0
                        and isinstance(short_sum, (int, float))
                        and short_sum > 0
                    ):
                        # Use distinct variable names to avoid type conflicts
                        value_long_weights = long_portfolio[value_col] / long_sum
                        value_short_weights = short_portfolio[value_col] / short_sum
                        long_return = (
                            long_portfolio["actual_ret"] * value_long_weights
                        ).sum()
                        short_return = (
                            short_portfolio["actual_ret"] * value_short_weights
                        ).sum()

                # --- 4. Validate Calculated Returns ---
                if not isinstance(long_return, (float, int)) or not isinstance(
                    short_return, (float, int)
                ):
                    logger.warning(
                        f"Could not calculate valid long/short returns for date {date} and strategy {strat.name}. Skipping."
                    )
                    continue

                portfolio_return = long_return - short_return

                # --- 5. Validate Risk-Free Rate ---
                risk_free_rate = monthly_data[
                    self.config.input_data.risk_free_rate_col
                ].first()
                if not isinstance(risk_free_rate, (float, int)):
                    logger.warning(
                        f"Could not find a valid risk-free rate for date {date}. Skipping."
                    )
                    continue

                all_monthly_returns.append(
                    {
                        "date": date,
                        "strategy": strat.name,
                        "portfolio_return": portfolio_return,
                        "risk_free_rate": risk_free_rate,
                    }
                )

        if not all_monthly_returns:
            return pl.DataFrame()

        return pl.DataFrame(all_monthly_returns)

    def run(self, project_root: Union[str, Path]):
        """Main entry point to run the portfolio evaluation pipeline."""
        project_root = Path(project_root).expanduser()
        self.reporter = reporter.PortfolioReporter(
            project_root / "portfolios" / "results"
        )

        logger.info("--- Loading Portfolio Data ---")
        all_model_data = self._load_data(project_root)

        for model_name, data in all_model_data.items():
            logger.info(f"--- Processing Portfolios for Model: {model_name} ---")

            monthly_returns_df = self._calculate_monthly_returns(data)
            if monthly_returns_df.is_empty():
                logger.warning(
                    f"No monthly returns could be calculated for model '{model_name}'."
                )
                continue

            for strat_name_raw, strategy_returns in monthly_returns_df.group_by(
                "strategy"
            ):
                strat_name = str(strat_name_raw)
                logger.info(f"Evaluating strategy: {strat_name}")

                strategy_returns = strategy_returns.sort("date")
                if self.reporter:
                    self.reporter.save_monthly_returns(
                        model_name, strat_name, strategy_returns
                    )

                summary_metrics: Dict[str, Any] = {}
                if "sharpe_ratio" in self.config.metrics:
                    summary_metrics["sharpe_ratio"] = metrics.annualized_sharpe_ratio(
                        strategy_returns["portfolio_return"],
                        strategy_returns["risk_free_rate"],
                    )
                if "expected_shortfall" in self.config.metrics:
                    summary_metrics["expected_shortfall"] = metrics.expected_shortfall(
                        strategy_returns["portfolio_return"]
                    )

                if "cumulative_return" in self.config.metrics:
                    cum_ret_series = metrics.cumulative_return(
                        strategy_returns["portfolio_return"]
                    )
                    if not cum_ret_series.is_empty():
                        summary_metrics["final_cumulative_return"] = cum_ret_series[-1]

                    plot_df = strategy_returns.with_columns(
                        cumulative_return=cum_ret_series
                    )
                    if self.reporter:
                        self.reporter.plot_cumulative_returns(
                            model_name, strat_name, plot_df
                        )

                if self.reporter:
                    self.reporter.generate_report(
                        model_name, strat_name, summary_metrics
                    )

        logger.info("Portfolio evaluation completed successfully.")
