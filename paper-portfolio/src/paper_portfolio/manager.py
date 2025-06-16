import polars as pl
import logging
from pathlib import Path
from typing import Dict, Union, Any, Optional, Tuple

from .config_parser import PortfolioConfig
from .evaluation import metrics, reporter

logger = logging.getLogger(__name__)


class PortfolioManager:
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.reporter: reporter.PortfolioReporter | None = None

    def _load_additional_dataset(
        self, project_root: Path, config: Any
    ) -> Optional[pl.DataFrame]:
        """Helper function to load and parse an additional dataset (e.g., risk-free, benchmark)."""
        if config is None:
            return None

        file_path = (
            project_root / "portfolios" / "additional_datasets" / config.file_name
        )
        if not file_path.is_file():
            logger.error(f"Additional dataset file not found at: {file_path}")
            return None

        logger.info(f"Loading additional dataset from: {file_path}")
        try:
            df = pl.read_csv(file_path)

            # Check if the date column was inferred as a number (e.g., 20220131)
            # If so, cast it to a string before attempting to parse it as a date.
            if df[config.date_column].dtype.is_numeric():
                df = df.with_columns(pl.col(config.date_column).cast(pl.Utf8))

            # Now, safely parse the date string
            df = df.with_columns(
                pl.col(config.date_column).str.to_date(format=config.date_format)
            )

            return df
        except Exception as e:
            logger.error(f"Failed to load or parse additional dataset {file_path}: {e}")
            return None

    def _load_and_merge_data(self, project_root: Path) -> Dict[str, pl.DataFrame]:
        """
        Loads prediction files and merges them with all required additional datasets.
        This is the core of the new decoupled data loading strategy.
        """
        input_conf = self.config.input_data
        predictions_dir = project_root / "models" / "predictions"

        # Load all necessary additional datasets once
        risk_free_conf = input_conf.risk_free_dataset
        risk_free_df = self._load_additional_dataset(project_root, risk_free_conf)
        if risk_free_df is None:
            raise FileNotFoundError("Risk-free rate dataset could not be loaded.")
        risk_free_df = risk_free_df.rename(
            {
                risk_free_conf.date_column: "date",
                risk_free_conf.return_column: "risk_free_rate",
            }
        ).select(["date", "risk_free_rate"])

        value_weights_df = None
        if input_conf.company_value_weights:
            vw_conf = input_conf.company_value_weights
            value_weights_df = self._load_additional_dataset(project_root, vw_conf)
            if value_weights_df is not None:
                value_weights_df = value_weights_df.rename(
                    {
                        vw_conf.date_column: "date",
                        vw_conf.id_column: "permno",
                        vw_conf.value_weight_col: "value_weight",
                    }
                ).select(["date", "permno", "value_weight"])

        market_benchmark_df = None
        if input_conf.market_benchmark:
            bm_conf = input_conf.market_benchmark
            market_benchmark_df = self._load_additional_dataset(project_root, bm_conf)
            if market_benchmark_df is not None:
                market_benchmark_df = market_benchmark_df.rename(
                    {bm_conf.date_column: "date", bm_conf.return_column: "index_ret"}
                ).select(["date", "index_ret"])

        # --- Merge all data sources for each model's predictions ---
        merged_data_for_models = {}
        for model_name in input_conf.prediction_model_names:
            pred_file = predictions_dir / f"{model_name}_predictions.parquet"
            if not pred_file.exists():
                logger.warning(
                    f"Prediction file for model '{model_name}' not found. Skipping."
                )
                continue

            # Start with the prediction data
            preds_df = pl.read_parquet(pred_file)

            # Join with risk-free data (time-series join)
            final_df = preds_df.join(risk_free_df, on="date", how="left")

            # Join with value-weighting data if it exists (panel join)
            if value_weights_df is not None:
                final_df = final_df.join(
                    value_weights_df, on=["date", "permno"], how="left"
                )

            # Join with market benchmark data if it exists (time-series join)
            if market_benchmark_df is not None:
                final_df = final_df.join(market_benchmark_df, on="date", how="left")

            merged_data_for_models[model_name] = final_df
            logger.info(
                f"Successfully loaded and merged all data for model '{model_name}'."
            )

        return merged_data_for_models

    def _calculate_cross_sectional_returns(
        self, data: pl.DataFrame
    ) -> Tuple[pl.DataFrame, bool]:
        """
        Calculates monthly returns for 10 deciles of assets based on predicted returns.
        """
        date_col = self.config.input_data.precomputed_prediction_files.date_column
        is_descending = False

        monthly_decile_returns = (
            data.drop_nulls(subset=["predicted_ret", "actual_ret"])
            .with_columns(
                pl.col("predicted_ret")
                .qcut(
                    10,
                    labels=[f"Decile {i + 1}" for i in range(10)],
                    allow_duplicates=True,
                )
                .over(date_col)
                .alias("decile")
            )
            .group_by([date_col, "decile"], maintain_order=True)
            .agg(pl.col("actual_ret").mean().alias("return"))
            .sort(date_col, "decile")
        )

        if monthly_decile_returns.is_empty():
            logger.warning("Could not calculate any cross-sectional decile returns.")
            return pl.DataFrame(), is_descending

        cumulative_decile_returns = (
            monthly_decile_returns.group_by("decile", maintain_order=True)
            .agg(
                pl.col(date_col),
                ((1 + pl.col("return")).cum_prod() - 1).alias("cumulative_return"),
            )
            .explode(["date", "cumulative_return"])
            .sort(date_col)
        )

        return cumulative_decile_returns, is_descending

    def _calculate_monthly_returns(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates portfolio returns for each month based on all strategies.
        """
        all_monthly_returns = []
        pred_conf = self.config.input_data.precomputed_prediction_files
        date_col = pred_conf.date_column
        id_col = pred_conf.id_column

        for (date,), monthly_data in data.group_by(date_col, maintain_order=True):
            monthly_data = monthly_data.drop_nulls(
                subset=["predicted_ret", "actual_ret"]
            )
            if monthly_data.is_empty():
                continue

            n_unique_preds = monthly_data["predicted_ret"].n_unique()
            if n_unique_preds == 1:
                logger.warning(
                    f"For date {date}, all {monthly_data.height} stocks have the same predicted return. "
                    "Portfolio construction for this month will be based on arbitrary rankings and may not be meaningful."
                )

            for strat in self.config.portfolio_strategies:
                monthly_data_for_strat = monthly_data
                if strat.weighting_scheme == "value":
                    # Use the standardized internal column name 'value_weight' which was set during data loading
                    value_weight_col_name = "value_weight"
                    if value_weight_col_name not in monthly_data.columns:
                        logger.error(
                            f"Value weighting requested, but the column '{value_weight_col_name}' was not found in the merged data. "
                            "Please check the 'company_values_weights' configuration. Skipping strategy."
                        )
                        continue
                    monthly_data_for_strat = monthly_data.filter(
                        pl.col(value_weight_col_name) > 0
                    )

                n = monthly_data_for_strat.height
                if n == 0:
                    continue

                ranked_data = monthly_data_for_strat.with_columns(
                    (
                        pl.col("predicted_ret").rank("ordinal", descending=False) / n
                    ).alias("pred_rank_pct")
                )

                long_portfolio = ranked_data.filter(
                    pl.col("pred_rank_pct").is_between(
                        strat.long_quantiles[0], strat.long_quantiles[1], closed="right"
                    )
                ).drop_nulls(subset=["actual_ret"])

                short_portfolio = ranked_data.filter(
                    pl.col("pred_rank_pct").is_between(
                        strat.short_quantiles[0],
                        strat.short_quantiles[1],
                        closed="right",
                    )
                ).drop_nulls(subset=["actual_ret"])

                if long_portfolio.is_empty() or short_portfolio.is_empty():
                    continue

                long_permnos = long_portfolio.get_column(id_col).to_list()
                short_permnos = short_portfolio.get_column(id_col).to_list()

                long_return: Union[float, int, None] = None
                short_return: Union[float, int, None] = None

                if strat.weighting_scheme == "equal":
                    mean_long = long_portfolio["actual_ret"].mean()
                    mean_short = short_portfolio["actual_ret"].mean()
                    if isinstance(mean_long, (float, int)) and isinstance(
                        mean_short, (float, int)
                    ):
                        long_return, short_return = mean_long, mean_short
                elif strat.weighting_scheme == "value":
                    long_sum, short_sum = (
                        long_portfolio[value_weight_col_name].sum(),
                        short_portfolio[value_weight_col_name].sum(),
                    )
                    if (
                        isinstance(long_sum, (int, float))
                        and long_sum > 0
                        and isinstance(short_sum, (int, float))
                        and short_sum > 0
                    ):
                        value_long_weights = (
                            long_portfolio[value_weight_col_name] / long_sum
                        )
                        value_short_weights = (
                            short_portfolio[value_weight_col_name] / short_sum
                        )
                        long_return = (
                            long_portfolio["actual_ret"] * value_long_weights
                        ).sum()
                        short_return = (
                            short_portfolio["actual_ret"] * value_short_weights
                        ).sum()

                if not isinstance(long_return, (float, int)) or not isinstance(
                    short_return, (float, int)
                ):
                    logger.warning(
                        f"Could not calculate valid long/short returns for date {date} and strategy {strat.name}. Skipping."
                    )
                    continue

                portfolio_return = long_return - short_return
                risk_free_rate = monthly_data["risk_free_rate"].first()
                if not isinstance(risk_free_rate, (float, int)):
                    logger.warning(
                        f"Could not find a valid risk-free rate for date {date}. Skipping."
                    )
                    continue

                all_monthly_returns.append(
                    {
                        "date": date,
                        "strategy": strat.name,
                        "long_return": long_return,
                        "short_return": short_return,
                        "portfolio_return": portfolio_return,
                        "risk_free_rate": risk_free_rate,
                        "long_leg_permnos": long_permnos,
                        "short_leg_permnos": short_permnos,
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

        logger.info("--- Loading and Merging Portfolio Data ---")
        all_model_data = self._load_and_merge_data(project_root)

        for model_name, data in all_model_data.items():
            if self.config.cross_sectional_analysis:
                logger.info(
                    f"--- Performing Cross-Sectional Analysis for Model: {model_name} ---"
                )
                cross_sectional_df, is_descending = (
                    self._calculate_cross_sectional_returns(data)
                )

                if not cross_sectional_df.is_empty() and self.reporter:
                    self.reporter.plot_cross_sectional_returns(
                        model_name, cross_sectional_df, descending_sort=is_descending
                    )

            logger.info(f"--- Processing Portfolios for Model: {model_name} ---")
            monthly_returns_df = self._calculate_monthly_returns(data)
            if monthly_returns_df.is_empty():
                logger.warning(
                    f"No monthly returns could be calculated for model '{model_name}'."
                )
                continue

            for (strat_name_raw,), strategy_returns in monthly_returns_df.group_by(
                "strategy", maintain_order=True
            ):
                strat_name = str(strat_name_raw)
                logger.info(f"Evaluating strategy: {strat_name}")
                strategy_returns = strategy_returns.sort("date")

                # The join with benchmark data is now handled in _load_and_merge_data
                # So we can directly use the 'index_ret' column if it exists

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
                    plot_df = strategy_returns.with_columns(
                        cumulative_long=((1 + pl.col("long_return")).cum_prod() - 1),
                        cumulative_short=((1 - pl.col("short_return")).cum_prod() - 1),
                        cumulative_portfolio=(
                            (1 + pl.col("portfolio_return")).cum_prod() - 1
                        ),
                        cumulative_risk_free=(
                            (1 + pl.col("risk_free_rate") / 12).cum_prod() - 1
                        ),
                    )

                    if "index_ret" in data.columns:
                        plot_df = plot_df.join(
                            data.select(["date", "index_ret"]).unique(subset="date"),
                            on="date",
                            how="left",
                        ).with_columns(
                            cumulative_index=((1 + pl.col("index_ret")).cum_prod() - 1)
                        )

                    if not plot_df.is_empty():
                        summary_metrics["final_cumulative_return"] = plot_df[
                            "cumulative_portfolio"
                        ][-1]

                    if self.reporter:
                        index_name = (
                            self.config.input_data.market_benchmark.name
                            if self.config.input_data.market_benchmark
                            else None
                        )
                        self.reporter.plot_cumulative_returns(
                            model_name, strat_name, plot_df, index_name=index_name
                        )

                if self.reporter:
                    self.reporter.generate_report(
                        model_name, strat_name, summary_metrics
                    )

        logger.info("Portfolio evaluation completed successfully.")
