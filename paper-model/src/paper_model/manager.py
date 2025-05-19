import numpy as np
import polars as pl
import logging
from pathlib import Path
from typing import Dict, Any, List, cast
import joblib  # type: ignore[import-untyped]
import pickle
from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]
from statsmodels.regression.linear_model import RegressionResultsWrapper  # type: ignore[import-untyped]


from paper_model.config_parser import (
    ModelsConfig,
)

from paper_model.models.base import BaseModel
from paper_model.models.fama_french import FamaFrench3FactorModel
from paper_model.models.linear_regression import SimpleLinearRegression
from paper_model.evaluation.reporter import EvaluationReporter
from paper_model.evaluation.metrics import (
    mean_squared_error,
    r2_out_of_sample,
    r2_adj_out_of_sample,
)

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model training, evaluation, and checkpoint generation based on a YAML configuration.
    Supports rolling window evaluation.
    """

    MODEL_REGISTRY: Dict[str, type[BaseModel]] = {
        "fama_french_3_factor": FamaFrench3FactorModel,
        "linear_regression": SimpleLinearRegression,
    }

    METRIC_FUNCTIONS: Dict[str, Any] = {
        "mse": mean_squared_error,
        "r2_oos": r2_out_of_sample,
        "r2_adj_oos": r2_adj_out_of_sample,
    }

    def __init__(self, config: ModelsConfig):
        """
        Initializes the ModelManager with a parsed models configuration dictionary.

        Args:
            config: The parsed models configuration dictionary (a ModelsConfig object).
        """
        self.config = config
        self.models: Dict[str, BaseModel] = {}
        self.all_evaluation_results: Dict[str, List[Dict[str, Any]]] = {}
        self.all_prediction_results: Dict[str, pl.DataFrame] = {}
        self._project_root: Path | None = None

    def _load_processed_data(self) -> pl.DataFrame:
        """
        Loads a processed dataset from the project's data/processed directory,
        handling 'splitted' configuration.
        """
        if self._project_root is None:
            raise ValueError("Project root must be set before loading data.")

        input_data_config = self.config.input_data
        dataset_name = input_data_config.dataset_name
        splitted_by = input_data_config.splitted.value
        date_column = input_data_config.date_column

        processed_data_dir = self._project_root / "data" / "processed"

        df: pl.DataFrame
        if splitted_by == "year":
            data_files = list(processed_data_dir.glob(f"{dataset_name}_*.parquet"))
            if not data_files:
                raise FileNotFoundError(
                    f"No processed data files found for dataset '{dataset_name}' "
                    f"partitioned by year in '{processed_data_dir}'. Ensure paper-data has run successfully."
                )
            logger.info(
                f"Loading processed data for '{dataset_name}' from {len(data_files)} files (splitted by year)..."
            )
            df = pl.concat([pl.read_parquet(f) for f in data_files])
        elif splitted_by == "none":
            filename = f"{dataset_name}.parquet"
            data_path = processed_data_dir / filename
            if not data_path.is_file():
                raise FileNotFoundError(
                    f"Processed data file '{filename}' not found in '{processed_data_dir}'. "
                    f"Ensure paper-data has run successfully and is not partitioned."
                )
            logger.info(
                f"Loading processed data for '{dataset_name}' from '{data_path}' (not splitted)..."
            )
            df = pl.read_parquet(data_path)
        else:
            raise ValueError(
                f"Unsupported 'splitted' configuration: {splitted_by}. Must be 'year' or 'none'."
            )

        if df[date_column].dtype != pl.Date:
            df = df.with_columns(pl.col(date_column).cast(pl.Date).alias(date_column))

        df = df.sort(date_column, input_data_config.id_column)

        logger.info(f"Loaded data for '{dataset_name}'. Shape: {df.shape}")
        return df

    def _initialize_models(self) -> None:
        """
        Initializes model instances based on the 'models' section of the config.
        """
        for model_config_pydantic in self.config.models:
            model_name = model_config_pydantic.name
            model_type = model_config_pydantic.type

            if model_type not in self.MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown model type: '{model_type}'. "
                    f"Available types: {list(self.MODEL_REGISTRY.keys())}"
                )

            model_class = self.MODEL_REGISTRY[model_type]

            model_config_dict = model_config_pydantic.dict()

            model_config_dict["date_column"] = self.config.input_data.date_column
            model_config_dict["id_column"] = self.config.input_data.id_column
            model_config_dict["risk_free_rate_col"] = (
                self.config.input_data.risk_free_rate_col
            )

            self.models[model_name] = model_class(model_name, model_config_dict)
            self.all_evaluation_results[model_name] = []
            self.all_prediction_results[model_name] = pl.DataFrame()
            logger.info(f"Initialized model: '{model_name}' of type '{model_type}'.")

    def _run_rolling_window_evaluation(self, data: pl.DataFrame) -> None:
        """
        Executes rolling window evaluation for all initialized models.
        """
        eval_config = self.config.evaluation
        implementation = eval_config.implementation.value

        if implementation != "rolling window":
            raise NotImplementedError(
                f"Evaluation implementation '{implementation}' not supported. Only 'rolling window' is implemented."
            )

        train_months = eval_config.train_month
        testing_months = eval_config.testing_month
        step_months = eval_config.step_month
        metrics_to_compute = eval_config.metrics

        if not all([train_months, testing_months, step_months]):
            raise ValueError(
                "Rolling window evaluation requires 'train_month', 'testing_month', and 'step_month' to be specified."
            )
        if not metrics_to_compute:
            logger.warning(
                "No metrics specified for evaluation. Skipping metric calculation."
            )

        date_col = self.config.input_data.date_column
        id_col = self.config.input_data.id_column
        target_col_map = {
            model_name: (
                model_instance.config.get("target_return_col")
                or model_instance.config.get("target_column")
                or "return"
            )
            for model_name, model_instance in self.models.items()
        }

        unique_dates = data.select(pl.col(date_col)).unique().sort(date_col).to_series()
        unique_months = unique_dates.dt.truncate("1mo").unique().sort().to_list()

        if len(unique_months) < train_months + testing_months:
            logger.error(
                f"Not enough unique months ({len(unique_months)}) for the specified rolling window configuration "
                f"(train: {train_months}, test: {testing_months}). Minimum required: {train_months + testing_months}."
            )
            return

        logger.info(
            f"Starting rolling window evaluation with {len(unique_months)} unique months."
        )
        logger.info(
            f"Train window: {train_months} months, Test window: {testing_months} months, Step: {step_months} months."
        )

        window_start_idx = 0
        while True:
            train_end_idx = window_start_idx + train_months
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + testing_months

            if test_end_idx > len(unique_months):
                logger.info("Reached end of data for rolling window. Stopping.")
                break

            train_end_date = unique_months[train_end_idx - 1]
            test_end_date = unique_months[test_end_idx - 1]

            train_data = data.filter(pl.col(date_col) <= train_end_date)
            test_data = data.filter(
                (pl.col(date_col) > train_end_date)
                & (pl.col(date_col) <= test_end_date)
            )

            if train_data.is_empty() or test_data.is_empty():
                logger.warning(
                    f"Skipping window ending {test_end_date.strftime('%Y-%m')}: Empty train or test data."
                )
                window_start_idx += step_months
                continue

            logger.info(
                f"Processing window: Train up to {train_end_date.strftime('%Y-%m')}, "
                f"Test from {unique_months[test_start_idx].strftime('%Y-%m')} to {test_end_date.strftime('%Y-%m')}"
            )

            for model_name, model_instance in self.models.items():
                logger.info(f"--- Running Model '{model_name}' for current window ---")
                try:
                    model_instance.train(train_data)

                    y_true_test = test_data.select(
                        target_col_map[model_name]
                    ).to_series()
                    y_pred_test = model_instance.predict(test_data)

                    test_identifiers = test_data.select([date_col, id_col])
                    prediction_df_aligned = pl.DataFrame(
                        {
                            date_col: test_identifiers.get_column(date_col),
                            id_col: test_identifiers.get_column(id_col),
                            "y_true": y_true_test,
                            "y_pred": y_pred_test,
                        }
                    ).drop_nulls()

                    if prediction_df_aligned.is_empty():
                        logger.warning(
                            f"No valid true/predicted pairs for model '{model_name}' in this test window. Skipping evaluation."
                        )
                        window_metrics = {
                            metric: np.nan for metric in metrics_to_compute
                        }
                    else:
                        y_true_aligned = prediction_df_aligned["y_true"]
                        y_pred_aligned = prediction_df_aligned["y_pred"]

                        window_metrics = {}
                        for metric_name in metrics_to_compute:
                            if metric_name in self.METRIC_FUNCTIONS:
                                n_predictors = 0
                                if isinstance(model_instance, FamaFrench3FactorModel):
                                    n_predictors = len(model_instance.factor_cols)
                                elif isinstance(model_instance, SimpleLinearRegression):
                                    n_predictors = len(model_instance.feature_cols)

                                if metric_name == "r2_adj_oos":
                                    window_metrics[metric_name] = self.METRIC_FUNCTIONS[
                                        metric_name
                                    ](
                                        y_true_aligned.to_numpy(),
                                        y_pred_aligned.to_numpy(),
                                        n_predictors,
                                    )
                                else:
                                    window_metrics[metric_name] = self.METRIC_FUNCTIONS[
                                        metric_name
                                    ](
                                        y_true_aligned.to_numpy(),
                                        y_pred_aligned.to_numpy(),
                                    )
                            else:
                                logger.warning(
                                    f"Metric '{metric_name}' not recognized. Skipping."
                                )
                                window_metrics[metric_name] = np.nan

                    window_metrics["window_end_date"] = test_end_date.strftime(
                        "%Y-%m-%d"
                    )
                    self.all_evaluation_results[model_name].append(window_metrics)
                    logger.info(
                        f"Model '{model_name}' metrics for window ending {test_end_date.strftime('%Y-%m')}: {window_metrics}"
                    )

                    if model_instance.config.get("save_prediction_results", False):
                        prediction_output_df = pl.DataFrame(
                            {
                                date_col: prediction_df_aligned.get_column(date_col),
                                id_col: prediction_df_aligned.get_column(id_col),
                                "predicted_ret": prediction_df_aligned.get_column(
                                    "y_pred"
                                ),
                                "actual_ret": prediction_df_aligned.get_column(
                                    "y_true"
                                ),
                            }
                        )
                        self.all_prediction_results[model_name] = pl.concat(
                            [
                                self.all_prediction_results[model_name],
                                prediction_output_df,
                            ]
                        )
                        logger.debug(
                            f"Appended {prediction_output_df.shape[0]} prediction results for model '{model_name}'. Total: {self.all_prediction_results[model_name].shape[0]}"
                        )

                    if model_instance.config.get("save_model_checkpoints", False):
                        assert self._project_root is not None
                        checkpoint_dir = (
                            self._project_root / "models" / "saved" / model_name
                        )
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        checkpoint_filename = (
                            checkpoint_dir
                            / f"{model_name}_model_checkpoint_{test_end_date.strftime('%Y%m%d')}.pkl"
                        )

                        if isinstance(model_instance.model, LinearRegression):
                            joblib.dump(model_instance.model, checkpoint_filename)
                            logger.info(
                                f"Model checkpoint for '{model_name}' saved to: {checkpoint_filename}"
                            )
                        elif isinstance(
                            model_instance, FamaFrench3FactorModel
                        ) and isinstance(model_instance.model, dict):
                            ff_models = cast(
                                Dict[Any, RegressionResultsWrapper],
                                model_instance.model,
                            )
                            if all(
                                isinstance(m, RegressionResultsWrapper)
                                for m in ff_models.values()
                            ):
                                with open(checkpoint_filename, "wb") as f:
                                    pickle.dump(ff_models, f)
                                logger.info(
                                    f"Model checkpoint for '{model_name}' saved to: {checkpoint_filename}"
                                )
                            else:
                                logger.warning(
                                    f"Fama-French model '{model_name}' contains non-RegressionResultsWrapper objects. Skipping checkpoint."
                                )
                        else:
                            logger.warning(
                                f"Model type for '{model_name}' not supported for direct checkpointing. Skipping."
                            )

                except Exception as e:
                    logger.error(
                        f"Error running model '{model_name}' for window ending {test_end_date.strftime('%Y-%m')}: {e}",
                        exc_info=True,
                    )
                    window_metrics = {metric: np.nan for metric in metrics_to_compute}
                    window_metrics["window_end_date"] = test_end_date.strftime(
                        "%Y-%m-%d"
                    )
                    self.all_evaluation_results[model_name].append(window_metrics)

            window_start_idx += step_months

    def _export_results(self) -> None:
        """
        Exports aggregated evaluation reports and all accumulated prediction results.
        """
        if self._project_root is None:
            raise ValueError("Project root must be set before exporting results.")

        eval_output_dir = self._project_root / "models" / "evaluations"
        prediction_output_dir = self._project_root / "models" / "predictions"
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        prediction_output_dir.mkdir(parents=True, exist_ok=True)

        reporter = EvaluationReporter(eval_output_dir)

        logger.info("--- Exporting Model Results ---")

        for model_name, metrics_list in self.all_evaluation_results.items():
            if metrics_list:
                aggregated_metrics = {}
                for metric_name in self.config.evaluation.metrics:
                    valid_values = []
                    for m in metrics_list:
                        metric_value = m.get(metric_name)
                        if metric_value is not None:
                            try:
                                float_value = float(metric_value)
                                if not np.isnan(float_value):
                                    valid_values.append(float_value)
                            except (ValueError, TypeError):
                                logger.warning(
                                    f"Metric '{metric_name}' for model '{model_name}' has non-numeric value: {metric_value}. Skipping for aggregation."
                                )
                                continue

                    if valid_values:
                        np_values = np.array(valid_values, dtype=float)
                        aggregated_metrics[f"avg_{metric_name}"] = np.mean(np_values)
                        aggregated_metrics[f"std_{metric_name}"] = np.std(np_values)
                    else:
                        aggregated_metrics[f"avg_{metric_name}"] = np.nan
                        aggregated_metrics[f"std_{metric_name}"] = np.nan

                reporter.generate_text_report(model_name, aggregated_metrics)
                reporter.save_metrics_to_parquet(model_name, metrics_list)
            else:
                logger.warning(
                    f"No evaluation results to export for model '{model_name}'."
                )

        for model_name, prediction_df in self.all_prediction_results.items():
            if not prediction_df.is_empty():
                prediction_filename = (
                    prediction_output_dir / f"{model_name}_predictions.parquet"
                )
                prediction_df.write_parquet(prediction_filename)
                logger.info(
                    f"Prediction results for '{model_name}' saved to: {prediction_filename}"
                )
            else:
                logger.warning(
                    f"No prediction results generated for model '{model_name}'. Skipping export."
                )

        logger.info("Model results export completed successfully.")

    def run(self, project_root: str | Path) -> Dict[str, pl.DataFrame]:
        """
        Executes the model pipeline: initialization, training, evaluation, and export.

        Args:
            project_root: The root directory of the PAPER project (e.g., 'PAPER/ThesisExample').

        Returns:
            A dictionary of the generated model checkpoints (predictions).
        """
        self._project_root = Path(project_root).expanduser()
        logger.info(f"Running model pipeline for project: {self._project_root}")

        logger.info("--- Initializing Models ---")
        self._initialize_models()

        logger.info("--- Loading Processed Data ---")
        full_data = self._load_processed_data()

        logger.info("--- Running Rolling Window Evaluation ---")
        self._run_rolling_window_evaluation(full_data)

        logger.info("--- Exporting Results ---")
        self._export_results()

        logger.info("Model pipeline completed successfully.")
        return self.all_prediction_results
