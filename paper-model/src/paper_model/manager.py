import numpy as np
import polars as pl
import logging
from pathlib import Path
from typing import Dict, Any, List, Union
import joblib  # type: ignore

from paper_model.config_parser import ModelsConfig
from paper_model.models.base import BaseModel
from paper_model.models.sklearn_model import SklearnModel
from paper_model.evaluation.reporter import EvaluationReporter
from paper_model.evaluation.metrics import (
    mean_squared_error,
    r2_out_of_sample,
    r2_adj_out_of_sample,
)

logger = logging.getLogger(__name__)


class ModelManager:
    MODEL_REGISTRY: Dict[str, type[BaseModel]] = {
        "ols": SklearnModel,
        "enet": SklearnModel,
        "pcr": SklearnModel,
        "pls": SklearnModel,
    }

    METRIC_FUNCTIONS: Dict[str, Any] = {
        "mse": mean_squared_error,
        "r2_oos": r2_out_of_sample,
        "r2_adj_oos": r2_adj_out_of_sample,
    }

    def __init__(self, config: ModelsConfig):
        self.config = config
        self.models: Dict[str, BaseModel] = {}
        self.all_evaluation_results: Dict[str, List[Dict[str, Any]]] = {}
        self.all_prediction_results: Dict[str, pl.DataFrame] = {}
        self._project_root: Path | None = None

    def _load_processed_data(self) -> pl.DataFrame:
        if self._project_root is None:
            raise ValueError("Project root must be set before loading data.")

        input_conf = self.config.input_data
        processed_dir = self._project_root / "data" / "processed"

        if input_conf.splitted == "year":
            files = list(processed_dir.glob(f"{input_conf.dataset_name}_*.parquet"))
            if not files:
                raise FileNotFoundError(
                    f"No data files found for {input_conf.dataset_name} in {processed_dir}"
                )
            df = pl.concat([pl.read_parquet(f) for f in files])
        else:
            file_path = processed_dir / f"{input_conf.dataset_name}.parquet"
            if not file_path.is_file():
                raise FileNotFoundError(f"Data file {file_path} not found.")
            df = pl.read_parquet(file_path)

        if df[input_conf.date_column].dtype != pl.Date:
            df = df.with_columns(pl.col(input_conf.date_column).cast(pl.Date))

        return df.sort(input_conf.date_column, input_conf.id_column)

    def _initialize_models(self) -> None:
        for model_config in self.config.models:
            model_name = model_config.name
            model_type = model_config.type

            if model_type not in self.MODEL_REGISTRY:
                raise ValueError(f"Unknown model type: '{model_type}'")

            model_class = self.MODEL_REGISTRY[model_type]

            # Pass the full model config dictionary
            model_config_dict = model_config.dict()
            model_config_dict["date_column"] = self.config.input_data.date_column
            model_config_dict["id_column"] = self.config.input_data.id_column

            self.models[model_name] = model_class(model_name, model_config_dict)
            self.all_evaluation_results[model_name] = []
            self.all_prediction_results[model_name] = pl.DataFrame()
            logger.info(f"Initialized model: '{model_name}' of type '{model_type}'.")

    def _run_rolling_window_evaluation(self, data: pl.DataFrame) -> None:
        eval_config = self.config.evaluation
        date_col = self.config.input_data.date_column
        id_col = self.config.input_data.id_column

        unique_dates = data.get_column(date_col).unique().sort()
        unique_months = unique_dates.dt.truncate("1mo").unique().sort().to_list()

        if len(unique_months) < eval_config.train_month + eval_config.testing_month:
            logger.error("Not enough data for the rolling window configuration.")
            return

        logger.info(f"Starting rolling window evaluation: {len(unique_months)} months.")

        window_start_idx = 0
        while True:
            train_end_idx = window_start_idx + eval_config.train_month
            test_end_idx = train_end_idx + eval_config.testing_month

            if test_end_idx > len(unique_months):
                break

            train_end_date = unique_months[train_end_idx - 1]
            test_end_date = unique_months[test_end_idx - 1]

            train_data = data.filter(pl.col(date_col) <= train_end_date)
            test_data = data.filter(
                (pl.col(date_col) > train_end_date)
                & (pl.col(date_col) <= test_end_date)
            )

            if train_data.is_empty() or test_data.is_empty():
                window_start_idx += eval_config.step_month
                continue

            logger.info(f"Processing window ending {test_end_date.strftime('%Y-%m')}")

            for model_name, model_instance in self.models.items():
                try:
                    model_instance.train(train_data)

                    target_col = model_instance.config["target_column"]
                    y_true_test = test_data.get_column(target_col)
                    y_pred_test = model_instance.predict(test_data)

                    # Align predictions and true values
                    aligned_df = pl.DataFrame(
                        {"y_true": y_true_test, "y_pred": y_pred_test}
                    ).drop_nulls()

                    window_metrics = {}
                    if not aligned_df.is_empty():
                        y_true_np = aligned_df["y_true"].to_numpy()
                        y_pred_np = aligned_df["y_pred"].to_numpy()
                        n_predictors = len(model_instance.config["feature_columns"])

                        for metric_name in eval_config.metrics:
                            if metric_name in self.METRIC_FUNCTIONS:
                                if metric_name == "r2_adj_oos":
                                    window_metrics[metric_name] = self.METRIC_FUNCTIONS[
                                        metric_name
                                    ](y_true_np, y_pred_np, n_predictors)
                                else:
                                    window_metrics[metric_name] = self.METRIC_FUNCTIONS[
                                        metric_name
                                    ](y_true_np, y_pred_np)

                    window_metrics["window_end_date"] = test_end_date.strftime(
                        "%Y-%m-%d"
                    )
                    self.all_evaluation_results[model_name].append(window_metrics)

                    if model_instance.config.get("save_prediction_results"):
                        preds_to_save = test_data.select(
                            [date_col, id_col]
                        ).with_columns(
                            predicted_ret=y_pred_test, actual_ret=y_true_test
                        )
                        self.all_prediction_results[model_name] = pl.concat(
                            [self.all_prediction_results[model_name], preds_to_save]
                        )

                    if model_instance.config.get("save_model_checkpoints"):
                        self._save_checkpoint(model_instance, test_end_date)

                except Exception as e:
                    logger.error(
                        f"Error running model '{model_name}': {e}", exc_info=True
                    )

            window_start_idx += eval_config.step_month

    def _save_checkpoint(self, model_instance: BaseModel, date: Any) -> None:
        if self._project_root is None:
            return
        checkpoint_dir = self._project_root / "models" / "saved"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            checkpoint_dir
            / f"{model_instance.name}_checkpoint_{date.strftime('%Y%m%d')}.joblib"
        )
        joblib.dump(model_instance.model, filename)
        logger.info(f"Saved checkpoint for '{model_instance.name}' to {filename}")

    def _export_results(self) -> None:
        if self._project_root is None:
            return

        eval_dir = self._project_root / "models" / "evaluations"
        pred_dir = self._project_root / "models" / "predictions"
        eval_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        reporter = EvaluationReporter(eval_dir)

        for name, results in self.all_evaluation_results.items():
            if results:
                reporter.save_metrics_to_parquet(name, results)
                # Aggregate and generate text report
                agg_metrics = {}
                df = pl.DataFrame(results)
                for metric in self.config.evaluation.metrics:
                    agg_metrics[f"avg_{metric}"] = df.get_column(metric).mean()
                    agg_metrics[f"std_{metric}"] = df.get_column(metric).std()
                reporter.generate_text_report(name, agg_metrics)

        for name, preds in self.all_prediction_results.items():
            if not preds.is_empty():
                preds.write_parquet(pred_dir / f"{name}_predictions.parquet")

    def run(self, project_root: Union[str, Path]) -> Dict[str, pl.DataFrame]:
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
