import polars as pl
import logging
from pathlib import Path
from typing import Dict, Any, List, Union
import joblib  # type: ignore

from paper_model.config_parser import ModelsConfig
from paper_model.models.base import BaseModel
from paper_model.models.sklearn_model import SklearnModel
from paper_model.models.torch_model import TorchModel
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
        "glm": SklearnModel,
        "rf": SklearnModel,
        "gbrt": SklearnModel,
        "nn": TorchModel,
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
            files = sorted(
                list(processed_dir.glob(f"{input_conf.dataset_name}_*.parquet"))
            )
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

            # Pass the Pydantic model object itself as the config
            model_config_dict = model_config.model_dump()
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

        unique_months = (
            data.get_column(date_col).dt.truncate("1mo").unique().sort().to_list()
        )

        total_window_months = (
            eval_config.train_month
            + eval_config.validation_month
            + eval_config.testing_month
        )
        if len(unique_months) < total_window_months:
            logger.error(
                f"Not enough data for the rolling window configuration. "
                f"Need {total_window_months} months, but have {len(unique_months)}."
            )
            return

        logger.info(
            f"Starting rolling window evaluation with {len(unique_months)} total months."
        )

        window_start_idx = 0
        while True:
            train_end_idx = window_start_idx + eval_config.train_month
            validation_end_idx = train_end_idx + eval_config.validation_month
            test_end_idx = validation_end_idx + eval_config.testing_month

            if test_end_idx > len(unique_months):
                logger.info("Reached the end of the dataset. Stopping evaluation.")
                break

            # Define date boundaries for clarity
            train_start_date = unique_months[window_start_idx]
            train_end_date = unique_months[train_end_idx - 1]
            validation_start_date = unique_months[train_end_idx]
            validation_end_date = unique_months[validation_end_idx - 1]
            test_start_date = unique_months[validation_end_idx]
            test_end_date = unique_months[test_end_idx - 1]

            logger.info(
                f"Processing window: Train from {train_start_date.strftime('%Y-%m')} to "
                f"{train_end_date.strftime('%Y-%m')}"
            )
            if eval_config.validation_month > 0:
                logger.info(
                    f"  Validation from {validation_start_date.strftime('%Y-%m')} to "
                    f"{validation_end_date.strftime('%Y-%m')}"
                )
            logger.info(
                f"  Test from {test_start_date.strftime('%Y-%m')} to "
                f"{test_end_date.strftime('%Y-%m')}"
            )

            # --- Slicing Data for the Current Window ---
            train_data = data.filter(
                pl.col(date_col).is_between(train_start_date, train_end_date)
            )

            validation_data = None
            if eval_config.validation_month > 0:
                validation_data = data.filter(
                    pl.col(date_col).is_between(
                        validation_start_date, validation_end_date
                    )
                )

            for model_name, model_instance in self.models.items():
                try:
                    if train_data.is_empty():
                        logger.warning(
                            f"Skipping model '{model_name}' due to empty training data."
                        )
                        continue

                    model_instance.train(train_data, validation_data)

                    # Month-by-month evaluation on the test set
                    for test_month_idx in range(validation_end_idx, test_end_idx):
                        current_month = unique_months[test_month_idx]
                        monthly_test_data = data.filter(
                            pl.col(date_col).dt.truncate("1mo") == current_month
                        )

                        if monthly_test_data.is_empty():
                            continue

                        target_col = model_instance.config["target_column"]
                        y_true_month = monthly_test_data.get_column(target_col)
                        y_pred_month = model_instance.predict(monthly_test_data)

                        aligned_df = pl.DataFrame(
                            {"y_true": y_true_month, "y_pred": y_pred_month}
                        ).drop_nulls()

                        if aligned_df.is_empty():
                            continue

                        y_true_np = aligned_df["y_true"].to_numpy()
                        y_pred_np = aligned_df["y_pred"].to_numpy()
                        n_predictors = len(model_instance.config["feature_columns"])

                        monthly_metrics = {
                            "evaluation_date": current_month.strftime("%Y-%m-%d")
                        }
                        for metric_name in eval_config.metrics:
                            metric_func = self.METRIC_FUNCTIONS.get(metric_name)
                            if metric_func:
                                if metric_name == "r2_adj_oos":
                                    monthly_metrics[metric_name] = metric_func(
                                        y_true_np, y_pred_np, n_predictors
                                    )
                                else:
                                    monthly_metrics[metric_name] = metric_func(
                                        y_true_np, y_pred_np
                                    )

                        self.all_evaluation_results[model_name].append(monthly_metrics)

                        if model_instance.config.get("save_prediction_results"):
                            preds_to_save = monthly_test_data.select(
                                [date_col, id_col]
                            ).with_columns(
                                predicted_ret=y_pred_month, actual_ret=y_true_month
                            )
                            self.all_prediction_results[model_name] = pl.concat(
                                [self.all_prediction_results[model_name], preds_to_save]
                            )

                    if model_instance.config.get("save_model_checkpoints"):
                        self._save_checkpoint(model_instance, train_end_date)

                except Exception as e:
                    logger.error(
                        f"Error running model '{model_name}' in window ending {train_end_date.strftime('%Y-%m')}: {e}",
                        exc_info=True,
                    )

            window_start_idx += eval_config.step_month

    # ... (the rest of the ModelManager class remains the same) ...
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
                agg_metrics = {}
                df = pl.DataFrame(results)
                for metric in self.config.evaluation.metrics:
                    if metric in df.columns:
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
