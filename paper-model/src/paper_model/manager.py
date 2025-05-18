from pathlib import Path
import polars as pl
import logging
from typing import Dict, Any

from paper_model.config_parser import load_config
from paper_model.models.base import BaseModel
from paper_model.models.fama_french import FamaFrench3FactorModel
from paper_model.models.linear_regression import SimpleLinearRegression
from paper_model.evaluation.reporter import EvaluationReporter

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model training, evaluation, and checkpoint generation based on a YAML configuration.
    """

    MODEL_REGISTRY: Dict[str, type[BaseModel]] = {
        "fama_french_3_factor": FamaFrench3FactorModel,
        "linear_regression": SimpleLinearRegression,
        # Add other models here as they are implemented
    }

    def __init__(self, config_path: str | Path):
        """
        Initializes the ModelManager with a path to the models configuration file.

        Args:
            config_path: The path to the models configuration YAML file.
        """
        self.config = load_config(config_path)
        self.models: Dict[str, BaseModel] = {}
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
        self.checkpoints: Dict[str, pl.DataFrame] = {}
        self._project_root: Path | None = None

    def _load_processed_data(self, dataset_name: str) -> pl.DataFrame:
        """
        Loads a processed dataset from the project's data/processed directory.
        Assumes data is partitioned by year and stored as Parquet.
        """
        if self._project_root is None:
            raise ValueError("Project root must be set before loading data.")

        processed_data_dir = self._project_root / "data" / "processed"
        base_filename = dataset_name  # Assuming dataset_name is the base filename from data-config.yaml export

        # Find all parquet files matching the base filename (e.g., final_dataset_2024.parquet)
        # This assumes the export from paper-data uses a consistent naming convention.
        data_files = list(processed_data_dir.glob(f"{base_filename}*.parquet"))

        if not data_files:
            raise FileNotFoundError(
                f"No processed data files found for dataset '{dataset_name}' "
                f"in '{processed_data_dir}'. Ensure paper-data has run successfully."
            )

        logger.info(
            f"Loading processed data for '{dataset_name}' from {len(data_files)} files..."
        )
        # Read all parquet files into a single DataFrame
        df = pl.concat([pl.read_parquet(f) for f in data_files])
        logger.info(f"Loaded data for '{dataset_name}'. Shape: {df.shape}")
        return df

    def _initialize_models(self) -> None:
        """
        Initializes model instances based on the 'models' section of the config.
        """
        for model_config in self.config.get("models", []):
            model_name = model_config["name"]
            model_type = model_config["type"]

            if model_type not in self.MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown model type: '{model_type}'. "
                    f"Available types: {list(self.MODEL_REGISTRY.keys())}"
                )

            model_class = self.MODEL_REGISTRY[model_type]
            self.models[model_name] = model_class(model_name, model_config)
            logger.info(f"Initialized model: '{model_name}' of type '{model_type}'.")

    def _run_models(self) -> None:
        """
        Runs the training, evaluation, and checkpoint generation for each initialized model.
        """
        for model_name, model_instance in self.models.items():
            logger.info(f"--- Running Model: {model_name} ---")
            input_data_config = self.config["input_data"]
            dataset_to_use = input_data_config["dataset_name"]

            # Load the specific processed dataset required by the model
            data = self._load_processed_data(dataset_to_use)

            # Pass date_column and id_column from config to model instance if available
            # This allows models to know which columns to use for time-series/panel operations
            model_instance.config["date_column"] = input_data_config.get(
                "date_column", "date"
            )
            model_instance.config["id_column"] = input_data_config.get(
                "id_column", "permco"
            )
            model_instance.config["risk_free_rate_col"] = input_data_config.get(
                "risk_free_rate_col", "rf"
            )

            metrics, checkpoint = model_instance.run(data)
            self.evaluation_results[model_name] = metrics
            self.checkpoints[model_name] = checkpoint
            logger.info(f"Model '{model_name}' completed. Metrics: {metrics}")

    def _export_results(self) -> None:
        """
        Exports evaluation reports and model checkpoints.
        """
        if self._project_root is None:
            raise ValueError("Project root must be set before exporting results.")

        eval_output_dir = self._project_root / "models" / "evaluations"
        checkpoint_output_dir = self._project_root / "models" / "saved"

        eval_output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_output_dir.mkdir(parents=True, exist_ok=True)

        reporter = EvaluationReporter(eval_output_dir)

        logger.info("--- Exporting Model Results ---")
        for model_name, metrics in self.evaluation_results.items():
            reporter.generate_report(model_name, metrics)

        for model_name, checkpoint_df in self.checkpoints.items():
            if not checkpoint_df.is_empty():
                checkpoint_filename = (
                    checkpoint_output_dir / f"{model_name}_checkpoint.parquet"
                )
                checkpoint_df.write_parquet(checkpoint_filename)
                logger.info(
                    f"Checkpoint for '{model_name}' saved to: {checkpoint_filename}"
                )
            else:
                logger.warning(
                    f"No checkpoint generated for model '{model_name}'. Skipping export."
                )

        logger.info("Model results export completed successfully.")

    def run(self, project_root: str | Path) -> Dict[str, pl.DataFrame]:
        """
        Executes the model pipeline: initialization, training, evaluation, and export.

        Args:
            project_root: The root directory of the PAPER project (e.g., 'PAPER/ThesisExample').

        Returns:
            A dictionary of the generated model checkpoints.
        """
        self._project_root = Path(project_root).expanduser()
        logger.info(f"Running model pipeline for project: {self._project_root}")

        logger.info("--- Initializing Models ---")
        self._initialize_models()

        logger.info("--- Running Models ---")
        self._run_models()

        logger.info("--- Exporting Results ---")
        self._export_results()

        logger.info("Model pipeline completed successfully.")
        return self.checkpoints
