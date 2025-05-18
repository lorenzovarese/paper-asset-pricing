from pathlib import Path
from typing import Any, Dict, List
import logging
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """
    Generates and saves evaluation reports for models.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_text_report(self, model_name: str, metrics: Dict[str, Any]) -> None:
        """
        Generates a text-based report for a given model's evaluation metrics.

        Args:
            model_name: The name of the model.
            metrics: A dictionary of evaluation metrics.
        """
        report_filename = self.output_dir / f"{model_name}_evaluation_report.txt"
        with open(report_filename, "w") as f:
            f.write(f"--- Model Evaluation Report: {model_name} ---\n")
            f.write(
                f"Generated on: {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', [], None))}\n\n"
            )
            f.write("Metrics:\n")
            for key, value in metrics.items():
                # Handle both single values and lists/arrays of values
                if isinstance(value, (float, int)):
                    f.write(f"  {key}: {value:.4f}\n")
                elif isinstance(value, (list, np.ndarray)):
                    f.write(f"  {key}: {np.mean(value):.4f} (Avg across windows)\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n--- End of Report ---\n")
        logger.info(f"Evaluation text report saved to: {report_filename}")

    def save_metrics_to_parquet(
        self, model_name: str, metrics_data: List[Dict[str, Any]]
    ) -> None:
        """
        Saves detailed evaluation metrics (e.g., per rolling window) to a Parquet file.

        Args:
            model_name: The name of the model.
            metrics_data: A list of dictionaries, where each dictionary contains metrics
                          for a specific evaluation window/step.
        """
        if not metrics_data:
            logger.warning(f"No metrics data to save for model '{model_name}'.")
            return

        df = pl.DataFrame(metrics_data)
        output_filename = self.output_dir / f"{model_name}_evaluation_metrics.parquet"
        df.write_parquet(output_filename)
        logger.info(f"Detailed evaluation metrics saved to: {output_filename}")
