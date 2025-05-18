from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """
    Generates and saves evaluation reports for models.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, model_name: str, metrics: Dict[str, Any]) -> None:
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
                f.write(f"  {key}: {value:.4f}\n")  # Format floats for readability
            f.write("\n--- End of Report ---\n")
        logger.info(f"Evaluation report saved to: {report_filename}")

    # Future: Add methods for generating plots, HTML reports, etc.
