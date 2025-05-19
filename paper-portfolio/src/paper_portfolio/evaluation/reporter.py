from pathlib import Path
import logging
import polars as pl
import matplotlib.pyplot as plt
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)


class PortfolioReporter:
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self, model_name: str, strategy_name: str, metrics: Dict[str, Any]
    ):
        """Generates a text-based report for a given portfolio strategy."""
        report_filename = self.output_dir / f"{model_name}_{strategy_name}_report.txt"
        with open(report_filename, "w") as f:
            f.write(f"--- Portfolio Performance Report ---\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Strategy: {strategy_name}\n")
            f.write("-" * 30 + "\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
            f.write("-" * 30 + "\n")
        logger.info(f"Portfolio report saved to: {report_filename}")

    def save_monthly_returns(
        self, model_name: str, strategy_name: str, returns_df: pl.DataFrame
    ):
        """Saves the detailed monthly returns to a Parquet file."""
        if returns_df.is_empty():
            return
        output_filename = (
            self.output_dir / f"{model_name}_{strategy_name}_monthly_returns.parquet"
        )
        returns_df.write_parquet(output_filename)
        logger.info(f"Monthly returns saved to: {output_filename}")

    def plot_cumulative_returns(
        self, model_name: str, strategy_name: str, returns_df: pl.DataFrame
    ):
        """Plots and saves the cumulative return chart."""
        if returns_df.is_empty() or "cumulative_return" not in returns_df.columns:
            return

        plt.figure(figsize=(12, 7))
        plt.plot(
            returns_df["date"],
            returns_df["cumulative_return"],
            label="Cumulative Return",
        )
        plt.title(f"Cumulative Return for {model_name} - {strategy_name}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plot_filename = (
            self.output_dir / f"{model_name}_{strategy_name}_cumulative_return.png"
        )
        plt.savefig(plot_filename)
        plt.close()
        logger.info(f"Cumulative return plot saved to: {plot_filename}")
