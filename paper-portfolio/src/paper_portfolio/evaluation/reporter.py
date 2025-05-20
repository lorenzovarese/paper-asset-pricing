from pathlib import Path
import logging
import polars as pl
import matplotlib.pyplot as plt
from typing import Dict, Any, Union, Optional

logger = logging.getLogger(__name__)


class PortfolioReporter:
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cross_sectional_dir = self.output_dir / "cross_sectional_analysis"
        self.cross_sectional_dir.mkdir(exist_ok=True)

    def generate_report(
        self, model_name: str, strategy_name: str, metrics: Dict[str, Any]
    ):
        """Generates a text-based report for a given portfolio strategy."""
        report_filename = self.output_dir / f"{model_name}_{strategy_name}_report.txt"
        with open(report_filename, "w") as f:
            f.write("--- Portfolio Performance Report ---\n")
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
        self,
        model_name: str,
        strategy_name: str,
        returns_df: pl.DataFrame,
        index_name: Optional[str] = None,
    ):
        """
        Plots and saves the cumulative return chart for the long, short,
        and combined long-short portfolio, including optional benchmarks.
        """
        required_cols = [
            "cumulative_long",
            "cumulative_short",
            "cumulative_portfolio",
            "cumulative_risk_free",
        ]

        if returns_df.is_empty() or not all(
            c in returns_df.columns for c in required_cols
        ):
            logger.warning(
                f"Missing required columns for plotting in strategy '{strategy_name}'. Skipping plot."
            )
            return

        plt.figure(figsize=(12, 7))

        # Plot standard components
        plt.plot(
            returns_df["date"],
            returns_df["cumulative_long"],
            label="Long Component",
            color="green",
            linewidth=1.5,
        )
        plt.plot(
            returns_df["date"],
            returns_df["cumulative_short"],
            label="Short Component",
            color="red",
            linewidth=1.5,
        )
        plt.plot(
            returns_df["date"],
            returns_df["cumulative_portfolio"],
            label="Long-Short Strategy",
            color="blue",
            linewidth=2.5,
        )
        plt.plot(
            returns_df["date"],
            returns_df["cumulative_risk_free"],
            label="Risk-Free Benchmark",
            color="gray",
            linestyle="--",
            linewidth=1.5,
        )

        # Conditionally plot the market index benchmark if its data is present
        if "cumulative_index" in returns_df.columns and index_name:
            # Filter out any rows where index data might be missing after the join
            plot_data = returns_df.filter(pl.col("cumulative_index").is_not_null())
            if not plot_data.is_empty():
                plt.plot(
                    plot_data["date"],
                    plot_data["cumulative_index"],
                    label=f"{index_name} Benchmark",
                    color="black",
                    linestyle="--",
                    linewidth=2.0,
                )

        plt.title(f"Cumulative Return for {model_name} - {strategy_name}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()

        plot_filename = (
            self.output_dir / f"{model_name}_{strategy_name}_cumulative_return.png"
        )
        plt.savefig(plot_filename)
        plt.close()
        logger.info(f"Cumulative return plot saved to: {plot_filename}")

    def plot_cross_sectional_returns(
        self,
        model_name: str,
        decile_returns_df: pl.DataFrame,
        descending_sort: bool,
    ):
        """
        Plots and saves the cumulative return for each performance decile.
        """
        if decile_returns_df.is_empty():
            logger.warning(
                f"Decile returns data for model '{model_name}' is empty. Skipping cross-sectional plot."
            )
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        try:
            colormap = plt.colormaps.get_cmap("tab10")
        except AttributeError:
            colormap = plt.cm.get_cmap("tab10")

        unique_deciles = decile_returns_df["decile"].unique().to_list()
        sorted_deciles = sorted(unique_deciles, key=lambda s: int(s.split(" ")[1]))
        num_deciles = len(sorted_deciles)

        for i, decile_name in enumerate(sorted_deciles):
            decile_data = decile_returns_df.filter(
                pl.col("decile") == decile_name
            ).sort("date")
            if decile_data.is_empty():
                continue

            color = colormap(i / (num_deciles - 1) if num_deciles > 1 else 0.5)

            ax.plot(
                decile_data["date"],
                decile_data["cumulative_return"],
                label=str(decile_name),
                color=color,
                linewidth=1.5,
            )

        ax.set_title(f"Cross-Sectional Cumulative Returns by Decile for {model_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(
            title="Performance Decile", bbox_to_anchor=(1.05, 1), loc="upper left"
        )

        if descending_sort:
            note = "Note: Deciles are formed on predicted returns. Decile 1 represents the highest predictions, and Decile 10 the lowest."
        else:
            note = "Note: Deciles are formed on predicted returns. Decile 1 represents the lowest predictions, and Decile 10 the highest."

        fig.text(
            0.5,
            0.01,
            note,
            wrap=True,
            ha="center",
            fontsize=9,
            style="italic",
            color="gray",
        )

        fig.tight_layout(rect=(0, 0.05, 1, 1))

        plot_filename = (
            self.cross_sectional_dir / f"{model_name}_cross_sectional_returns.png"
        )
        fig.savefig(plot_filename, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Cross-sectional returns plot saved to: {plot_filename}")
