import typer
from pathlib import Path

from paperassetpricing.portfolios.performance import PerformanceBuilder

app = typer.Typer(help="Build, save & plot portfolio performance")


@app.command("run")
def run_portfolio(
    config: Path = typer.Option(
        ...,
        "-c",
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="YAML config for portfolio construction",
    ),
) -> None:
    """
    1. Load config
    2. Compute portfolio returns (long/short)
    3. Save CSV of monthly returns
    4. Plot cumulative returns
    """
    # 1) parse config & build
    builder = PerformanceBuilder.from_yaml(config)
    perf_df = builder.build()

    # 2) write out the CSV and the plot
    csv_out = builder.save_csv(perf_df)
    plot_out = builder.plot_cumulative(perf_df)

    typer.secho(f"✅ Monthly returns saved to {csv_out}", fg=typer.colors.GREEN)
    typer.secho(f"✅ Cumulative chart saved to {plot_out}", fg=typer.colors.GREEN)
