import typer
from pathlib import Path

from paperassetpricing.etl.aggregator import aggregate_from_yaml

app = typer.Typer(
    help="P.A.P.E.R. CLI: Platform for Asset Pricing Experiment & Research"
)


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):
    """
    If no sub-command is provided, show this help and exit.
    """
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command("aggregate")
def aggregate_cmd(
    config: Path = typer.Option(
        ...,
        "-c",
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to aggregation YAML config.",
    ),
    output: Path = typer.Option(
        ...,
        "-o",
        "--output",
        file_okay=True,
        dir_okay=False,
        help="Where to write the resulting CSV or Parquet.",
    ),
) -> None:
    """
    Read all sources, merge them, apply transformations, and write to OUTPUT.
    """
    df, cfg = aggregate_from_yaml(config)

    fmt = (cfg.output.format if cfg.output and cfg.output.format else "parquet").lower()
    if fmt == "csv":
        df.to_csv(output, index=False)
    else:
        df.to_parquet(output, engine="auto", compression="snappy", index=False)

    typer.echo(f"✅ Aggregated data written to {output!r}")
