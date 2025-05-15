from pathlib import Path
import typer
import yaml

from paperassetpricing.etl.schema import AggregationConfig
from paperassetpricing.etl.aggregator import DataAggregator

app = typer.Typer()


@app.command()
def run(
    spec: Path = typer.Option(
        ...,
        "-c",
        "--config",
        exists=True,
        file_okay=True,
        help="YAML spec file for aggregation (schema-validated).",
    ),
    output: Path = typer.Option(
        None, "-o", "--output", help="Where to write the aggregated file."
    ),
):
    """
    Load, merge, transform and optionally write the aggregated dataset.
    """
    # 1. parse config
    raw = yaml.safe_load(spec.read_text(encoding="utf-8"))
    cfg = AggregationConfig.model_validate(raw)

    # 2. run aggregator
    aggregator = DataAggregator(cfg)
    # load, merge, then apply transformations
    merged_df = aggregator.load().merge()
    aggregated_df = aggregator.apply_transformations(merged_df)

    # 3. output
    if output:
        fmt = (
            cfg.output.format
            if cfg.output and cfg.output.format
            else output.suffix.lstrip(".")
        )
        if fmt == "csv":
            aggregated_df.to_csv(output, index=False)
        else:
            aggregated_df.to_parquet(
                output,
                engine="auto",
                compression="snappy",
                index=False,
            )
        typer.secho(f"Wrote aggregated data to {output}", fg=typer.colors.GREEN)
    else:
        typer.secho(
            "Aggregation complete. No output path given; returning DataFrame object.",
            fg=typer.colors.YELLOW,
        )
        typer.echo(aggregated_df)
