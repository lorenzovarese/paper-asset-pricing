from pathlib import Path
import yaml
import typer

from paperassetpricing.etl.schema import AggregationConfig
from paperassetpricing.etl.aggregator import DataAggregator


def aggregate(
    spec: Path = typer.Option(
        ...,
        "-c",
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="YAML spec file for aggregation (schema-validated).",
    ),
    output: Path = typer.Option(
        None,
        "-o",
        "--output",
        file_okay=True,
        dir_okay=False,
        help="Where to write the aggregated file.",
    ),
) -> None:
    """
    Load, merge, transform and optionally write the aggregated dataset.
    """
    raw = yaml.safe_load(spec.read_text(encoding="utf-8"))
    cfg = AggregationConfig.model_validate(raw)

    agg = DataAggregator(cfg)
    merged_df = agg.load().merge()
    aggregated_df = agg.apply_transformations(merged_df)

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
                output, engine="auto", compression="snappy", index=False
            )
        typer.secho(f"Wrote aggregated data to {output}", fg=typer.colors.GREEN)
    else:
        typer.secho(
            "Aggregation complete. No output path given; printing DataFrame.",
            fg=typer.colors.YELLOW,
        )
        typer.echo(aggregated_df)
