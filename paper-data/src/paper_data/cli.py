import yaml
import typer
import getpass
import polars as pl
from pyarrow import Table  # type: ignore[import-untyped]
from pyarrow.parquet import ParquetWriter  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]
from pathlib import Path

from paper_data.ingestion.http import HTTPConnector  # type: ignore[import-untyped]
from paper_data.ingestion.local import LocalConnector  # type: ignore[import-untyped]
from paper_data.ingestion.wrds_conn import WRDSConnector  # type: ignore[import-untyped]
from paper_data.wrangling.analyzer.ui import display_report  # type: ignore[import-untyped]
from paper_data.wrangling.cleaner.cleaner import RawDataset, CleanerFactory  # type: ignore[import-untyped]

app = typer.Typer(help="paper-data CLI: ingest, profile, and clean your datasets.")

CONNECTOR_MAP = {
    "http": HTTPConnector,
    "local": LocalConnector,
    "wrds": WRDSConnector,
    # TODO: add more connectors as needed
}


@app.command()
def ingest(
    config: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="YAML ingestion config",
    ),
):
    """
    Download all datasets defined in CONFIG and save under `output_dir`.
    """
    cfg = yaml.safe_load(config.read_text())
    out_base = Path(cfg.get("output_dir", "PAPER/data"))
    out_base.mkdir(parents=True, exist_ok=True)

    for ds in cfg.get("datasets", []):
        name, kind = ds["name"], ds["connector"]
        params = ds.get("params", {}).copy()
        save_as = ds.get("save_as", f"{name}.csv")

        # prompt for password if requested
        if params.get("password") == "prompt":
            params["password"] = getpass.getpass(f"{name} WRDS password: ")

        Connector = CONNECTOR_MAP.get(kind)
        if Connector is None:
            typer.secho(f"Unknown connector '{kind}' for '{name}'", fg=typer.colors.RED)
            continue

        typer.secho(f"[ingest] → {name} via {kind}", fg=typer.colors.BLUE)
        try:
            df = Connector(**params).get_data()
        except Exception as e:
            typer.secho(f"Failed to fetch '{name}': {e}", fg=typer.colors.RED, err=True)
            continue

        out_path = out_base / save_as
        if out_path.suffix.lower() in {".parquet", ".pq"}:
            df.write_parquet(out_path)
        else:
            df.write_csv(out_path)

        typer.secho(f"[ingest] ✔ {out_path}", fg=typer.colors.GREEN)


@app.command()
def profile(
    source: Path = typer.Argument(..., exists=True, help="Path to CSV/Parquet file"),
    output: Path = typer.Option(
        "data_profile.html", help="HTML report filename (saved under PAPER/)"
    ),
    n_rows: int = typer.Option(None, help="Limit to last N rows"),
    explorative: bool = typer.Option(False, help="Enable explorative mode"),
    minimal: bool = typer.Option(True, help="Minimal profiling"),
):
    """
    Generate a profiling report for SOURCE, save it under PAPER/.
    """
    report_path = display_report(
        str(source),
        output_path=str(output),
        max_rows=n_rows,
        minimal=minimal,
        explorative=explorative,
    )
    typer.secho(f"[profile] ✔ {report_path}", fg=typer.colors.GREEN)


@app.command()
def clean(
    source: Path = typer.Argument(..., exists=True, help="Raw CSV/Parquet file"),
    config: Path = typer.Option(
        ..., exists=True, file_okay=True, help="YAML cleaning pipeline config"
    ),
    output_dir: Path = typer.Option(
        Path("PAPER/cleaned"), help="Directory to save cleaned datasets"
    ),
    objective: str = typer.Option("firm", help="Cleaning objective: 'firm' or 'macro'"),
):
    """
    Clean SOURCE according to CONFIG and write cleaned data to OUTPUT_DIR.
    """
    # 1) Load raw data
    ext = source.suffix.lower()
    if ext == ".csv":
        df = pl.read_csv(source, infer_schema_length=10_000)
    elif ext in {".parquet", ".pq"}:
        df = pl.read_parquet(source)
    else:
        typer.secho(f"Unsupported source format: {ext}", fg=typer.colors.RED)
        raise typer.Exit(1)

    # 2) Build cleaner pipeline
    pipeline = yaml.safe_load(config.read_text()).get("cleaning", {}).get(objective, [])
    raw = RawDataset(df, objective=objective)  # type: ignore[arg-type]
    cleaner = CleanerFactory.get_cleaner(raw)

    for step in pipeline:
        for method, args in step.items():
            getattr(cleaner, method)(**(args or {}))
            typer.secho(f"[clean] applied {method}", fg=typer.colors.BLUE)

    cleaned_df = cleaner.df

    # 3) Persist cleaned output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (source.stem + "_cleaned.parquet")
    cleaned_df.write_parquet(out_path)
    typer.secho(f"[clean] ✔ {out_path}", fg=typer.colors.GREEN)


@app.command("convert")
def convert(
    source: Path = typer.Argument(..., exists=True, help="CSV file to convert"),
):
    """
    Convert a large CSV into Parquet in streaming fashion using Polars.
    """
    out_path = source.with_suffix(".parquet")
    writer: ParquetWriter | None = None

    typer.secho(f"[convert] → {source} → {out_path}", fg=typer.colors.BLUE)

    # 1) Build a lazy CSV scan (no chunk_size arg)
    lazy_df = pl.scan_csv(str(source))

    # 2) Execute in streaming mode; get DataFrame
    print("Collecting data...")
    streaming_df = lazy_df.collect(engine="streaming")

    # 3) Convert to Arrow Table, then iterate its RecordBatches
    for rb in tqdm(streaming_df.to_arrow().to_batches(), desc="Chunks"):
        table = Table.from_batches([rb])
        if writer is None:
            writer = ParquetWriter(str(out_path), table.schema)
        writer.write_table(table)

    if writer:
        writer.close()
        typer.secho(f"[convert] ✔ wrote {out_path}", fg=typer.colors.GREEN)
    else:
        typer.secho("[convert] ⚠️ no data read from CSV", fg=typer.colors.RED)


if __name__ == "__main__":
    app()
