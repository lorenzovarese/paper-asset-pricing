import os
import yaml
import typer
import getpass
from pathlib import Path

from paper_data.ingestion.http import HTTPConnector
from paper_data.ingestion.local import LocalConnector
from paper_data.ingestion.wrds_conn import WRDSConnector

app = typer.Typer(
    help="Ingest datasets according to a YAML spec for reproducible asset-pricing research."
)

CONNECTOR_MAP = {
    "http": HTTPConnector,
    "local": LocalConnector,
    "wrds": WRDSConnector,
    # TODO: extend with "google_drive", "huggingface" as needed
}


@app.command()
def ingest(
    config: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to YAML ingestion configuration file",
        rich_help_panel="Configuration",
    ),
):
    """
    Load all datasets defined in the YAML CONFIG, saving outputs under `output_dir`.
    """
    # 1) Load and validate config
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    out_base = Path(cfg.get("output_dir", "PAPER/data"))
    out_base.mkdir(parents=True, exist_ok=True)

    # 2) Iterate over each dataset spec
    for ds in cfg.get("datasets", []):
        name = ds["name"]
        kind = ds["connector"]
        params = ds.get("params", {}).copy()
        save_as = ds.get("save_as", f"{name}.csv")

        # Interactive prompt for passwords
        if params.get("password") == "prompt":
            params["password"] = getpass.getpass(f"{name} WRDS password: ")

        # Instantiate connector
        ConnectorCls = CONNECTOR_MAP.get(kind)
        if ConnectorCls is None:
            typer.secho(
                f"⚠️  Unknown connector '{kind}' for dataset '{name}'",
                fg=typer.colors.RED,
            )
            continue

        typer.secho(f"→ Ingesting '{name}' via {kind}", fg=typer.colors.BLUE)
        connector = ConnectorCls(**params)
        try:
            df = connector.get_data()
        except Exception as e:
            typer.secho(
                f"✖ Failed to fetch '{name}': {e}", fg=typer.colors.RED, err=True
            )
            continue

        # 3) Persist to disk
        out_path = out_base / save_as
        if out_path.suffix.lower() in {".parquet", ".pq"}:
            df.to_parquet(out_path, index=False)
        else:
            df.to_csv(out_path, index=False)

        typer.secho(f"✔ Saved '{name}' → {out_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
