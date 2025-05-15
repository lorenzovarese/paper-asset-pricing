from pathlib import Path
import yaml
import typer
import pandas as pd

from paperassetpricing.models import get_model
from paperassetpricing.metrics import mean_squared_error, r2_score


def experiment(
    config: Path = typer.Option(
        ...,
        "-c",
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="YAML config for experiment.",
    ),
) -> None:
    """
    1. Load pre-aggregated dataset
    2. Instantiate & fit a registered model
    3. Rolling-forward evaluation
    4. Save the final trained model
    """
    exp = yaml.safe_load(config.read_text(encoding="utf-8"))

    # --- load data ---
    data_path = Path(exp["dataset"]["path"])
    if data_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, parse_dates=True)
    date_col = exp["dataset"].get("date_column", "date")
    df["date"] = pd.to_datetime(df[date_col])
    df = df.sort_values("date")

    # --- features & target ---
    target = exp["model"]["target"]
    exclude = set(exp["model"].get("exclude", [])) | {
        target,
        exp["dataset"].get("id_column", "permno"),
        date_col,
    }
    features = [c for c in df.columns if c not in exclude]

    # --- model instantiation ---
    name = exp["model"]["name"]
    ModelCls = get_model(name)
    params = exp["model"].get("params", {})
    typer.echo(f"Using model '{name}' with params {params}")
    model = ModelCls(**params)

    # --- rolling-forward eval ---
    ev = exp["evaluation"]
    ty, vy, ty2, ry = (
        ev["train_years"],
        ev["val_years"],
        ev["test_years"],
        ev["roll_years"],
    )

    start, end = df["date"].min(), df["date"].max()
    while True:
        train_end = start + pd.DateOffset(years=ty) - pd.Timedelta(days=1)
        test_end = train_end + pd.DateOffset(years=vy + ty2)
        if test_end > end:
            break

        train_df = df[(df["date"] >= start) & (df["date"] <= train_end)]
        test_df = df[(df["date"] > train_end) & (df["date"] <= test_end)]

        X_train, y_train = train_df[features].values, train_df[target].values
        X_test, y_test = test_df[features].values, test_df[target].values

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        typer.echo(
            f"Window {start.date()}→{test_end.date()}: MSE={mse:.4f}, R²={r2:.4f}"
        )

        start = start + pd.DateOffset(years=ry)

    # --- save final model ---
    out = Path(exp["model"]["output_path"])
    model.save(out)
    typer.secho(f"✅ Final model saved to {out}", fg=typer.colors.GREEN)
