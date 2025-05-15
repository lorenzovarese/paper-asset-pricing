from pathlib import Path
import yaml
import typer
import pandas as pd

import polars as pl  # for lazy parquet slicing
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
    1. Load pre-aggregated dataset lazily (Parquet) or eagerly (CSV)
    2. Instantiate & fit a registered model
    3. Rolling-forward evaluation, only loading each window
    4. Save the final trained model
    """
    exp = yaml.safe_load(config.read_text(encoding="utf-8"))
    ds_conf = exp["dataset"]
    data_path = Path(ds_conf["path"])
    date_col = ds_conf.get("date_column", "date")
    id_col = ds_conf.get("id_column", "permno")

    # --- prepare lazy source or in-memory fallback ---
    is_parquet = data_path.suffix.lower() in {".parquet", ".pq"}
    if is_parquet:
        ds = pl.scan_parquet(data_path)
        # compute overall min/max dates without loading full data
        stats = ds.select(
            [
                pl.col(date_col).min().alias("start"),
                pl.col(date_col).max().alias("end"),
            ]
        ).collect()
        window_start = stats["start"][0]
        window_end = stats["end"][0]
        all_cols = list(ds.schema.keys())
    else:
        df_tmp = pd.read_csv(data_path, parse_dates=[date_col])
        df_tmp[date_col] = pd.to_datetime(df_tmp[date_col])
        window_start = df_tmp[date_col].min()
        window_end = df_tmp[date_col].max()
        all_cols = df_tmp.columns.tolist()
        del df_tmp  # free memory

    # --- determine features & target once ---
    target = exp["model"]["target"]
    exclude = set(exp["model"].get("exclude", [])) | {target, id_col, date_col}
    features = [c for c in all_cols if c not in exclude]

    # --- instantiate model ---
    name = exp["model"]["name"]
    ModelCls = get_model(name)
    params = exp["model"].get("params", {})
    typer.echo(f"Using model '{name}' with params {params}")
    model = ModelCls(**params)

    # --- rolling-forward parameters ---
    ev = exp["evaluation"]
    t_years, v_years, test_years, roll_years = (
        ev["train_years"],
        ev["val_years"],
        ev["test_years"],
        ev["roll_years"],
    )

    # --- loop over windows, but only load each slice into pandas ---
    start = window_start
    while True:
        train_end = start + pd.DateOffset(years=t_years) - pd.Timedelta(days=1)
        test_end = train_end + pd.DateOffset(years=v_years + test_years)

        if test_end > window_end:
            break

        if is_parquet:
            # Polars lazy filter + collect
            train_pl = (
                ds.filter(pl.col(date_col).is_between(start, train_end, closed="both"))
                .select([*features, target])
                .collect()
            )
            test_pl = (
                ds.filter(
                    pl.col(date_col).is_between(train_end, test_end, closed="right")
                )
                .select([*features, target])
                .collect()
            )
            train_df = train_pl.to_pandas()
            test_df = test_pl.to_pandas()
        else:
            # full CSV read + slice (less ideal for huge CSVs)
            df_all = pd.read_csv(data_path, parse_dates=[date_col])
            df_all[date_col] = pd.to_datetime(df_all[date_col])
            mask_tr = (df_all[date_col] >= start) & (df_all[date_col] <= train_end)
            mask_te = (df_all[date_col] > train_end) & (df_all[date_col] <= test_end)
            train_df = df_all.loc[mask_tr, features + [target]]
            test_df = df_all.loc[mask_te, features + [target]]
            del df_all

        # prepare arrays, fit, predict, evaluate
        X_train, y_train = train_df[features].values, train_df[target].values
        X_test, y_test = test_df[features].values, test_df[target].values

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        typer.echo(
            f"Window {start.date()}→{test_end.date()}: MSE={mse:.4f}, R²={r2:.4f}"
        )

        start = start + pd.DateOffset(years=roll_years)

    # --- save final model ---
    out = Path(exp["model"]["output_path"])
    model.save(out)
    typer.secho(f"✅ Final model saved to {out}", fg=typer.colors.GREEN)
