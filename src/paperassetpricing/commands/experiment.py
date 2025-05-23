# src/paperassetpricing/commands/experiment.py
from pathlib import Path
import yaml
import typer
import pandas as pd
import polars as pl

from paperassetpricing.helpers.date_utils import parse_and_normalize_date
from paperassetpricing.models import get_model
from paperassetpricing.metrics import (
    mean_squared_error,
    r2_out_of_sample,
    r2_adj_out_of_sample,
)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def prepare_data_source(data_path: Path, date_col: str):
    """Return (is_parquet, lazy_ds_or_None, window_start, window_end, all_columns)."""
    if data_path.suffix.lower() in {".parquet", ".pq"}:
        ds = pl.scan_parquet(data_path)
        stats = ds.select(
            [
                pl.col(date_col).min().alias("start"),
                pl.col(date_col).max().alias("end"),
            ]
        ).collect()
        raw_start, raw_end = stats["start"][0], stats["end"][0]
        start = parse_and_normalize_date(raw_start, align_to="monthly")
        end = parse_and_normalize_date(raw_end, align_to="monthly")
        return True, ds, start, end, list(ds.schema.keys())
    else:
        df = pd.read_csv(data_path, parse_dates=[date_col])
        # align column itself
        df[date_col] = pd.to_datetime(df[date_col])
        raw_start, raw_end = df[date_col].min(), df[date_col].max()
        start = parse_and_normalize_date(raw_start, align_to="monthly")
        end = parse_and_normalize_date(raw_end, align_to="monthly")
        return False, None, start, end, df.columns.tolist()


def get_features_and_target(cfg: dict, all_cols: list[str]):
    """
    If cfg['model']['include_features'] is present, use exactly those columns.
    Otherwise use every column in `all_cols` except:
      - the `target`
      - the dataset id_column
      - the dataset date_column
    """
    mconf = cfg["model"]
    target = mconf["target"]

    include = mconf.get("include_features")
    if include:
        # sanity check
        missing = set(include) - set(all_cols)
        if missing:
            raise typer.BadParameter(
                f"include_features not found in dataset: {missing}. \n\nThe available columns are: {all_cols}"
            )
        return include, target

    # default: every column except the identifiers+target
    exclude = {
        target,
        cfg["dataset"].get("id_column", "permno"),
        cfg["dataset"].get("date_column", "date"),
    }
    features = [c for c in all_cols if c not in exclude]
    return features, target


def instantiate_model(cfg: dict):
    name = cfg["model"]["name"]
    params = cfg["model"].get("params", {})
    typer.echo(f"Using model '{name}' with params {params}")
    return get_model(name)(**params)


def window_generator(start, end, t, v, test, roll):
    """Yield (window_start, train_end, test_end) until test_end> end."""
    while True:
        train_end = start + pd.DateOffset(years=t) - pd.Timedelta(days=1)
        test_end = train_end + pd.DateOffset(years=v + test)
        if test_end > end:
            break
        yield start, train_end, test_end
        start = start + pd.DateOffset(years=roll)


def load_window(is_parquet, ds, data_path, date_col, features, target, w):
    """Return (train_df, test_df) for one window w=(start,train_end,test_end)."""
    start, train_end, test_end = w
    if is_parquet:
        train = (
            ds.filter(pl.col(date_col).is_between(start, train_end, closed="both"))
            .select([*features, target])
            .collect()
            .to_pandas()
        )
        test = (
            ds.filter(pl.col(date_col).is_between(train_end, test_end, closed="right"))
            .select([*features, target])
            .collect()
            .to_pandas()
        )
    else:
        df_all = pd.read_csv(data_path, parse_dates=[date_col])
        df_all[date_col] = pd.to_datetime(df_all[date_col])
        mask_tr = (df_all[date_col] >= start) & (df_all[date_col] <= train_end)
        mask_te = (df_all[date_col] > train_end) & (df_all[date_col] <= test_end)
        train = df_all.loc[mask_tr, features + [target]]
        test = df_all.loc[mask_te, features + [target]]
        del df_all
    return train, test


def evaluate_window(model, train_df, test_df, features, target):
    """
    Fit on train_df, predict on test_df, and return:
      (mse, r2_oos, r2_adj_oos)
    """
    X_tr, y_tr = train_df[features].values, train_df[target].values
    X_te, y_te = test_df[features].values, test_df[target].values
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    mse = mean_squared_error(y_te, y_pred)
    r2_oos = r2_out_of_sample(y_te, y_pred)
    # we pass len(features) as the number of predictors p_z
    r2_adj = r2_adj_out_of_sample(y_te, y_pred, n_predictors=len(features))

    return mse, r2_oos, r2_adj


def save_model_and_metrics(model, results: list[dict], model_path: Path):
    model.save(model_path)
    typer.secho(f"✅ Final model saved to {model_path}", fg=typer.colors.GREEN)

    df = pd.DataFrame(results)
    out_csv = model_path.parent / f"{model_path.stem}_results.csv"
    df.to_csv(out_csv, index=False)
    typer.secho(f"✅ Experiment metrics written to {out_csv}", fg=typer.colors.GREEN)


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
    cfg = load_config(config)
    ds_conf = cfg["dataset"]
    data_path = Path(ds_conf["path"])
    date_col = ds_conf.get("date_column", "date")

    # prepare
    is_parquet, ds, start, end, all_cols = prepare_data_source(data_path, date_col)
    features, target = get_features_and_target(cfg, all_cols)
    model = instantiate_model(cfg)

    ev = cfg["evaluation"]
    windows = window_generator(
        start,
        end,
        ev["train_years"],
        ev["val_years"],
        ev["test_years"],
        ev["roll_years"],
    )

    results = []
    for w in windows:
        typer.echo(f"Evaluating window {w[0].date()}→{w[2].date()}...")
        train_df, test_df = load_window(
            is_parquet, ds, data_path, date_col, features, target, w
        )
        mse, r2_oos, r2_adj = evaluate_window(
            model, train_df, test_df, features, target
        )
        typer.echo(
            f"Window {w[0].date()}→{w[2].date()}: "
            f"MSE={mse:.4f}, R²_oos={r2_oos:.4f}, R²_adj_oos={r2_adj:.4f}"
        )
        results.append(
            {
                "train_start": w[0].date(),
                "train_end": w[1].date(),
                "test_end": w[2].date(),
                "mse": mse,
                "r2_oos": r2_oos,
                "r2_adj_oos": r2_adj,
            }
        )

    # save outputs
    model_path = Path(cfg["model"]["output_path"])
    save_model_and_metrics(model, results, model_path)
