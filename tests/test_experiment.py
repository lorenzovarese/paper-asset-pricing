import pytest
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from typer import BadParameter

from paperassetpricing.commands.experiment import (
    load_config,
    prepare_data_source,
    get_features_and_target,
    instantiate_model,
    window_generator,
    load_window,
    evaluate_window,
    save_model_and_metrics,
    experiment,
)
from paperassetpricing.models.linear_model import LinearModel

# --- Helpers -------------------------------------------------------------


class DummyModel:
    """Predict all ones regardless of fit."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(X.shape[0], dtype=float)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("dummy")


# --- Fixtures ------------------------------------------------------------


@pytest.fixture
def tmp_csv(tmp_path):
    # create a small CSV for testing load_window
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30", "2020-05-31"]
            ),
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
            "permno": [1, 1, 1, 1, 1],
        }
    )
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def tmp_parquet(tmp_path):
    # same data but Parquet
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30", "2020-05-31"]
            ),
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
            "permno": [1, 1, 1, 1, 1],
        }
    )
    p = tmp_path / "data.parquet"
    df.to_parquet(p, index=False)
    return p


@pytest.fixture
def basic_cfg(tmp_path, tmp_csv):
    # minimal config for integration
    cfg = {
        "dataset": {"path": str(tmp_csv), "date_column": "date", "id_column": "permno"},
        "model": {
            "name": "ols",
            "target": "y",
            "params": {},
            "output_path": str(tmp_path / "m.joblib"),
        },
        "evaluation": {
            "train_years": 1,
            "val_years": 0,
            "test_years": 1,
            "roll_years": 1,
        },
    }
    f = tmp_path / "cfg.yaml"
    f.write_text(yaml.safe_dump(cfg))
    return f, cfg


# --- Tests ---------------------------------------------------------------


def test_load_config(tmp_path):
    d = {"a": 1, "b": [2, 3]}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d))
    assert load_config(p) == d


def test_prepare_data_source_csv(tmp_csv):
    is_parquet, ds, start, end, cols = prepare_data_source(tmp_csv, "date")
    assert not is_parquet and ds is None
    assert pd.Timestamp("2020-01-31") == start
    assert pd.Timestamp("2020-05-31") == end
    assert set(cols) == {"date", "x", "y", "permno"}


def test_prepare_data_source_parquet(tmp_parquet):
    is_parquet, ds, start, end, cols = prepare_data_source(tmp_parquet, "date")
    assert is_parquet and hasattr(ds, "filter")
    assert pd.Timestamp("2020-01-31") == start
    assert pd.Timestamp("2020-05-31") == end
    assert set(cols) == {"date", "x", "y", "permno"}


def test_get_features_and_target_default():
    cfg = {"dataset": {"id_column": "id", "date_column": "d"}, "model": {"target": "t"}}
    all_cols = ["id", "d", "a", "b", "t"]
    feats, tgt = get_features_and_target(cfg, all_cols)
    assert tgt == "t"
    assert set(feats) == {"a", "b"}


def test_get_features_and_target_include():
    cfg = {
        "dataset": {"id_column": "id", "date_column": "d"},
        "model": {"target": "t", "include_features": ["a", "b"]},
    }
    all_cols = ["id", "d", "a", "b", "t", "c"]
    feats, tgt = get_features_and_target(cfg, all_cols)
    assert feats == ["a", "b"] and tgt == "t"


def test_get_features_and_target_missing():
    cfg = {
        "dataset": {"id_column": "id", "date_column": "d"},
        "model": {"target": "t", "include_features": ["z"]},
    }
    all_cols = ["id", "d", "a", "b", "t"]
    with pytest.raises(BadParameter):
        get_features_and_target(cfg, all_cols)


def test_instantiate_model_cpu():
    cfg = {"model": {"name": "ols", "params": {}}}
    m = instantiate_model({"model": cfg["model"]})
    assert isinstance(m, LinearModel)


def test_window_generator():
    start = pd.Timestamp("2000-01-31")
    end = pd.Timestamp("2005-12-31")
    wins = list(window_generator(start, end, 2, 0, 1, 2))
    # two windows:
    #  - 2000-01-31 → 2002-01-30 → 2003-01-30
    #  - 2002-01-31 → 2004-01-30 → 2005-01-30
    assert len(wins) == 2
    w0, w1 = wins
    assert w0[0] == start
    assert w0[1] == pd.Timestamp("2002-01-30")
    assert w0[2] == pd.Timestamp("2003-01-30")
    assert w1[0] == pd.Timestamp("2002-01-31")
    assert w1[1] == pd.Timestamp("2004-01-30")
    assert w1[2] == pd.Timestamp("2005-01-30")


def test_load_window_csv(tmp_csv):
    # window: Jan→Mar for training, Apr→May for test
    start = pd.Timestamp("2020-01-31")
    train_end = pd.Timestamp("2020-03-30")
    test_end = pd.Timestamp("2020-05-31")
    train, test = load_window(
        False, None, tmp_csv, "date", ["x"], "y", (start, train_end, test_end)
    )
    # with train_end=2020-03-30, only Jan & Feb are included
    assert list(train["x"]) == [1, 2]
    # test should contain Mar/Apr/May after 03-30 through 05-31
    assert list(test["x"]) == [3, 4, 5]


def test_load_window_parquet(tmp_parquet):
    start = pd.Timestamp("2020-02-29")
    train_end = pd.Timestamp("2020-03-30")
    test_end = pd.Timestamp("2020-05-31")
    is_par, ds, *_ = prepare_data_source(tmp_parquet, "date")
    train, test = load_window(
        is_par, ds, tmp_parquet, "date", ["x"], "y", (start, train_end, test_end)
    )
    # with train_end=2020-03-30, only the 02-29 row is included
    assert list(train["x"]) == [2]
    # test covers from >03-30 up to 05-31
    assert list(test["x"]) == [3, 4, 5]


def test_evaluate_window_constant():
    # y is always 1 -> perfect fit of ConstantDummy -> mse=0,r2=0 by definition
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 1, 1]})
    mse, r2_oos, r2_adj_oos = evaluate_window(DummyModel(), df, df, ["x"], "y")
    assert pytest.approx(mse) == 0.0
    assert pytest.approx(r2_oos) == 0.0
    assert pytest.approx(r2_adj_oos) == 0.0


def test_save_model_and_metrics(tmp_path):
    results = [
        {
            "train_start": "2020-01-31",
            "train_end": "2020-02-29",
            "test_end": "2020-03-31",
            "mse": 0.1,
            "r2": 0.9,
        }
    ]
    model_path = tmp_path / "mdl.joblib"
    save_model_and_metrics(DummyModel(), results, model_path)
    # model file
    assert (tmp_path / "mdl.joblib").exists()
    # metrics CSV
    out_csv = tmp_path / "mdl_results.csv"
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert list(df.columns) == ["train_start", "train_end", "test_end", "mse", "r2"]
    assert df.iloc[0]["mse"] == 0.1
    assert df.iloc[0]["r2"] == 0.9


def test_experiment_integration_csv(tmp_path, basic_cfg):
    cfg_file, raw_cfg = basic_cfg
    # build a trivial CSV: one train row + one test row
    df = pd.DataFrame(
        {
            "date": ["2020-01-31", "2021-01-31", "2022-01-31"],
            "permno": [1, 1, 1],
            "x": [10.0, 11.0, 12.0],
            "y": [1.0, 2.0, 3.0],
        }
    )
    data_csv = tmp_path / "small.csv"
    df.to_csv(data_csv, index=False)
    # tweak cfg to point at small.csv
    cfg = raw_cfg.copy()
    cfg["dataset"]["path"] = str(data_csv)
    cfg_file.write_text(yaml.safe_dump(cfg))

    # run experiment
    experiment(cfg_file)

    # results
    model_file = Path(cfg["model"]["output_path"])
    assert model_file.exists()
    res_csv = model_file.parent / f"{model_file.stem}_results.csv"
    assert res_csv.exists()
    res_df = pd.read_csv(res_csv)
    # one window: train=2020-01-31→2021-01-30,test_end=2022-01-30
    assert len(res_df) == 1
    assert (
        "mse" in res_df.columns
        and "r2_oos" in res_df.columns
        and "r2_adj_oos" in res_df.columns
    )
