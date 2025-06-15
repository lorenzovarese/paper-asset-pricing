import polars as pl
import pytest
from datetime import date

from paper_portfolio.manager import PortfolioManager  # type: ignore
from paper_portfolio.config_parser import (  # type: ignore
    PortfolioConfig,
    InputDataConfig,
    PortfolioStrategyConfig,
)


@pytest.fixture
def simple_config(tmp_path):
    inp = InputDataConfig(
        prediction_model_names=["m1"],
        processed_dataset_name="pfx",
        date_column="date",
        id_column="id",
        risk_free_rate_col="rf",
        value_weight_col="vw",
    )
    strat = PortfolioStrategyConfig(
        name="s1",
        weighting_scheme="equal",
        long_quantiles=[0.0, 1.0],
        short_quantiles=[0.0, 1.0],
    )
    cfg = PortfolioConfig(input_data=inp, strategies=[strat], metrics=["sharpe_ratio"])
    return cfg


def test_load_data_no_processed(tmp_path, simple_config):
    mgr = PortfolioManager(simple_config)
    proj = tmp_path / "proj"
    # no data/processed
    with pytest.raises(FileNotFoundError):
        mgr._load_data(proj)


def test_load_data_success(tmp_path, simple_config):
    mgr = PortfolioManager(simple_config)
    proj = tmp_path / "proj"
    # make processed and predictions dirs
    (proj / "data" / "processed").mkdir(parents=True)
    (proj / "models" / "predictions").mkdir(parents=True)
    # write two processed parquet files
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 31), date(2020, 2, 29)],
            "id": [1, 2],
            "rf": [0.01, 0.02],
            "vw": [10.0, 20.0],
        }
    )
    df.write_parquet(proj / "data" / "processed" / "pfx_202001.parquet")
    df.write_parquet(proj / "data" / "processed" / "pfx_202002.parquet")
    # write one predictions file
    preds = pl.DataFrame(
        {
            "date": [date(2020, 1, 31)],
            "id": [1],
            "predicted_ret": [0.05],
            "actual_ret": [0.04],
        }
    )
    preds.write_parquet(proj / "models" / "predictions" / "m1_predictions.parquet")
    data = mgr._load_data(proj)
    assert "m1" in data
    out = data["m1"]
    # joined columns
    for col in ("predicted_ret", "actual_ret", "rf", "vw"):
        assert col in out.columns


def test_load_index_data(tmp_path, simple_config):
    from paper_portfolio.config_parser import MarketBenchmarkConfig

    mgr = PortfolioManager(simple_config)
    proj = tmp_path / "proj"
    idx_dir = proj / "portfolios" / "indexes"
    idx_dir.mkdir(parents=True)
    # write CSV
    csv = idx_dir / "bm.csv"
    csv.write_text("date,ret\n2020-01-31,0.03\n2020-02-29,0.04")
    bench = MarketBenchmarkConfig(
        name="BM",
        file_name="bm.csv",
        date_column="date",
        return_column="ret",
        date_format="%Y-%m-%d",
    )
    df = mgr._load_index_data(proj, bench)
    assert df is not None
    assert list(df.columns) == ["date", "index_ret"]
    # missing file
    bench2 = MarketBenchmarkConfig(
        name="X",
        file_name="nope.csv",
        date_column="date",
        return_column="ret",
        date_format="%Y-%m-%d",
    )
    assert mgr._load_index_data(proj, bench2) is None


def test_calculate_cross_sectional_returns(simple_config):
    mgr = PortfolioManager(simple_config)
    # create 10 assets on same date
    dates = [date(2020, 1, 31)] * 10
    preds = list(range(10))
    acts = [0.01 * i for i in range(10)]
    df = pl.DataFrame({"date": dates, "predicted_ret": preds, "actual_ret": acts})
    cs, desc = mgr._calculate_cross_sectional_returns(df)
    assert not cs.is_empty()
    assert not desc
    assert "cumulative_return" in cs.columns


def test_calculate_monthly_returns(simple_config):
    mgr = PortfolioManager(simple_config)
    dates = [date(2020, 1, 31)] * 4
    df = pl.DataFrame(
        {
            "date": dates,
            "id": [1, 2, 3, 4],
            "predicted_ret": [0.1, 0.2, 0.3, 0.4],
            "actual_ret": [0.05, 0.06, 0.07, 0.08],
            "rf": [0.01, 0.01, 0.01, 0.01],
            "vw": [10.0, 20.0, 30.0, 40.0],
        }
    )
    # use simple config strat that covers full [0,1] quantile
    out = mgr._calculate_monthly_returns(df)
    assert not out.is_empty()
    # one row per strategy (only 1 strat)
    assert out.filter(pl.col("strategy") == "s1").height >= 1
    for col in ["long_return", "short_return", "portfolio_return", "risk_free_rate"]:
        assert col in out.columns
