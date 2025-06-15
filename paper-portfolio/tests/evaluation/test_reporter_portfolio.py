import polars as pl
import matplotlib

matplotlib.use("Agg")  # headless

import pytest
from paper_portfolio.evaluation.reporter import PortfolioReporter  # type: ignore


@pytest.fixture
def tmp_out(tmp_path):
    return tmp_path / "out"


def test_generate_report(tmp_out):
    rep = PortfolioReporter(tmp_out)
    metrics = {"sharpe_ratio": 1.2345, "expected_shortfall": -0.1234}
    rep.generate_report("modelA", "strat1", metrics)
    f = tmp_out / "modelA_strat1_report.txt"
    assert f.exists()
    text = f.read_text()
    assert "Model: modelA" in text
    assert "Strategy: strat1" in text
    assert "sharpe_ratio: 1.2345" in text


def test_save_monthly_returns(tmp_out):
    rep = PortfolioReporter(tmp_out)
    # empty DataFrame → no file
    empty = pl.DataFrame()
    # currently save_monthly_returns will write no files if df.is_empty()
    rep.save_monthly_returns("m", "s", empty)
    # should still create the "s" subfolder, but no actual CSV inside
    entries = list(tmp_out.rglob("*"))
    # allow only directories, no .csv files
    assert all(p.is_dir() for p in entries), (
        f"Unexpected files or CSVs created: {entries}"
    )

    # non-empty → parquet
    df = pl.DataFrame(
        {"date": [1, 2], "strategy": ["s", "s"], "portfolio_return": [0.1, 0.2]}
    )
    rep.save_monthly_returns("m", "s", df)
    pq = tmp_out / "m_s_monthly_returns.parquet"
    assert pq.exists()


def test_plot_cumulative_returns(tmp_out):
    rep = PortfolioReporter(tmp_out)
    # missing columns → skip
    df_fail = pl.DataFrame({"date": [1, 2], "cumulative_long": [0.1, 0.2]})
    rep.plot_cumulative_returns("m", "s", df_fail)
    assert not (tmp_out / "m_s_cumulative_return.png").exists()

    # full columns → produce png
    df = pl.DataFrame(
        {
            "date": [1, 2, 3],
            "cumulative_long": [0.1, 0.2, 0.3],
            "cumulative_short": [-0.05, 0.0, 0.05],
            "cumulative_portfolio": [0.05, 0.2, 0.35],
            "cumulative_risk_free": [0.0, 0.0, 0.0],
        }
    )
    rep.plot_cumulative_returns("m", "s", df, index_name=None)
    assert (tmp_out / "m_s_cumulative_return.png").exists()


def test_plot_cross_sectional_returns(tmp_out):
    rep = PortfolioReporter(tmp_out)
    # empty df → skip
    rep.plot_cross_sectional_returns("m", pl.DataFrame(), descending_sort=False)
    assert not any((tmp_out / "cross_sectional_analysis").iterdir())

    # create 10‐decile DataFrame for one date
    dates = [1] * 10
    deciles = [f"Decile {i + 1}" for i in range(10)]
    returns = [0.01 * i for i in range(10)]
    df = pl.DataFrame({"date": dates, "decile": deciles, "cumulative_return": returns})
    rep.plot_cross_sectional_returns("m", df, descending_sort=True)
    png = tmp_out / "cross_sectional_analysis" / "m_cross_sectional_returns.png"
    assert png.exists()
