import numpy as np
import polars as pl
import pytest

from paper_portfolio.evaluation.metrics import (  # type: ignore
    annualized_sharpe_ratio,
    expected_shortfall,
    cumulative_return,
)


def test_sharpe_ratio_basic():
    rets = pl.Series("r", [0.01, 0.02, -0.005, 0.015])
    rf = pl.Series("rf", [0.0, 0.0, 0.0, 0.0])
    sr = annualized_sharpe_ratio(rets, rf, periods_per_year=12)
    # excess returns same as returns; compute manually
    mu_opt = rets.mean()
    assert mu_opt is not None, "Series.mean() unexpectedly returned None"
    mu = mu_opt
    sd_opt = rets.std()
    assert sd_opt is not None, "Series.std() unexpectedly returned None"
    sd = sd_opt
    expected = (mu / sd) * np.sqrt(12)  # type: ignore
    assert pytest.approx(sr, rel=1e-6) == expected


def test_sharpe_ratio_empty():
    empty = pl.Series("r", dtype=pl.Float64)
    assert np.isnan(annualized_sharpe_ratio(empty, empty))


def test_expected_shortfall_basic():
    rets = pl.Series("r", [-0.1, 0.0, 0.05, 0.1])
    # at 95% conf, alpha=0.05, VaR=lowest 5% quantile = -0.1
    es = expected_shortfall(rets, confidence_level=0.95)
    # only returns <= -0.1 → [-0.1], mean = -0.1
    assert es == pytest.approx(-0.1)


def test_expected_shortfall_empty():
    empty = pl.Series("r", dtype=pl.Float64)
    assert np.isnan(expected_shortfall(empty))


def test_cumulative_return_basic():
    rets = pl.Series("r", [0.1, 0.1, -0.05])
    cr = cumulative_return(rets)
    # [0.1, 0.1, -0.05] → [0.1, 1.1*1.1-1=0.21, 1.21*0.95-1=0.1495]
    expected = [0.1, 0.21, 0.1495]
    # compare element‐wise within tolerance
    actual = list(cr)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a == pytest.approx(e, rel=1e-6)


def test_cumulative_return_empty():
    empty = pl.Series("r", dtype=pl.Float64)
    out = cumulative_return(empty)
    assert isinstance(out, pl.Series)
    assert out.is_empty()
