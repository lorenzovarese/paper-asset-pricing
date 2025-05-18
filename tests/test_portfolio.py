import pandas as pd
import numpy as np
import pytest
from paperassetpricing.portfolios.performance import (
    monthly_portfolio_backtest,
    compute_performance_metrics,
)  # type: ignore


# A simple model that predicts the first feature as the signal
class PredictModel:
    def fit(self, X, y):
        return self

    def predict(self, X):
        # assume X is 2D, first column is the desired signal
        return X[:, 0]


def test_monthly_backtest_equal_weight_two_months():
    # --- Setup train and test DataFrames ---
    train_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "x": [1, 2],
            "ret": [10, 20],
        }
    )
    test_df = pd.DataFrame(
        {
            # two months: Mar and Apr
            "date": pd.to_datetime(
                ["2020-03-31", "2020-03-31", "2020-04-30", "2020-04-30"]
            ),
            "x": [1, 3, 2, 4],
            "ret": [10, 30, 20, 40],
        }
    )

    model = PredictModel()

    # Use 50th percentile for both long and short splits
    df = monthly_portfolio_backtest(
        model=model,
        train_df=train_df,
        test_df=test_df,
        features=["x"],
        date_col="date",
        signal_col="signal",
        ret_col="ret",
        long_lower_q=0.5,  # median
        long_upper_q=None,  # top tail
        short_lower_q=None,
        short_upper_q=0.5,  # bottom tail
        weight_mode="equal",
        weight_col=None,
        val_years=0,
    )

    # Expect two rows, for Mar and Apr
    assert list(df["date"]) == [
        pd.Timestamp("2020-03-31"),
        pd.Timestamp("2020-04-30"),
    ]
    # For Mar: signal=[1,3], median=2 → long_leg ret=[30], short_leg ret=[10]
    assert df.iloc[0]["long_ret"] == 30
    assert df.iloc[0]["short_ret"] == 10
    assert df.iloc[0]["port_ret"] == 20
    # For Apr: signal=[2,4], median=3 → long_ret=40, short_ret=20
    assert df.iloc[1]["long_ret"] == 40
    assert df.iloc[1]["short_ret"] == 20
    assert df.iloc[1]["port_ret"] == 20


def test_monthly_backtest_value_weight():
    # Single forecasting month, test value-weighting behavior
    train_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-31")],
            "x": [1],
            "ret": [1],
            "w": [1],
        }
    )
    test_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-03-31"), pd.Timestamp("2020-03-31")],
            "x": [1, 2],
            "ret": [10, 20],
            "w": [1, 3],
        }
    )

    model = PredictModel()

    df = monthly_portfolio_backtest(
        model=model,
        train_df=train_df,
        test_df=test_df,
        features=["x"],
        date_col="date",
        signal_col="signal",
        ret_col="ret",
        long_lower_q=0.0,  # threshold = min(signal)=1
        long_upper_q=None,  # take all >=1
        short_lower_q=None,
        short_upper_q=0.0,  # bottom tail: <= min(signal)=1
        weight_mode="value",
        weight_col="w",
        val_years=0,
    )

    # There should be one row for 2020-03-31
    assert len(df) == 1
    # Value‐weighted long return: (10*1 + 20*3) / (1+3) = 70/4 = 17.5
    assert pytest.approx(df.iloc[0]["long_ret"], rel=1e-6) == 17.5
    # Short leg: only one asset at signal=min → ret=10, weight=1 → short_ret=10
    assert df.iloc[0]["short_ret"] == 10
    # port_ret = 17.5 - 10 = 7.5
    assert pytest.approx(df.iloc[0]["port_ret"], rel=1e-6) == 7.5


def test_compute_performance_metrics_zero_returns():
    # All zero returns → zero ann_ret, zero ann_vol, nan sharpe, zero drawdown
    ret = pd.Series([0.0, 0.0, 0.0, 0.0])
    perf = compute_performance_metrics(ret, periods_per_year=4)
    assert perf["annual_return"] == 0.0
    assert perf["annual_vol"] == 0.0
    assert np.isnan(perf["sharpe_ratio"])
    assert perf["max_drawdown"] == 0.0


def test_compute_performance_metrics_nonzero():
    # Constant 10% monthly return → ann_return=0.1*12, zero volatility
    ret = pd.Series([0.1, 0.1, 0.1])
    perf = compute_performance_metrics(ret)
    assert pytest.approx(perf["annual_return"], rel=1e-6) == 0.1 * 12
    assert pytest.approx(perf["annual_vol"], rel=1e-6) == 0.0
    assert np.isnan(perf["sharpe_ratio"])
    # Cumulative returns: [1.1,1.21,1.331] → drawdown = 0
    assert perf["max_drawdown"] == 0.0
