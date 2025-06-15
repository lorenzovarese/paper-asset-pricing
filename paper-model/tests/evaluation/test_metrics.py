import numpy as np
import pytest
from paper_model.evaluation.metrics import (  # type: ignore
    mean_squared_error,
    r2_out_of_sample,
    r2_adj_out_of_sample,
)


def test_mse_zero():
    y = [1.0, 2.0, 3.0]
    assert mean_squared_error(y, y) == 0.0


def test_mse_nonzero():
    assert mean_squared_error([0, 0], [1, 1]) == 1.0


def test_r2_oos_regular():
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.0, 2.0, 4.0])
    # ss_res = 1, ss_tot = 1+4+9=14 → R2 = 1 - 1/14
    assert pytest.approx(r2_out_of_sample(yt, yp), rel=1e-6) == 1 - 1 / 14


def test_r2_oos_zero_true():
    yt = np.array([0, 0])
    assert r2_out_of_sample(yt, np.array([0, 0])) == 1.0
    assert r2_out_of_sample(yt, np.array([1, 1])) == -np.inf


def test_r2_adj_oos_no_adjustment():
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.0, 2.0, 3.0])
    # n=3, p_z=2 → n <= p_z+1 → return unadjusted R² = 1.0
    assert r2_adj_out_of_sample(yt, yp, n_predictors=2) == 1.0


def test_r2_adj_oos_with_adjustment():
    yt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    yp = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    # ss_res = 1, ss_tot=55 → R2 = 1 - 1/55
    r2 = 1 - 1 / 55
    # Adjustment: n=5, p_z=1 → factor = (n-1)/(n-p_z-1) = 4/3
    adjusted = 1 - (1 - r2) * (4 / 3)
    assert (
        pytest.approx(r2_adj_out_of_sample(yt, yp, n_predictors=1), rel=1e-6)
        == adjusted
    )
