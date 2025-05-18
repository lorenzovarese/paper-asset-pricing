import numpy as np
from typing import Sequence


def mean_squared_error(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    """
    Compute the mean squared error regression loss.

    MSE = (1/n) * Σ (y_true_i - y_pred_i)^2
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length."
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean((yt - yp) ** 2))


def r2_out_of_sample(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Out-of-sample R² as in Gu et al. (2020):
    R²_oos = 1 - (Σ(y_true_i - y_pred_i)^2) / (Σ(y_true_i)^2)
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length."
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true**2)

    if ss_tot == 0:
        # If true values are all zero:
        # If predictions are also all zero, then ss_res is 0, R2 is 1.0 (perfect prediction of zeros).
        # If predictions are not all zero, then ss_res > 0, and R2 is undefined or very negative.
        return 1.0 if ss_res == 0 else -np.inf
    return float(1 - ss_res / ss_tot)


def r2_adj_out_of_sample(
    y_true: np.ndarray, y_pred: np.ndarray, n_predictors: int
) -> float:
    """
    Adjusted out-of-sample R² (Gu et al. 2020, Eq. 3.28):

      R²_adj_oos = 1
        - (1 - R²_oos) * (n - 1) / (n - p_z - 1)

    where
      n = number of test observations,
      p_z = number of predictors.
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length."
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    n = yt.size

    # not enough degrees of freedom → return unadjusted
    if n <= n_predictors + 1:
        return r2_out_of_sample(yt, yp)

    r2_oos = r2_out_of_sample(yt, yp)
    adj_factor = (n - 1) / (n - n_predictors - 1)
    return float(1 - (1 - r2_oos) * adj_factor)
