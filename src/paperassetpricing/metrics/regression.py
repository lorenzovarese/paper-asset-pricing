import numpy as np
from typing import Sequence
from numpy.typing import ArrayLike


def mean_squared_error(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    """
    Compute the mean squared error regression loss.

    MSE = (1/n) * Σ (y_true_i - y_pred_i)^2
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean((yt - yp) ** 2))


def r2_out_of_sample(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """
    Out-of-sample R² as in Gu et al. (2020):
      R²_oos = 1 - SS_res / SS_tot,
      where SS_res = Σ(y_true_i - y_pred_i)^2,
            SS_tot = Σ(y_true_i)^2  (benchmark = zero forecast).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum(yt**2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def r2_adj_out_of_sample(
    y_true: ArrayLike, y_pred: ArrayLike, n_predictors: int
) -> float:
    """
    Adjusted out-of-sample R² (Gu et al. 2020, Eq. 3.28):

      R²_adj_oos = 1
        - (1 - R²_oos) * (n - 1) / (n - p_z - 1)

    where
      n = number of test observations,
      p_z = number of predictors.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    n = yt.size

    # not enough degrees of freedom → return unadjusted
    if n <= n_predictors + 1:
        return r2_out_of_sample(yt, yp)

    r2_oos = r2_out_of_sample(yt, yp)
    adj_factor = (n - 1) / (n - n_predictors - 1)
    return float(1 - (1 - r2_oos) * adj_factor)
