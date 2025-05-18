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
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean((yt - yp) ** 2))


def r2_out_of_sample(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Out-of-sample R^2 as in Gu et al. (2020):
      R^2_oos = 1 - SS_res / SS_tot,
      where SS_res = Σ(y_true_i - y_pred_i)^2,
            SS_tot = Σ(y_true_i)^2 (benchmark = zero forecast).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum(yt**2)
    # If all y_true are zero, define R^2_oos = 0 to avoid division by zero
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)
