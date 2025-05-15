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


def r2_score(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    """
    Compute the R^2 (coefficient of determination) regression score.

    R^2 = 1 - (SS_res / SS_tot)
    where SS_res = Σ (y_true_i - y_pred_i)^2
          SS_tot = Σ (y_true_i - mean(y_true))^2
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    # If variance is zero, define R^2 as 0.0 to avoid division by zero
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)
