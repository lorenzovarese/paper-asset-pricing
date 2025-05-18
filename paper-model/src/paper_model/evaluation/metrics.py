import numpy as np
from sklearn.metrics import r2_score, mean_squared_error  # type: ignore[import-untyped]


def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the R-squared (coefficient of determination).

    Args:
        y_true: Array of true target values.
        y_pred: Array of predicted target values.

    Returns:
        The R-squared value.
    """
    return r2_score(y_true, y_pred)


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Squared Error (MSE).

    Args:
        y_true: Array of true target values.
        y_pred: Array of predicted target values.

    Returns:
        The Mean Squared Error value.
    """
    return mean_squared_error(y_true, y_pred)


# Add more metrics as needed (e.g., Sharpe ratio, Information Ratio, etc.
# These might require more context like risk-free rates or factor returns,
# and might be better suited for the portfolio module or a dedicated
# backtesting/performance analysis module.)
