import polars as pl
import numpy as np
import logging

logger = logging.getLogger(__name__)


def annualized_sharpe_ratio(
    portfolio_returns: pl.Series, risk_free_rate: pl.Series, periods_per_year: int = 12
) -> float:
    """
    Calculates the annualized Sharpe ratio with robust type checking.
    Assumes portfolio_returns are monthly and risk_free_rate is annualized.
    """
    if portfolio_returns.is_empty():
        return np.nan

    # De-annualize the risk-free rate to get the monthly rate before calculating excess returns.
    monthly_risk_free_rate = risk_free_rate / periods_per_year
    excess_returns = portfolio_returns - monthly_risk_free_rate

    # Get the mean and validate its type.
    mean_val = excess_returns.mean()
    if not isinstance(mean_val, (float, int)):
        logger.warning(
            f"Mean of excess returns is not a valid number (got {type(mean_val)}). Cannot compute Sharpe ratio."
        )
        return np.nan

    # Get the std dev and validate its type and value.
    std_val = excess_returns.std()
    if not isinstance(std_val, (float, int)):
        logger.warning(
            f"Standard deviation of excess returns is not a valid number (got {type(std_val)}). Cannot compute Sharpe ratio."
        )
        return np.nan

    if std_val == 0:
        logger.warning(
            "Standard deviation of excess returns is zero. Cannot compute Sharpe ratio."
        )
        return np.nan

    # At this point, mean_val and std_val are guaranteed to be numbers, and std_val is non-zero.
    # The division is now safe and type-correct.
    sharpe_ratio = mean_val / std_val

    return float(sharpe_ratio * np.sqrt(periods_per_year))


def expected_shortfall(
    portfolio_returns: pl.Series, confidence_level: float = 0.95
) -> float:
    """
    Calculates the Expected Shortfall (ES) with robust type checking.
    """
    if portfolio_returns.is_empty():
        return np.nan

    alpha = 1 - confidence_level

    # Get the quantile and validate its type.
    var = portfolio_returns.quantile(alpha)
    if not isinstance(var, (float, int)):
        logger.warning(
            f"Quantile (VaR) is not a valid number (got {type(var)}). Cannot compute Expected Shortfall."
        )
        return np.nan

    # Get the mean of the tail and validate its type.
    es = portfolio_returns.filter(portfolio_returns <= var).mean()
    if not isinstance(es, (float, int)):
        logger.warning(
            f"Mean of tail returns is not a valid number (got {type(es)}). Cannot compute Expected Shortfall."
        )
        return np.nan

    return float(es)


def cumulative_return(portfolio_returns: pl.Series) -> pl.Series:
    """Calculates the cumulative return over time."""
    if portfolio_returns.is_empty():
        # Return an empty series of the correct type, preserving the name
        return pl.Series(name=portfolio_returns.name, dtype=pl.Float64)

    # This operation is inherently type-safe with Polars Series and does not require extra checks.
    return (1 + portfolio_returns).cum_prod() - 1
