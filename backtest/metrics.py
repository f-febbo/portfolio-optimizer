"""
Performance metrics replicating Table 2 from the paper.

All metrics computed from daily returns and portfolio values.
"""

import numpy as np
import pandas as pd
from scipy import stats


def annual_return(daily_returns: pd.Series) -> float:
    """Annualized geometric return."""
    total = (1 + daily_returns).prod()
    n_years = len(daily_returns) / 252.0
    if n_years <= 0 or total <= 0:
        return 0.0
    return total ** (1.0 / n_years) - 1.0


def cumulative_return(daily_returns: pd.Series) -> float:
    """Total cumulative return."""
    return (1 + daily_returns).prod() - 1.0


def annual_volatility(daily_returns: pd.Series) -> float:
    """Annualized standard deviation of returns."""
    return daily_returns.std() * np.sqrt(252)


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio (assuming daily risk-free rate)."""
    excess = daily_returns - risk_free_rate / 252.0
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(252)


def max_drawdown(daily_values: pd.Series) -> float:
    """Maximum drawdown (returned as a negative number)."""
    cummax = daily_values.cummax()
    drawdown = (daily_values - cummax) / cummax
    return drawdown.min()


def calmar_ratio(daily_returns: pd.Series, daily_values: pd.Series) -> float:
    """Calmar ratio: annualized return / abs(max drawdown)."""
    ann_ret = annual_return(daily_returns)
    mdd = max_drawdown(daily_values)
    if mdd == 0:
        return 0.0
    return ann_ret / abs(mdd)


def sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    excess = daily_returns - risk_free_rate / 252.0
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    downside_std = np.sqrt(np.mean(downside ** 2))
    return (excess.mean() / downside_std) * np.sqrt(252)


def omega_ratio(daily_returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega ratio: probability-weighted gain/loss ratio."""
    gains = daily_returns[daily_returns > threshold] - threshold
    losses = threshold - daily_returns[daily_returns <= threshold]
    if losses.sum() == 0:
        return float("inf") if gains.sum() > 0 else 1.0
    return 1.0 + gains.sum() / losses.sum()


def stability(daily_returns: pd.Series) -> float:
    """R-squared of a linear fit to cumulative log returns."""
    cum_log = np.log1p(daily_returns).cumsum()
    if len(cum_log) < 2:
        return 0.0
    x = np.arange(len(cum_log))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, cum_log)
    return r_value ** 2


def tail_ratio(daily_returns: pd.Series) -> float:
    """Ratio of 95th percentile to absolute 5th percentile."""
    p95 = np.percentile(daily_returns, 95)
    p5 = np.percentile(daily_returns, 5)
    if p5 == 0:
        return 0.0
    return abs(p95 / p5)


def daily_value_at_risk(daily_returns: pd.Series, confidence: float = 0.05) -> float:
    """Historical VaR at given confidence level (returned as negative)."""
    return np.percentile(daily_returns, confidence * 100)


def portfolio_turnover(daily_weights: pd.DataFrame) -> float:
    """
    Average daily portfolio weight change.
    delta_pw = mean(sum(|w_t - w_{t-1}|)) across all days.
    """
    if len(daily_weights) < 2:
        return 0.0
    diffs = daily_weights.diff().abs().sum(axis=1)
    return diffs.iloc[1:].mean()


def compute_all_metrics(
    daily_returns: pd.Series,
    daily_values: pd.Series,
    daily_weights: pd.DataFrame = None,
) -> dict:
    """
    Compute all performance metrics from Table 2 of the paper.

    Returns a dictionary of metric_name -> value.
    """
    # Skip the first day (return is 0 by construction)
    dr = daily_returns.iloc[1:] if len(daily_returns) > 1 else daily_returns

    metrics = {
        "Annual return": annual_return(dr),
        "Cumulative returns": cumulative_return(dr),
        "Annual volatility": annual_volatility(dr),
        "Sharpe ratio": sharpe_ratio(dr),
        "Calmar ratio": calmar_ratio(dr, daily_values),
        "Stability": stability(dr),
        "Max drawdown": max_drawdown(daily_values),
        "Omega ratio": omega_ratio(dr),
        "Sortino ratio": sortino_ratio(dr),
        "Skew": float(dr.skew()),
        "Kurtosis": float(dr.kurtosis()),
        "Tail ratio": tail_ratio(dr),
        "Daily value at risk": daily_value_at_risk(dr),
    }

    if daily_weights is not None:
        metrics["Portfolio turnover"] = portfolio_turnover(daily_weights)

    return metrics
