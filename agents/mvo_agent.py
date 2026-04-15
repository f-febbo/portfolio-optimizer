"""
Mean-Variance Optimization (MVO) agent.

Uses a 60-day lookback to estimate expected returns and covariance,
then solves for the maximum Sharpe ratio portfolio using PyPortfolioOpt
with Ledoit-Wolf shrinkage on the covariance matrix.
"""

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from sklearn.covariance import LedoitWolf

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def _enforce_psd(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Enforce positive semi-definite condition on covariance matrix.
    Sets negative eigenvalues to a small positive value and rebuilds.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def compute_mvo_weights(
    returns_window: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Compute optimal portfolio weights using Max Sharpe MVO.

    Args:
        returns_window: DataFrame of daily log returns, shape (lookback, n_assets).
        risk_free_rate: Risk-free rate for Sharpe calculation.

    Returns:
        Dictionary mapping ticker -> weight.
    """
    tickers = returns_window.columns.tolist()
    n_assets = len(tickers)

    # Expected returns: annualized sample mean
    mu = returns_window.mean() * 252

    # Covariance: Ledoit-Wolf shrinkage, annualized
    try:
        cov_shrink = CovarianceShrinkage(returns_window).ledoit_wolf()
    except Exception:
        # Fallback: manual Ledoit-Wolf via sklearn
        lw = LedoitWolf().fit(returns_window.values)
        cov_shrink = pd.DataFrame(
            lw.covariance_ * 252,
            index=tickers, columns=tickers,
        )

    # Enforce PSD
    cov_values = _enforce_psd(cov_shrink.values)
    cov_shrink = pd.DataFrame(cov_values, index=tickers, columns=tickers)

    # Optimize for max Sharpe
    try:
        ef = EfficientFrontier(mu, cov_shrink, weight_bounds=(0, 1))
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights = ef.clean_weights()
    except Exception:
        # Fallback: try minimum volatility if max_sharpe fails
        # (can happen when all expected returns are negative)
        try:
            ef = EfficientFrontier(mu, cov_shrink, weight_bounds=(0, 1))
            ef.min_volatility()
            weights = ef.clean_weights()
        except Exception:
            # Last resort: equal weight
            weights = {t: 1.0 / n_assets for t in tickers}

    return weights


def run_mvo_strategy(
    log_returns: pd.DataFrame,
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    lookback: int = None,
) -> pd.DataFrame:
    """
    Run the MVO strategy over a date range, producing daily target weights.

    Args:
        log_returns: Full DataFrame of daily log returns (all dates).
        prices: Full DataFrame of daily prices (all dates).
        start_date: First date of the backtest period.
        end_date: Last date of the backtest period.
        lookback: Number of lookback days for mean/covariance estimation.

    Returns:
        DataFrame of daily weights with DatetimeIndex, columns = tickers.
    """
    lookback = lookback or config.LOOKBACK
    tickers = log_returns.columns.tolist()

    # Get dates within the backtest period
    mask = (log_returns.index >= start_date) & (log_returns.index <= end_date)
    backtest_dates = log_returns.index[mask]

    weight_records = []

    for date in backtest_dates:
        # Get the position of this date in the full index
        date_pos = log_returns.index.get_loc(date)

        if date_pos < lookback:
            # Not enough history — go all cash (zero weights)
            weights = {t: 0.0 for t in tickers}
        else:
            # Extract lookback window
            window = log_returns.iloc[date_pos - lookback:date_pos]

            # Only include tickers that have non-zero data in this window
            active_tickers = [t for t in tickers
                              if window[t].abs().sum() > 0]

            if len(active_tickers) < 2:
                weights = {t: 0.0 for t in tickers}
            else:
                active_weights = compute_mvo_weights(window[active_tickers])
                weights = {t: active_weights.get(t, 0.0) for t in tickers}

        weight_records.append(weights)

    weights_df = pd.DataFrame(weight_records, index=backtest_dates)
    return weights_df


if __name__ == "__main__":
    from data.features import compute_features

    dataset = compute_features()
    print("Running MVO strategy for 2020...")

    weights = run_mvo_strategy(
        dataset.log_returns, dataset.prices,
        start_date="2020-01-01", end_date="2020-12-31",
    )
    print(f"Generated {len(weights)} days of weights")
    print(f"Average weights:\n{weights.mean()}")
