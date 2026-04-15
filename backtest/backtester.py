"""
Unified backtesting engine for both DRL and MVO strategies.

Takes daily target weights and simulates portfolio rebalancing with
whole-share constraints, tracking daily values, positions, and returns.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    daily_values: pd.Series        # Portfolio value per day
    daily_returns: pd.Series       # Simple returns per day
    daily_weights: pd.DataFrame    # Actual weights per day (after rounding)
    daily_shares: pd.DataFrame     # Share counts per day
    daily_cash: pd.Series          # Cash held per day
    strategy_name: str = ""
    start_date: str = ""
    end_date: str = ""


class Backtester:
    """
    Simulates daily portfolio rebalancing with whole-share constraints.

    Both DRL and MVO produce daily target weights. This backtester
    applies the same whole-share conversion to ensure fair comparison.
    """

    def __init__(self, prices: pd.DataFrame, initial_cash: float = None):
        """
        Args:
            prices: DataFrame of daily prices, DatetimeIndex, columns = tickers.
            initial_cash: Starting portfolio value.
        """
        self.prices = prices
        self.initial_cash = initial_cash or config.INITIAL_CASH
        self.tickers = prices.columns.tolist()

    def run(self, target_weights: pd.DataFrame, strategy_name: str = "") -> BacktestResult:
        """
        Execute the backtest.

        Args:
            target_weights: DataFrame of daily target weights.
                            DatetimeIndex, columns = tickers, values in [0, 1].
            strategy_name: Label for this strategy.

        Returns:
            BacktestResult with daily portfolio tracking data.
        """
        # Align dates
        common_dates = target_weights.index.intersection(self.prices.index)
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between weights and prices")

        target_weights = target_weights.loc[common_dates]
        prices = self.prices.loc[common_dates]

        n_days = len(common_dates)
        n_assets = len(self.tickers)

        # Storage
        values = np.zeros(n_days)
        returns = np.zeros(n_days)
        weights_actual = np.zeros((n_days, n_assets))
        shares_history = np.zeros((n_days, n_assets), dtype=int)
        cash_history = np.zeros(n_days)

        # Initialize: all cash
        cash = float(self.initial_cash)
        shares = np.zeros(n_assets, dtype=int)

        for i in range(n_days):
            day_prices = prices.iloc[i].values.astype(float)

            # Current portfolio value
            port_val = float(np.sum(shares * day_prices) + cash)

            # Target allocation
            tw = target_weights.iloc[i].values.astype(float)
            tw_sum = tw.sum()
            if tw_sum > 0:
                tw = tw / tw_sum  # Normalize to sum to 1

            # Convert to whole shares
            target_dollars = tw * port_val
            new_shares = np.floor(
                target_dollars / np.maximum(day_prices, 1e-8)
            ).astype(int)

            # Compute remaining cash
            total_allocated = float(np.sum(new_shares * day_prices))
            new_cash = port_val - total_allocated

            # Safety check
            while new_cash < 0:
                largest = np.argmax(new_shares)
                if new_shares[largest] > 0:
                    new_shares[largest] -= 1
                    total_allocated = float(np.sum(new_shares * day_prices))
                    new_cash = port_val - total_allocated
                else:
                    break

            shares = new_shares
            cash = new_cash

            # Record actual weights
            if port_val > 0:
                asset_vals = shares * day_prices
                weights_actual[i] = asset_vals / port_val
            else:
                weights_actual[i] = 0.0

            values[i] = port_val
            shares_history[i] = shares
            cash_history[i] = cash

            # Daily return
            if i > 0:
                returns[i] = (values[i] - values[i - 1]) / values[i - 1] if values[i - 1] > 0 else 0.0

        return BacktestResult(
            daily_values=pd.Series(values, index=common_dates, name="portfolio_value"),
            daily_returns=pd.Series(returns, index=common_dates, name="daily_return"),
            daily_weights=pd.DataFrame(weights_actual, index=common_dates, columns=self.tickers),
            daily_shares=pd.DataFrame(shares_history, index=common_dates, columns=self.tickers),
            daily_cash=pd.Series(cash_history, index=common_dates, name="cash"),
            strategy_name=strategy_name,
            start_date=str(common_dates[0].date()),
            end_date=str(common_dates[-1].date()),
        )
