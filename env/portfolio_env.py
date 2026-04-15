"""
Custom Gymnasium environment for portfolio allocation via market replay.

The environment simulates daily trading on S&P 500 sector ETFs.
At each timestep, the agent outputs portfolio weights, which are
converted to whole-share positions. The reward is the Differential
Sharpe Ratio of the resulting portfolio return.
"""

import numpy as np
import gymnasium
from gymnasium import spaces

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from env.reward import DifferentialSharpeRatio


class PortfolioEnv(gymnasium.Env):
    """
    Portfolio allocation environment with market data replay.

    Observation (flattened vector):
      - Current portfolio weights: (n_assets + 1,) for sectors + cash
      - Log returns lookback: (n_assets, LOOKBACK) for each sector
      - Volatility features lookback: (3, LOOKBACK) for vol20, vol_ratio, VIX

    Action:
      - Continuous vector of size n_assets, passed through softmax to get
        target sector weights (long-only, sum-to-1).

    Reward:
      - Differential Sharpe Ratio of daily portfolio return.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        log_returns: np.ndarray,
        vol_features: np.ndarray,
        prices: np.ndarray,
        start_idx: int,
        end_idx: int,
        initial_cash: float = None,
        lookback: int = None,
        eta: float = None,
    ):
        """
        Args:
            log_returns: Array of shape (total_days, n_assets) with daily log returns.
            vol_features: Array of shape (total_days, 3) with standardized vol features.
            prices: Array of shape (total_days, n_assets) with daily prices.
            start_idx: First index in the data to use (must be >= lookback).
            end_idx: Last index in the data to use (inclusive).
            initial_cash: Starting portfolio value in dollars.
            lookback: Number of days of history in the observation.
            eta: Decay parameter for Differential Sharpe Ratio.
        """
        super().__init__()

        self.log_returns = log_returns
        self.vol_features = vol_features
        self.prices = prices
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.initial_cash = initial_cash or config.INITIAL_CASH
        self.lookback = lookback or config.LOOKBACK
        self.n_assets = log_returns.shape[1]

        eta = eta or config.ETA
        self.dsr = DifferentialSharpeRatio(eta=eta)

        # Action space: n_assets continuous values -> softmax -> weights
        # SB3 requires finite bounds; we use [-5, 5] since softmax saturates beyond that
        self.action_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(self.n_assets,), dtype=np.float32,
        )

        # Observation space: flattened vector
        # weights: (n_assets + 1) for sectors + cash
        # log_returns lookback: n_assets * lookback
        # vol_features lookback: 3 * lookback
        self.obs_dim = (
            (self.n_assets + 1)
            + self.n_assets * self.lookback
            + 3 * self.lookback
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32,
        )

        # State variables (set on reset)
        self.current_step = None
        self.cash = None
        self.shares = None
        self.portfolio_value = None
        self.weights = None

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _get_portfolio_value(self, step: int) -> float:
        """Calculate portfolio value at a given step."""
        current_prices = self.prices[step]
        return float(np.sum(self.shares * current_prices) + self.cash)

    def _rebalance(self, target_weights: np.ndarray, step: int):
        """
        Convert target weights to whole-share positions.
        Remainder goes to cash.
        """
        current_prices = self.prices[step]
        port_val = self._get_portfolio_value(step)

        # Allocate to whole shares
        target_dollars = target_weights * port_val
        new_shares = np.floor(target_dollars / np.maximum(current_prices, 1e-8)).astype(int)

        # Ensure we don't allocate more than portfolio value
        total_allocated = np.sum(new_shares * current_prices)
        new_cash = port_val - total_allocated

        # Safety: if negative cash due to rounding, reduce shares of largest position
        while new_cash < 0:
            largest_pos = np.argmax(new_shares)
            if new_shares[largest_pos] > 0:
                new_shares[largest_pos] -= 1
                total_allocated = np.sum(new_shares * current_prices)
                new_cash = port_val - total_allocated
            else:
                break

        self.shares = new_shares
        self.cash = new_cash
        self.portfolio_value = port_val

        # Compute actual weights after whole-share rounding
        if port_val > 0:
            asset_values = new_shares * current_prices
            self.weights = np.append(asset_values / port_val, new_cash / port_val)
        else:
            self.weights = np.zeros(self.n_assets + 1)
            self.weights[-1] = 1.0  # All cash

    def _get_obs(self) -> np.ndarray:
        """Construct the observation vector for the current step."""
        t = self.current_step

        # 1. Current portfolio weights (n_assets + 1): sectors + cash
        w = self.weights.copy()

        # 2. Log returns lookback: (n_assets, lookback) flattened
        ret_start = t - self.lookback
        ret_end = t
        returns_window = self.log_returns[ret_start:ret_end]  # (lookback, n_assets)
        returns_flat = returns_window.T.flatten()  # (n_assets * lookback,)

        # 3. Volatility features lookback: (3, lookback) flattened
        vol_window = self.vol_features[ret_start:ret_end]  # (lookback, 3)
        vol_flat = vol_window.T.flatten()  # (3 * lookback,)

        obs = np.concatenate([w, returns_flat, vol_flat]).astype(np.float32)

        # Replace any NaN/inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs

    def reset(self, seed=None, options=None):
        """Reset the environment to the beginning of the episode."""
        super().reset(seed=seed)

        self.current_step = self.start_idx
        self.cash = float(self.initial_cash)
        self.shares = np.zeros(self.n_assets, dtype=int)
        self.portfolio_value = self.initial_cash

        # Initial weights: 100% cash
        self.weights = np.zeros(self.n_assets + 1, dtype=np.float64)
        self.weights[-1] = 1.0  # Cash weight

        self.dsr.reset()

        obs = self._get_obs()
        info = {"portfolio_value": self.portfolio_value, "step": self.current_step}

        return obs, info

    def step(self, action: np.ndarray):
        """
        Execute one trading day.

        1. Apply softmax to get target weights
        2. Rebalance portfolio to whole shares
        3. Advance to next day
        4. Compute portfolio return and DSR reward
        """
        # 1. Convert action to target weights via softmax
        target_weights = self._softmax(action)

        # 2. Rebalance at current prices
        self._rebalance(target_weights, self.current_step)
        port_val_before = self.portfolio_value

        # 3. Advance one day
        self.current_step += 1
        terminated = self.current_step >= self.end_idx
        truncated = False

        # 4. Calculate new portfolio value with next day's prices
        port_val_after = self._get_portfolio_value(self.current_step)

        # Portfolio simple return
        if port_val_before > 0:
            portfolio_return = (port_val_after - port_val_before) / port_val_before
        else:
            portfolio_return = 0.0

        # Update portfolio value and weights
        self.portfolio_value = port_val_after
        if port_val_after > 0:
            current_prices = self.prices[self.current_step]
            asset_values = self.shares * current_prices
            self.weights = np.append(
                asset_values / port_val_after,
                self.cash / port_val_after,
            )
        else:
            self.weights = np.zeros(self.n_assets + 1)
            self.weights[-1] = 1.0

        # 5. Compute reward
        reward = self.dsr.step(portfolio_return)

        obs = self._get_obs()
        info = {
            "portfolio_value": port_val_after,
            "portfolio_return": portfolio_return,
            "step": self.current_step,
            "weights": self.weights.copy(),
        }

        return obs, float(reward), terminated, truncated, info


def make_env(log_returns, vol_features, prices, start_idx, end_idx,
             initial_cash=None, seed=0):
    """Factory function for creating PortfolioEnv instances (for VecEnv)."""
    def _init():
        env = PortfolioEnv(
            log_returns=log_returns,
            vol_features=vol_features,
            prices=prices,
            start_idx=start_idx,
            end_idx=end_idx,
            initial_cash=initial_cash,
        )
        env.reset(seed=seed)
        return env
    return _init
