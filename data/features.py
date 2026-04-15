"""
Feature engineering module: computes log returns, volatility metrics,
and expanding-window standardized features for the RL environment.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from data.fetch import fetch_prices


class FeatureDataset:
    """Container for all preprocessed data needed by the environment."""

    def __init__(self, prices: pd.DataFrame, log_returns: pd.DataFrame,
                 vol_features: pd.DataFrame, sector_tickers: list):
        self.prices = prices              # Raw prices (sectors only)
        self.log_returns = log_returns    # Daily log returns (sectors only)
        self.vol_features = vol_features  # Standardized vol20, vol_ratio, VIX
        self.sector_tickers = sector_tickers
        self.dates = log_returns.index

    def get_date_range_indices(self, start_date: str, end_date: str):
        """Return integer indices for a date range (inclusive)."""
        mask = (self.dates >= start_date) & (self.dates <= end_date)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            raise ValueError(f"No data found between {start_date} and {end_date}")
        return indices[0], indices[-1]


def compute_features(prices_df: pd.DataFrame = None) -> FeatureDataset:
    """
    Compute all features from raw prices.

    Returns a FeatureDataset with:
    - log_returns: daily log returns for sector ETFs
    - vol_features: expanding-window standardized vol20, vol20/vol60, VIX
    - prices: raw sector prices (for whole-share calculations)
    """
    if prices_df is None:
        prices_df = fetch_prices()

    # Identify available sector tickers
    sector_tickers = [t for t in config.SECTOR_TICKERS if t in prices_df.columns]

    # ── Log returns for sector ETFs ──────────────────────────────────────────
    sector_prices = prices_df[sector_tickers].copy()
    log_returns = np.log(sector_prices / sector_prices.shift(1))

    # For tickers that don't exist early (XLC, XLRE), set NaN returns to 0
    log_returns = log_returns.fillna(0.0)

    # ── S&P 500 volatility features ─────────────────────────────────────────
    spx_col = "SPX" if "SPX" in prices_df.columns else config.MARKET_TICKER
    vix_col = "VIX" if "VIX" in prices_df.columns else config.VIX_TICKER

    spx_log_ret = np.log(prices_df[spx_col] / prices_df[spx_col].shift(1))

    vol20 = spx_log_ret.rolling(window=config.VOL_SHORT_WINDOW).std()
    vol60 = spx_log_ret.rolling(window=config.VOL_LONG_WINDOW).std()
    vol_ratio = vol20 / vol60

    vix = prices_df[vix_col].copy()

    # ── Expanding-window standardization (prevents look-ahead bias) ──────────
    # At time t, standardize using mean/std of all data from start to t-1
    vol_features = pd.DataFrame(index=prices_df.index)

    for name, series in [("vol20", vol20), ("vol_ratio", vol_ratio), ("vix", vix)]:
        expanding_mean = series.expanding(min_periods=2).mean().shift(1)
        expanding_std = series.expanding(min_periods=2).std().shift(1)
        # Avoid division by zero
        expanding_std = expanding_std.replace(0, np.nan)
        vol_features[name] = (series - expanding_mean) / expanding_std

    # Fill NaN values at the start with 0 (insufficient history)
    vol_features = vol_features.fillna(0.0)

    # ── Align all DataFrames to the same date index ──────────────────────────
    # Drop rows where we don't have enough data (first ~60 days)
    valid_start = max(
        log_returns.first_valid_index(),
        vol_features.first_valid_index(),
    )
    # Need at least LOOKBACK days of history
    min_start_idx = config.LOOKBACK + config.VOL_LONG_WINDOW
    if min_start_idx < len(log_returns):
        valid_start = max(valid_start, log_returns.index[min_start_idx])

    log_returns = log_returns.loc[valid_start:]
    vol_features = vol_features.loc[valid_start:]
    sector_prices = sector_prices.loc[valid_start:]

    # Ensure all share the same index
    common_idx = log_returns.index.intersection(vol_features.index).intersection(sector_prices.index)
    log_returns = log_returns.loc[common_idx]
    vol_features = vol_features.loc[common_idx]
    sector_prices = sector_prices.loc[common_idx]

    print(f"Feature dataset: {len(common_idx)} trading days, "
          f"{len(sector_tickers)} sectors")
    print(f"  Date range: {common_idx[0].date()} to {common_idx[-1].date()}")

    return FeatureDataset(
        prices=sector_prices,
        log_returns=log_returns,
        vol_features=vol_features,
        sector_tickers=sector_tickers,
    )


if __name__ == "__main__":
    dataset = compute_features()
    print(f"\nLog returns shape: {dataset.log_returns.shape}")
    print(f"Vol features shape: {dataset.vol_features.shape}")
    print(f"Prices shape: {dataset.prices.shape}")
    print(f"\nLog returns sample:\n{dataset.log_returns.tail()}")
    print(f"\nVol features sample:\n{dataset.vol_features.tail()}")
