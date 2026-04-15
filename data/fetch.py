"""
Data acquisition module: downloads S&P 500 sector ETF prices via yfinance,
with local CSV caching to avoid redundant downloads.
"""

import os
import pandas as pd
import yfinance as yf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def fetch_prices(force_download: bool = False) -> pd.DataFrame:
    """
    Download adjusted close prices for all sector ETFs, S&P 500, and VIX.

    Returns a DataFrame with DatetimeIndex and one column per ticker.
    Caches to cache/raw_prices.csv for subsequent runs.
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(config.CACHE_DIR, "raw_prices.csv")

    if os.path.exists(cache_path) and not force_download:
        print(f"Loading cached prices from {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    all_tickers = config.SECTOR_TICKERS + [config.MARKET_TICKER, config.VIX_TICKER]
    print(f"Downloading prices for {len(all_tickers)} tickers "
          f"from {config.DATA_START_DATE} to {config.DATA_END_DATE}...")

    raw = yf.download(
        tickers=all_tickers,
        start=config.DATA_START_DATE,
        end=config.DATA_END_DATE,
        auto_adjust=True,
        progress=True,
    )

    # yf.download returns MultiIndex columns (Price, Ticker) — extract Close
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw

    # Rename market/VIX columns for clarity
    rename_map = {config.MARKET_TICKER: "SPX", config.VIX_TICKER: "VIX"}
    prices = prices.rename(columns=rename_map)

    # Sort by date
    prices = prices.sort_index()

    # Forward-fill then back-fill small gaps (holidays misaligned across tickers)
    prices = prices.ffill().bfill()

    print(f"Downloaded {len(prices)} trading days, "
          f"date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Tickers with data: {list(prices.columns)}")

    # Show availability for newer ETFs
    for ticker in ["XLC", "XLRE"]:
        if ticker in prices.columns:
            first_valid = prices[ticker].first_valid_index()
            print(f"  {ticker} first available: {first_valid.date() if first_valid else 'N/A'}")

    prices.to_csv(cache_path)
    print(f"Cached prices to {cache_path}")

    return prices


if __name__ == "__main__":
    df = fetch_prices(force_download=True)
    print(f"\nShape: {df.shape}")
    print(df.head())
