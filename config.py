"""
Central configuration for the DRL vs MVO Portfolio Allocation project.
All hyperparameters from Sood et al. (2023) are defined here.
"""

import os
from datetime import date

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# ── Tickers ────────────────────────────────────────────────────────────────────
SECTOR_TICKERS = [
    "XLB",   # Materials
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLV",   # Health Care
    "XLF",   # Financials
    "XLK",   # Information Technology
    "XLC",   # Communication Services (June 2018+)
    "XLU",   # Utilities
    "XLRE",  # Real Estate (Oct 2015+)
    "XLE",   # Energy
]
MARKET_TICKER = "^GSPC"  # S&P 500 for volatility features
VIX_TICKER = "^VIX"

N_SECTORS = len(SECTOR_TICKERS)

# ── Date range ─────────────────────────────────────────────────────────────────
# Fetch from 2005 to have enough lookback buffer for 2006 features
DATA_START_DATE = "2005-01-01"
DATA_END_DATE = date.today().isoformat()

# ── Feature parameters ─────────────────────────────────────────────────────────
LOOKBACK = 60          # T = 60 day observation window
VOL_SHORT_WINDOW = 20  # 20-day rolling std
VOL_LONG_WINDOW = 60   # 60-day rolling std

# ── Environment parameters ─────────────────────────────────────────────────────
INITIAL_CASH = 100_000
ETA = 1.0 / 252.0     # Differential Sharpe Ratio decay (~1 trading year)

# ── PPO hyperparameters (Table 1 from paper) ───────────────────────────────────
PPO_TOTAL_TIMESTEPS = 7_500_000
PPO_N_ENVS = 10
PPO_N_STEPS = 756      # 252 * 3
PPO_BATCH_SIZE = 1260   # 252 * 5
PPO_N_EPOCHS = 16
PPO_GAMMA = 0.9
PPO_GAE_LAMBDA = 0.9
PPO_CLIP_RANGE = 0.25
PPO_LR_START = 3e-4
PPO_LR_END = 1e-5
PPO_NET_ARCH = [64, 64]
PPO_LOG_STD_INIT = -1.0
PPO_N_SEEDS_PER_WINDOW = 5

# ── Sliding window configuration ──────────────────────────────────────────────
WINDOW_TRAIN_YEARS = 5
WINDOW_VAL_YEARS = 1    # Burn-in / validation year
WINDOW_TEST_YEARS = 1
WINDOW_SHIFT_YEARS = 1

# First training window: [2006, 2011), val=2011, test=2012
FIRST_TRAIN_START_YEAR = 2006
# Windows continue until test year reaches present
LAST_POSSIBLE_TEST_YEAR = date.today().year - 1  # Most recent complete year

# ── Debug mode (fast iteration) ────────────────────────────────────────────────
DEBUG_TOTAL_TIMESTEPS = 100_000
DEBUG_N_ENVS = 2
DEBUG_N_SEEDS = 1
DEBUG_N_WINDOWS = 2
