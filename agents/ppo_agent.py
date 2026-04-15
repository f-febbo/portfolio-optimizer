"""
PPO agent with sliding-window training scheme.

Implements the training pipeline from the paper:
- 10+ sliding windows (5yr train, 1yr validation, 1yr test)
- 5 agents per window with different seeds
- Best agent (highest validation Sharpe) seeds next window
- StableBaselines3 PPO with SubprocVecEnv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from env.portfolio_env import PortfolioEnv, make_env
from data.features import FeatureDataset

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


def linear_schedule(lr_start: float = None, lr_end: float = None):
    """Linear learning rate annealing from lr_start to lr_end."""
    lr_start = lr_start or config.PPO_LR_START
    lr_end = lr_end or config.PPO_LR_END

    def schedule(progress_remaining: float) -> float:
        return lr_end + progress_remaining * (lr_start - lr_end)

    return schedule


class RewardLoggingCallback(BaseCallback):
    """Logs mean episode reward during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


class ProgressSpinnerCallback(BaseCallback):
    """Shows a spinning progress indicator in the terminal during training."""

    _SPINNER = ["|", "/", "-", "\\"]

    def __init__(self, total_timesteps: int, update_interval: int = 5_000, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.update_interval = update_interval
        self._spin_idx = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_interval == 0:
            char = self._SPINNER[self._spin_idx % len(self._SPINNER)]
            pct = 100.0 * self.num_timesteps / self.total_timesteps
            print(
                f"\r    {char}  {self.num_timesteps:>10,} / {self.total_timesteps:,} steps  ({pct:5.1f}%)",
                end="",
                flush=True,
            )
            self._spin_idx += 1
        return True

    def _on_training_end(self) -> None:
        print(
            f"\r    Done  {self.total_timesteps:>10,} / {self.total_timesteps:,} steps (100.0%)",
            flush=True,
        )


def _get_window_dates(window_idx: int):
    """
    Compute the train/val/test date ranges for a given sliding window.

    Returns (train_start, train_end, val_start, val_end, test_start, test_end)
    as string dates "YYYY-01-01" / "YYYY-12-31".
    """
    base_year = config.FIRST_TRAIN_START_YEAR + window_idx * config.WINDOW_SHIFT_YEARS

    train_start = f"{base_year}-01-01"
    train_end = f"{base_year + config.WINDOW_TRAIN_YEARS - 1}-12-31"
    val_start = f"{base_year + config.WINDOW_TRAIN_YEARS}-01-01"
    val_end = f"{base_year + config.WINDOW_TRAIN_YEARS + config.WINDOW_VAL_YEARS - 1}-12-31"
    test_start = f"{base_year + config.WINDOW_TRAIN_YEARS + config.WINDOW_VAL_YEARS}-01-01"
    test_end = f"{base_year + config.WINDOW_TRAIN_YEARS + config.WINDOW_VAL_YEARS + config.WINDOW_TEST_YEARS - 1}-12-31"

    return train_start, train_end, val_start, val_end, test_start, test_end


def _date_to_idx(dataset: FeatureDataset, date_str: str, direction: str = "after"):
    """Find the nearest index for a date string in the dataset."""
    dates = dataset.dates
    target = pd.Timestamp(date_str)

    if direction == "after":
        mask = dates >= target
        if mask.any():
            return np.where(mask)[0][0]
        return len(dates) - 1
    else:
        mask = dates <= target
        if mask.any():
            return np.where(mask)[0][-1]
        return 0


def evaluate_agent(model, dataset: FeatureDataset, start_date: str, end_date: str) -> float:
    """
    Evaluate a trained PPO agent over a date range.
    Returns the Sharpe ratio of the portfolio.
    """
    start_idx = _date_to_idx(dataset, start_date, "after")
    end_idx = _date_to_idx(dataset, end_date, "before")

    # Ensure enough lookback
    start_idx = max(start_idx, config.LOOKBACK)

    env = PortfolioEnv(
        log_returns=dataset.log_returns.values,
        vol_features=dataset.vol_features.values,
        prices=dataset.prices.values,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    obs, info = env.reset()
    daily_returns = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        daily_returns.append(info.get("portfolio_return", 0.0))
        done = terminated or truncated

    if len(daily_returns) < 10:
        return -np.inf

    dr = np.array(daily_returns)
    if dr.std() == 0:
        return 0.0
    return float((dr.mean() / dr.std()) * np.sqrt(252))


def get_ppo_weights(model, dataset: FeatureDataset, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Run a trained PPO agent and collect daily target weights.

    Returns DataFrame with DatetimeIndex, columns = sector tickers.
    """
    start_idx = _date_to_idx(dataset, start_date, "after")
    end_idx = _date_to_idx(dataset, end_date, "before")
    start_idx = max(start_idx, config.LOOKBACK)

    env = PortfolioEnv(
        log_returns=dataset.log_returns.values,
        vol_features=dataset.vol_features.values,
        prices=dataset.prices.values,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    obs, info = env.reset()
    weight_records = []
    dates = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)

        # Compute softmax weights from action (same as env does internally)
        e_x = np.exp(action - np.max(action))
        target_w = e_x / e_x.sum()

        dates.append(dataset.dates[env.current_step])
        weight_records.append(target_w)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    weights_df = pd.DataFrame(
        weight_records,
        index=pd.DatetimeIndex(dates),
        columns=dataset.sector_tickers,
    )
    return weights_df


def train_all_windows(
    dataset: FeatureDataset,
    debug: bool = False,
    use_subproc: bool = False,
) -> dict:
    """
    Train PPO agents across all sliding windows.

    Args:
        dataset: Preprocessed feature dataset.
        debug: If True, use reduced hyperparameters for fast iteration.
        use_subproc: If True, use SubprocVecEnv (requires __name__ == "__main__").
                     Otherwise uses DummyVecEnv.

    Returns:
        Dictionary with window results, saved model paths, and test weights.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Determine number of windows
    max_test_year = config.LAST_POSSIBLE_TEST_YEAR
    n_windows = 0
    while True:
        _, _, _, _, test_start, test_end = _get_window_dates(n_windows)
        test_year = int(test_start[:4])
        if test_year > max_test_year:
            break
        n_windows += 1

    if debug:
        n_windows = min(n_windows, config.DEBUG_N_WINDOWS)

    print(f"Training across {n_windows} windows")

    # Training config
    total_timesteps = config.DEBUG_TOTAL_TIMESTEPS if debug else config.PPO_TOTAL_TIMESTEPS
    n_envs = config.DEBUG_N_ENVS if debug else config.PPO_N_ENVS
    n_seeds = config.DEBUG_N_SEEDS if debug else config.PPO_N_SEEDS_PER_WINDOW

    VecEnvClass = SubprocVecEnv if use_subproc else DummyVecEnv

    results = {
        "windows": [],
        "test_weights": {},
    }

    best_seed = 42  # Initial seed for first window

    for w in range(n_windows):
        train_start, train_end, val_start, val_end, test_start, test_end = _get_window_dates(w)
        test_year = int(test_start[:4])

        print(f"\n{'='*60}")
        print(f"Window {w+1}/{n_windows}: Train [{train_start} - {train_end}], "
              f"Val [{val_start} - {val_end}], Test [{test_start} - {test_end}]")
        print(f"{'='*60}")

        # Get data indices
        train_start_idx = max(_date_to_idx(dataset, train_start, "after"), config.LOOKBACK)
        train_end_idx = _date_to_idx(dataset, train_end, "before")
        val_start_idx = max(_date_to_idx(dataset, val_start, "after"), config.LOOKBACK)
        val_end_idx = _date_to_idx(dataset, val_end, "before")

        if train_end_idx <= train_start_idx or val_end_idx <= val_start_idx:
            print(f"  Skipping window {w+1}: insufficient data")
            continue

        # Generate seeds from best previous seed
        rng = np.random.default_rng(best_seed)
        seeds = rng.integers(0, 2**31, size=n_seeds).tolist()

        best_val_sharpe = -np.inf
        best_model = None
        best_model_seed = seeds[0]

        for s_idx, seed in enumerate(seeds):
            print(f"\n  Seed {s_idx+1}/{n_seeds} (seed={seed})")

            # Create vectorized environments
            env_fns = [
                make_env(
                    log_returns=dataset.log_returns.values,
                    vol_features=dataset.vol_features.values,
                    prices=dataset.prices.values,
                    start_idx=train_start_idx,
                    end_idx=train_end_idx,
                    seed=seed + i,
                )
                for i in range(n_envs)
            ]

            vec_env = VecEnvClass(env_fns)

            try:
                model = PPO(
                    "MlpPolicy",
                    vec_env,
                    n_steps=config.PPO_N_STEPS // n_envs if debug else config.PPO_N_STEPS,
                    batch_size=config.PPO_BATCH_SIZE // (config.PPO_N_ENVS // n_envs) if debug else config.PPO_BATCH_SIZE,
                    n_epochs=config.PPO_N_EPOCHS,
                    gamma=config.PPO_GAMMA,
                    gae_lambda=config.PPO_GAE_LAMBDA,
                    clip_range=config.PPO_CLIP_RANGE,
                    learning_rate=linear_schedule(),
                    policy_kwargs={
                        "net_arch": config.PPO_NET_ARCH,
                        "activation_fn": torch.nn.Tanh,
                        "log_std_init": config.PPO_LOG_STD_INIT,
                    },
                    seed=seed,
                    verbose=0,
                )

                callback = [
                    RewardLoggingCallback(),
                    ProgressSpinnerCallback(total_timesteps),
                ]
                model.learn(total_timesteps=total_timesteps, callback=callback)

                # Evaluate on validation set
                val_sharpe = evaluate_agent(model, dataset, val_start, val_end)
                print(f"    Validation Sharpe: {val_sharpe:.4f}")

                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    best_model = model
                    best_model_seed = seed

            except Exception as e:
                print(f"    Training failed: {e}")
            finally:
                vec_env.close()

        if best_model is None:
            print(f"  No successful training for window {w+1}")
            continue

        # Save best model
        model_path = os.path.join(config.MODELS_DIR, f"ppo_window_{w+1}.zip")
        best_model.save(model_path)
        print(f"\n  Best seed: {best_model_seed}, Val Sharpe: {best_val_sharpe:.4f}")
        print(f"  Model saved to {model_path}")

        # Get test weights
        test_weights = get_ppo_weights(best_model, dataset, test_start, test_end)
        results["test_weights"][test_year] = test_weights

        results["windows"].append({
            "window": w + 1,
            "train_period": f"{train_start} to {train_end}",
            "val_period": f"{val_start} to {val_end}",
            "test_period": f"{test_start} to {test_end}",
            "best_seed": int(best_model_seed),
            "val_sharpe": float(best_val_sharpe),
            "model_path": model_path,
        })

        # Use best seed for next window
        best_seed = best_model_seed

    # Save training metadata
    meta_path = os.path.join(config.RESULTS_DIR, "ppo_training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(results["windows"], f, indent=2)
    print(f"\nTraining metadata saved to {meta_path}")

    return results
