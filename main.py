"""
CLI entrypoint for the DRL vs MVO Portfolio Allocation project.

Usage:
    python main.py fetch              Download and cache market data
    python main.py features           Compute features from cached data
    python main.py train-ppo          Train PPO agents across sliding windows
    python main.py run-mvo            Run MVO strategy across all backtest years
    python main.py backtest           Run backtests for both strategies
    python main.py evaluate           Compute metrics and generate figures
    python main.py all                Run entire pipeline end-to-end

Flags:
    --debug                           Use reduced hyperparameters for fast testing
    --subproc                         Use SubprocVecEnv for PPO (requires __main__ guard)
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF logging (0=all, 3=errors only)

import argparse
import sys
import pickle
import numpy as np
import pandas as pd

import config
from data.fetch import fetch_prices
from data.features import compute_features
from agents.mvo_agent import run_mvo_strategy
from agents.ppo_agent import train_all_windows, _get_window_dates
from backtest.backtester import Backtester
from backtest.metrics import compute_all_metrics
from evaluation.visualizations import (
    plot_backtest_comparison,
    plot_strategy_analysis,
    plot_cumulative_returns,
)
from evaluation.tables import create_comparison_table, print_table


def cmd_fetch(args):
    """Download and cache market data."""
    print("=" * 60)
    print("Step 1: Fetching market data")
    print("=" * 60)
    prices = fetch_prices(force_download=True)
    print(f"Done. Shape: {prices.shape}")


def cmd_features(args):
    """Compute features from cached data."""
    print("=" * 60)
    print("Step 2: Computing features")
    print("=" * 60)
    dataset = compute_features()
    # Cache the dataset for later use
    cache_path = os.path.join(config.CACHE_DIR, "feature_dataset.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Feature dataset cached to {cache_path}")


def _load_dataset():
    """Load cached feature dataset, or compute if not cached."""
    cache_path = os.path.join(config.CACHE_DIR, "feature_dataset.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print("Feature dataset not cached. Computing...")
    dataset = compute_features()
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(dataset, f)
    return dataset


def _get_backtest_years(debug: bool = False) -> list:
    """Get list of backtest test years based on window configuration."""
    years = []
    w = 0
    max_test_year = config.LAST_POSSIBLE_TEST_YEAR
    while True:
        _, _, _, _, test_start, _ = _get_window_dates(w)
        test_year = int(test_start[:4])
        if test_year > max_test_year:
            break
        years.append(test_year)
        w += 1
    if debug:
        years = years[:config.DEBUG_N_WINDOWS]
    return years


def cmd_train_ppo(args):
    """Train PPO agents across sliding windows."""
    print("=" * 60)
    print("Step 3a: Training PPO agents")
    print("=" * 60)
    dataset = _load_dataset()
    results = train_all_windows(
        dataset,
        debug=args.debug,
        use_subproc=args.subproc,
    )
    # Save test weights
    weights_path = os.path.join(config.RESULTS_DIR, "ppo_test_weights.pkl")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(weights_path, "wb") as f:
        pickle.dump(results["test_weights"], f)
    print(f"PPO test weights saved to {weights_path}")


def cmd_run_mvo(args):
    """Run MVO strategy across all backtest years."""
    print("=" * 60)
    print("Step 3b: Running MVO strategy")
    print("=" * 60)
    dataset = _load_dataset()
    years = _get_backtest_years(args.debug)

    mvo_weights = {}
    for year in years:
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        print(f"\nMVO backtest year {year}...")
        weights = run_mvo_strategy(
            dataset.log_returns, dataset.prices,
            start_date=start, end_date=end,
        )
        mvo_weights[year] = weights
        print(f"  Generated {len(weights)} days of weights")

    weights_path = os.path.join(config.RESULTS_DIR, "mvo_test_weights.pkl")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(weights_path, "wb") as f:
        pickle.dump(mvo_weights, f)
    print(f"\nMVO test weights saved to {weights_path}")


def cmd_backtest(args):
    """Run backtests for both strategies."""
    print("=" * 60)
    print("Step 4: Running backtests")
    print("=" * 60)
    dataset = _load_dataset()

    # Load saved weights
    ppo_weights_path = os.path.join(config.RESULTS_DIR, "ppo_test_weights.pkl")
    mvo_weights_path = os.path.join(config.RESULTS_DIR, "mvo_test_weights.pkl")

    ppo_yearly_results = {}
    mvo_yearly_results = {}

    backtester = Backtester(dataset.prices)

    # PPO backtests
    if os.path.exists(ppo_weights_path):
        with open(ppo_weights_path, "rb") as f:
            ppo_weights = pickle.load(f)
        print("\nRunning PPO backtests...")
        for year, weights in sorted(ppo_weights.items()):
            print(f"  Year {year}: {len(weights)} days")
            result = backtester.run(weights, strategy_name=f"DRL_{year}")
            ppo_yearly_results[year] = result
    else:
        print(f"PPO weights not found at {ppo_weights_path}. Run 'train-ppo' first.")

    # MVO backtests
    if os.path.exists(mvo_weights_path):
        with open(mvo_weights_path, "rb") as f:
            mvo_weights = pickle.load(f)
        print("\nRunning MVO backtests...")
        for year, weights in sorted(mvo_weights.items()):
            print(f"  Year {year}: {len(weights)} days")
            result = backtester.run(weights, strategy_name=f"MVO_{year}")
            mvo_yearly_results[year] = result
    else:
        print(f"MVO weights not found at {mvo_weights_path}. Run 'run-mvo' first.")

    # Save backtest results
    results_path = os.path.join(config.RESULTS_DIR, "backtest_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump({"drl": ppo_yearly_results, "mvo": mvo_yearly_results}, f)
    print(f"\nBacktest results saved to {results_path}")


def cmd_evaluate(args):
    """Compute metrics, generate figures and tables."""
    print("=" * 60)
    print("Step 5: Evaluation")
    print("=" * 60)

    results_path = os.path.join(config.RESULTS_DIR, "backtest_results.pkl")
    if not os.path.exists(results_path):
        print(f"Backtest results not found at {results_path}. Run 'backtest' first.")
        return

    with open(results_path, "rb") as f:
        all_results = pickle.load(f)

    drl_results = all_results.get("drl", {})
    mvo_results = all_results.get("mvo", {})

    if not drl_results and not mvo_results:
        print("No results to evaluate.")
        return

    # Compute yearly metrics
    drl_yearly_metrics = {}
    mvo_yearly_metrics = {}

    for year, res in sorted(drl_results.items()):
        drl_yearly_metrics[year] = compute_all_metrics(
            res.daily_returns, res.daily_values, res.daily_weights
        )

    for year, res in sorted(mvo_results.items()):
        mvo_yearly_metrics[year] = compute_all_metrics(
            res.daily_returns, res.daily_values, res.daily_weights
        )

    # Table 2
    if drl_results and mvo_results:
        table = create_comparison_table(drl_results, mvo_results)
        print_table(table)

    # Figure 2: Backtest comparison
    if drl_yearly_metrics and mvo_yearly_metrics:
        print("\nGenerating Figure 2: Backtest comparison...")
        plot_backtest_comparison(drl_yearly_metrics, mvo_yearly_metrics)

    # Figures 3 & 4: Strategy-specific analysis
    for name, results_dict in [("DRL", drl_results), ("MVO", mvo_results)]:
        if results_dict:
            # Concatenate all yearly returns for aggregate analysis
            all_returns = pd.concat(
                [r.daily_returns for r in results_dict.values()]
            ).sort_index()
            print(f"\nGenerating {name} strategy analysis...")
            plot_strategy_analysis(all_returns, strategy_name=name)

    # Cumulative returns comparison
    if drl_results and mvo_results:
        drl_values = pd.concat([r.daily_values for r in drl_results.values()]).sort_index()
        mvo_values = pd.concat([r.daily_values for r in mvo_results.values()]).sort_index()
        common = drl_values.index.intersection(mvo_values.index)
        if len(common) > 0:
            print("\nGenerating cumulative returns comparison...")
            plot_cumulative_returns(drl_values.loc[common], mvo_values.loc[common])

    print("\nEvaluation complete! Results saved to:", config.RESULTS_DIR)


def cmd_all(args):
    """Run entire pipeline end-to-end."""
    cmd_fetch(args)
    cmd_features(args)
    cmd_run_mvo(args)
    cmd_train_ppo(args)
    cmd_backtest(args)
    cmd_evaluate(args)


def main():
    parser = argparse.ArgumentParser(
        description="DRL vs MVO Portfolio Allocation - Paper Replication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Use reduced hyperparameters for fast testing",
    )
    parser.add_argument(
        "--subproc", action="store_true",
        help="Use SubprocVecEnv for PPO training (faster but requires __main__ guard)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.add_parser("fetch", help="Download and cache market data")
    subparsers.add_parser("features", help="Compute features from cached data")
    subparsers.add_parser("train-ppo", help="Train PPO agents")
    subparsers.add_parser("run-mvo", help="Run MVO strategy")
    subparsers.add_parser("backtest", help="Run backtests")
    subparsers.add_parser("evaluate", help="Compute metrics and generate figures")
    subparsers.add_parser("all", help="Run entire pipeline")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "fetch": cmd_fetch,
        "features": cmd_features,
        "train-ppo": cmd_train_ppo,
        "run-mvo": cmd_run_mvo,
        "backtest": cmd_backtest,
        "evaluate": cmd_evaluate,
        "all": cmd_all,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
