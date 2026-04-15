"""
Summary statistics table replicating Table 2 from the paper.
"""

import os
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from backtest.metrics import compute_all_metrics


def create_comparison_table(
    drl_yearly_results: dict,
    mvo_yearly_results: dict,
) -> pd.DataFrame:
    """
    Create Table 2 from the paper: averaged statistics across all backtest years.

    Args:
        drl_yearly_results: {year: BacktestResult} for DRL
        mvo_yearly_results: {year: BacktestResult} for MVO

    Returns:
        DataFrame with metrics as rows, DRL/MVO as columns.
    """
    drl_metrics_list = []
    mvo_metrics_list = []
    drl_max_dd = 0.0
    mvo_max_dd = 0.0

    for year in sorted(drl_yearly_results.keys()):
        res = drl_yearly_results[year]
        m = compute_all_metrics(res.daily_returns, res.daily_values, res.daily_weights)
        drl_metrics_list.append(m)
        drl_max_dd = min(drl_max_dd, m["Max drawdown"])

    for year in sorted(mvo_yearly_results.keys()):
        res = mvo_yearly_results[year]
        m = compute_all_metrics(res.daily_returns, res.daily_values, res.daily_weights)
        mvo_metrics_list.append(m)
        mvo_max_dd = min(mvo_max_dd, m["Max drawdown"])

    if not drl_metrics_list or not mvo_metrics_list:
        print("No results to tabulate")
        return pd.DataFrame()

    # Average across years (except Max Drawdown which is the worst seen)
    metric_names = list(drl_metrics_list[0].keys())
    drl_avg = {}
    mvo_avg = {}

    for name in metric_names:
        if name == "Max drawdown":
            drl_avg[name] = drl_max_dd
            mvo_avg[name] = mvo_max_dd
        else:
            drl_vals = [m[name] for m in drl_metrics_list if name in m]
            mvo_vals = [m[name] for m in mvo_metrics_list if name in m]
            drl_avg[name] = sum(drl_vals) / len(drl_vals) if drl_vals else 0
            mvo_avg[name] = sum(mvo_vals) / len(mvo_vals) if mvo_vals else 0

    table = pd.DataFrame({"DRL": drl_avg, "MVO": mvo_avg})
    return table


def print_table(table: pd.DataFrame, save: bool = True):
    """Pretty-print and optionally save the comparison table."""
    print("\n" + "=" * 60)
    print("Table 2: Performance Statistics (averaged across backtests)")
    print("=" * 60)

    for metric in table.index:
        drl_val = table.loc[metric, "DRL"]
        mvo_val = table.loc[metric, "MVO"]
        print(f"  {metric:<25s}  {drl_val:>10.4f}  {mvo_val:>10.4f}")

    print("=" * 60)

    if save:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        path = os.path.join(config.RESULTS_DIR, "table2_comparison.csv")
        table.to_csv(path)
        print(f"Saved to {path}")
