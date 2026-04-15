"""
Visualization module replicating Figures 2, 3, and 4 from the paper.

- Figure 2: Sharpe ratio, max drawdown, avg daily weight change per year
- Figure 3: DRL monthly returns heatmap, annual returns, monthly distribution
- Figure 4: MVO monthly returns heatmap, annual returns, monthly distribution
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CLI usage
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def _ensure_results_dir():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)


def plot_backtest_comparison(
    drl_yearly_metrics: dict,
    mvo_yearly_metrics: dict,
    save: bool = True,
):
    """
    Replicate Figure 2: Three-panel comparison per year.

    Args:
        drl_yearly_metrics: {year: metrics_dict} for DRL
        mvo_yearly_metrics: {year: metrics_dict} for MVO
    """
    years = sorted(set(drl_yearly_metrics.keys()) & set(mvo_yearly_metrics.keys()))
    if not years:
        print("No overlapping years to plot")
        return

    drl_sharpe = [drl_yearly_metrics[y].get("Sharpe ratio", 0) for y in years]
    mvo_sharpe = [mvo_yearly_metrics[y].get("Sharpe ratio", 0) for y in years]
    drl_mdd = [drl_yearly_metrics[y].get("Max drawdown", 0) for y in years]
    mvo_mdd = [mvo_yearly_metrics[y].get("Max drawdown", 0) for y in years]
    drl_turnover = [drl_yearly_metrics[y].get("Portfolio turnover", 0) for y in years]
    mvo_turnover = [mvo_yearly_metrics[y].get("Portfolio turnover", 0) for y in years]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Backtest Performance: MVO vs Deep RL", fontsize=14, fontweight="bold")

    # Panel 1: Sharpe Ratio
    ax = axes[0]
    ax.plot(years, mvo_sharpe, "o-", label="MVO", color="tab:blue")
    ax.plot(years, drl_sharpe, "s-", label="DRL", color="tab:orange")
    ax.set_title("Sharpe Ratio")
    ax.set_xlabel("Backtest Year")
    ax.set_ylabel("Sharpe")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Maximum Drawdown
    ax = axes[1]
    ax.plot(years, mvo_mdd, "o-", label="MVO", color="tab:blue")
    ax.plot(years, drl_mdd, "s-", label="DRL", color="tab:orange")
    ax.set_title("Maximum Drawdown")
    ax.set_xlabel("Backtest Year")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Avg Daily Change in Portfolio Weights
    ax = axes[2]
    ax.plot(years, mvo_turnover, "o-", label="MVO", color="tab:blue")
    ax.plot(years, drl_turnover, "s-", label="DRL", color="tab:orange")
    ax.set_title("Avg. Daily Change in Portfolio Weights")
    ax.set_xlabel("Backtest Year")
    ax.set_ylabel("Δp_w")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        _ensure_results_dir()
        path = os.path.join(config.RESULTS_DIR, "figure2_backtest_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved Figure 2 to {path}")

    plt.close(fig)


def _compute_monthly_returns(daily_returns: pd.Series) -> pd.DataFrame:
    """Compute monthly returns from daily returns, organized as year x month."""
    monthly = (1 + daily_returns).resample("ME").prod() - 1
    table = pd.DataFrame(index=sorted(monthly.index.year.unique()))

    for idx in monthly.index:
        table.loc[idx.year, idx.month] = monthly.loc[idx]

    table.columns = range(1, 13)
    return table


def plot_strategy_analysis(
    daily_returns: pd.Series,
    strategy_name: str = "Strategy",
    save: bool = True,
):
    """
    Replicate Figures 3/4: Monthly returns heatmap, annual returns bar chart,
    and monthly returns distribution.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel a: Monthly returns heatmap
    ax = axes[0]
    monthly_table = _compute_monthly_returns(daily_returns)
    monthly_pct = monthly_table * 100

    sns.heatmap(
        monthly_pct,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Return (%)"},
        xticklabels=[str(m) for m in range(1, 13)],
    )
    ax.set_title(f"Monthly returns (%)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    # Panel b: Annual returns bar chart
    ax = axes[1]
    annual = (1 + daily_returns).resample("YE").prod() - 1
    years = annual.index.year
    colors = ["green" if r >= 0 else "red" for r in annual.values]
    ax.barh(years, annual.values * 100, color=colors, alpha=0.7)
    mean_ret = annual.mean() * 100
    ax.axvline(mean_ret, color="black", linestyle="--", label=f"Mean: {mean_ret:.1f}%")
    ax.set_title("Annual returns")
    ax.set_xlabel("Returns (%)")
    ax.set_ylabel("Year")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel c: Distribution of monthly returns
    ax = axes[2]
    monthly_flat = (1 + daily_returns).resample("ME").prod() - 1
    ax.hist(monthly_flat * 100, bins=30, color="tab:blue", alpha=0.7, edgecolor="black")
    mean_monthly = monthly_flat.mean() * 100
    ax.axvline(mean_monthly, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_monthly:.1f}%")
    ax.set_title("Distribution of monthly returns")
    ax.set_xlabel("Returns (%)")
    ax.set_ylabel("Number of months")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{strategy_name} Performance Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        _ensure_results_dir()
        safe_name = strategy_name.lower().replace(" ", "_")
        path = os.path.join(config.RESULTS_DIR, f"figure_{safe_name}_analysis.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {strategy_name} analysis to {path}")

    plt.close(fig)


def plot_cumulative_returns(
    drl_values: pd.Series,
    mvo_values: pd.Series,
    save: bool = True,
):
    """Plot cumulative portfolio values for both strategies."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Normalize to start at 1.0
    ax.plot(drl_values.index, drl_values / drl_values.iloc[0],
            label="DRL (PPO)", color="tab:orange", linewidth=1.5)
    ax.plot(mvo_values.index, mvo_values / mvo_values.iloc[0],
            label="MVO", color="tab:blue", linewidth=1.5)

    ax.set_title("Cumulative Portfolio Value (Normalized)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Value")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        _ensure_results_dir()
        path = os.path.join(config.RESULTS_DIR, "cumulative_returns.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved cumulative returns to {path}")

    plt.close(fig)
