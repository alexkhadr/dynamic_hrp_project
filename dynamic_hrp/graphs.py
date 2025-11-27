# dynamic_hrp/graphs.py
# -------------------------------------------------------------
# Visualization utilities for analyzing portfolio performance
# by market regime and strategy.
# -------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import math
from typing import Optional
import os

# -------------------------------------------------------------
# Global plot style settings
# -------------------------------------------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["savefig.dpi"] = 300   # default save quality


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
# -------------------------------------------------------------
# Function: plot_hist_by_regime
# -------------------------------------------------------------
def plot_hist_by_regime(
    pnl: pd.Series,
    regimes: pd.Series,
    bins: int = 40,
    save_path: Optional[str] = None,
    show: bool = True,
    verbose: bool = False,
    save_tables_to: Optional[str] = None,  # e.g., "Tables/per_regime_weekly_return_stats.md"
    save_table_markdown_fn=None,           # pass your utils_io.save_table_markdown
) -> Optional[pd.DataFrame]:
    """
    Plot histograms of weekly returns split by regimes.
    Optionally save the figure and the per-regime stats table.
    Returns the per-regime stats DataFrame.
    """
    tmp = pd.concat([pnl.rename("ret"), regimes.rename("regime")], axis=1).dropna()
    if tmp.empty:
        if verbose:
            print("No overlapping data between pnl and regimes to plot.")
        return None

    stats_df = tmp.groupby("regime")["ret"].agg(μ="mean", σ="std", N="count", min="min", max="max")
    if verbose:
        print("\nPer-regime weekly return stats:")
        print(stats_df)

    regs = sorted(tmp["regime"].unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in regs:
        r_ret = tmp.loc[tmp["regime"] == r, "ret"]
        ax.hist(r_ret, bins=bins, alpha=0.55, density=True, label=f"Regime {r}")
        ax.axvline(r_ret.mean(), linestyle="--", linewidth=1)

    ax.set_title("Weekly Portfolio Returns by Regime — Histogram (Density)")
    ax.set_xlabel("Weekly return")
    ax.set_ylabel("Density")
    ax.legend(title="Regime")
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # Save table, markdown-only, if requested
    if save_tables_to and save_table_markdown_fn is not None:
        # The helper expects a DataFrame (index kept)
        save_table_markdown_fn(stats_df, out_dir=os.path.dirname(save_tables_to),
                               name=os.path.splitext(os.path.basename(save_tables_to))[0])

    return stats_df


# -------------------------------------------------------------
# Function: plot_statewise_strategy_hists
# -------------------------------------------------------------   
def plot_statewise_strategy_hists(
    bt_dyn: dict,
    bt_eq: dict,
    bt_hrp: dict,
    bins: int = 40,
    sharex: bool = True,
    sharey: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
    verbose: bool = False,
    save_tables_to: Optional[str] = None,  # e.g., "Tables/per_state_strategy_summary.md"
    save_table_markdown_fn=None,
) -> Optional[pd.DataFrame]:
    """
    Plot histograms for each strategy across regimes.
    Optionally save the figure and the per-state summary table.
    Returns the per-state summary DataFrame (reset index).
    """
    pnl_dyn = bt_dyn["pnl"].rename("Dynamic HRP")
    regimes = bt_dyn["regimes"].rename("regime")
    pnl_eq  = bt_eq["pnl"].rename("Equal-Weight")
    pnl_hrp = bt_hrp["pnl"].rename("Static HRP (Var)")

    df = pd.concat([regimes, pnl_dyn, pnl_eq, pnl_hrp], axis=1).dropna(subset=["regime"])
    df = df.dropna(subset=["Dynamic HRP", "Equal-Weight", "Static HRP (Var)"])
    if df.empty:
        if verbose:
            print("No overlapping pnl/regime data to plot.")
        return None

    strategies = ["Dynamic HRP", "Equal-Weight", "Static HRP (Var)"]
    regs = sorted(df["regime"].unique())
    n_states = len(regs)
    n_strats = len(strategies)

    summary = (
        df.melt(id_vars="regime", value_vars=strategies, var_name="strategy", value_name="ret")
          .groupby(["regime", "strategy"])["ret"]
          .agg(mean="mean", std="std", N="count")
    )
    if verbose:
        print("\nPer-state summary (mean, std, N) for each strategy:")
        print(summary)

    summary_reset = summary.reset_index()

    fig, axes = plt.subplots(
        n_strats, n_states,
        figsize=(3.5 * n_states, 3 * n_strats),
        sharex=sharex, sharey=sharey
    )
    if n_strats == 1:
        axes = axes.reshape(1, -1)
    elif n_states == 1:
        axes = axes.reshape(-1, 1)

    for i, s in enumerate(strategies):
        for j, r in enumerate(regs):
            ax = axes[i, j]
            vals = df.loc[df["regime"] == r, s].dropna()
            if vals.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_axis_off()
                continue
            ax.hist(vals, bins=bins, density=True, alpha=0.75)
            ax.axvline(vals.mean(), linestyle="--", linewidth=1)
            if i == 0:
                ax.set_title(f"State {r}")
            if j == 0:
                ax.set_ylabel(f"{s}\nDensity")
            if i == n_strats - 1:
                ax.set_xlabel("Weekly return")

    fig.suptitle("Weekly Returns — Histograms by Strategy (rows) and Regime (columns)", y=1.02)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    if save_tables_to and save_table_markdown_fn is not None:
        save_table_markdown_fn(summary_reset, out_dir=os.path.dirname(save_tables_to),
                               name=os.path.splitext(os.path.basename(save_tables_to))[0])

    return summary_reset
        
# -------------------------------------------------------------
# Function: plot_cumulative_pnl
# -------------------------------------------------------------        
def plot_cumulative_pnl(
    bt_dyn: dict,
    bt_eq: dict,
    bt_hrp: dict,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot and optionally save the cumulative PnL (cumulative return) comparison
    between Dynamic HRP, Equal-Weight, and Static HRP strategies.

    Parameters
    ----------
    bt_dyn, bt_eq, bt_hrp : dict
        Backtest output dictionaries containing 'cum_pnl' Series.
    save_path : str, optional
        File path to save the figure (e.g., "Figures/cumulative_pnl.png").
    show : bool, default True
        Whether to display the figure.
    """
    # Combine the cumulative PnL series into one DataFrame
    df_cum = pd.concat({
        "Dynamic HRP": bt_dyn["cum_pnl"],
        "Equal Weight": bt_eq["cum_pnl"],
        "Static HRP (Var)": bt_hrp["cum_pnl"],
    }, axis=1)

    # Plot the cumulative performance comparison
    ax = df_cum.plot(
        figsize=(6, 3),
        title="Strategy vs Benchmarks (Cumulative Return)",
        linewidth=1.6
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    plt.tight_layout()

    # Save the figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    # Display or close
    if show:
        plt.show()
    else:
        plt.close()