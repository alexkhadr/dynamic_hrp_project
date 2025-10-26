# dynamic_hrp/graphs.py
# -------------------------------------------------------------
# Visualization utilities for analyzing portfolio performance
# by market regime and strategy.
# -------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import math

# -------------------------------------------------------------
# Function: plot_hist_by_regime
# -------------------------------------------------------------
def plot_hist_by_regime(pnl: pd.Series, regimes: pd.Series, bins: int = 40) -> None:
    """
    Plot histograms (density) of portfolio weekly returns split by regimes.
    Also prints per-regime summary statistics (mean, std, min, max, count).
    """
    # Combine PnL and regime labels into one DataFrame, dropping rows with missing data
    tmp = pd.concat([pnl.rename("ret"), regimes.rename("regime")], axis=1).dropna()

    # If no overlapping data exists, exit gracefully
    if tmp.empty:
        print("No overlapping data between pnl and regimes to plot.")
        return

    # Compute per-regime summary statistics
    stats = tmp.groupby("regime")["ret"].agg(
        μ="mean", σ="std", N="count", min="min", max="max"
    )
    print("\nPer-regime weekly return stats:")
    print(stats)

    # Prepare sorted list of regimes (ensures consistent order in plots)
    regs = sorted(tmp["regime"].unique())

    # Create a single plot (overlay histograms for each regime)
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in regs:
        # Extract returns for this specific regime
        r_ret = tmp.loc[tmp["regime"] == r, "ret"]

        # Plot histogram of returns for this regime
        ax.hist(r_ret, bins=bins, alpha=0.55, density=True, label=f"Regime {r}")

        # Add a dashed vertical line for the mean of that regime’s return distribution
        ax.axvline(r_ret.mean(), linestyle="--", linewidth=1)

    # Label axes and add title/legend
    ax.set_title("Weekly Portfolio Returns by Regime — Histogram (Density)")
    ax.set_xlabel("Weekly return")
    ax.set_ylabel("Density")
    ax.legend(title="Regime")
    plt.tight_layout()
    plt.show()


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
) -> None:
    """
    Plot histograms of weekly returns for each strategy (row)
    across regimes (columns).

    Layout:
      - Columns = market regimes (e.g., 0, 1, 2)
      - Rows    = strategies (Dynamic HRP, Equal-Weight, Static HRP)

    Parameters
    ----------
    bt_dyn : dict
        Dynamic HRP backtest output (must include ["pnl", "regimes"]).
    bt_eq : dict
        Equal-Weight backtest output (must include ["pnl"]).
    bt_hrp : dict
        Static HRP (Variance-based) backtest output (must include ["pnl"]).
    bins : int
        Number of histogram bins for each subplot.
    sharex, sharey : bool
        Whether subplots share common axes.
    """

    # --- Combine data across all strategies and align by date
    pnl_dyn = bt_dyn["pnl"].rename("Dynamic HRP")
    regimes = bt_dyn["regimes"].rename("regime")
    pnl_eq  = bt_eq["pnl"].rename("Equal-Weight")
    pnl_hrp = bt_hrp["pnl"].rename("Static HRP (Var)")

    # Merge all series into one DataFrame, removing rows with missing data
    df = pd.concat([regimes, pnl_dyn, pnl_eq, pnl_hrp], axis=1).dropna(subset=["regime"])
    df = df.dropna(subset=["Dynamic HRP", "Equal-Weight", "Static HRP (Var)"])

    # Stop if no overlapping data exists
    if df.empty:
        print("No overlapping pnl/regime data to plot.")
        return

    # --- Setup parameters for grid layout
    strategies = ["Dynamic HRP", "Equal-Weight", "Static HRP (Var)"]
    regs = sorted(df["regime"].unique())
    n_states = len(regs)     # number of regimes (columns)
    n_strats = len(strategies)  # number of strategies (rows)

    # Print summary statistics (mean, std, count) by regime and strategy
    print("\nPer-state summary (mean, std, N) for each strategy:")
    summary = (
        df.melt(id_vars="regime", value_vars=strategies, var_name="strategy", value_name="ret")
          .groupby(["regime", "strategy"])["ret"]
          .agg(mean="mean", std="std", N="count")
    )
    print(summary)

    # --- Create subplot grid (rows = strategies, cols = regimes)
    fig, axes = plt.subplots(
        n_strats, n_states,
        figsize=(3.5 * n_states, 3 * n_strats),
        sharex=sharex, sharey=sharey
    )

    # Handle edge cases where matplotlib returns 1D arrays of axes
    if n_strats == 1:
        axes = axes.reshape(1, -1)
    elif n_states == 1:
        axes = axes.reshape(-1, 1)

    # --- Plot histograms for each (strategy, regime) combination
    for i, s in enumerate(strategies):      # Each row is a strategy
        for j, r in enumerate(regs):        # Each column is a regime
            ax = axes[i, j]

            # Extract returns for this specific regime & strategy
            vals = df.loc[df["regime"] == r, s].dropna()

            # If no data exists, display placeholder text
            if vals.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_axis_off()
                continue

            # Plot histogram and mean line
            ax.hist(vals, bins=bins, density=True, alpha=0.75)
            ax.axvline(vals.mean(), linestyle="--", linewidth=1)

            # Annotate titles and labels only for outer plots to reduce clutter
            if i == 0:
                ax.set_title(f"State {r}")
            if j == 0:
                ax.set_ylabel(f"{s}\nDensity")
            if i == n_strats - 1:
                ax.set_xlabel("Weekly return")

    # --- Final layout adjustments
    fig.suptitle("Weekly Returns — Histograms by Strategy (rows) and Regime (columns)", y=1.02)
    plt.tight_layout()
    plt.show()
