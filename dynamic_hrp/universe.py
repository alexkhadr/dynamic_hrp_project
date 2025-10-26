# -------------------------------------------------------------
# Define Investable Universe
# -------------------------------------------------------------
# This function selects the core set of tradable instruments
# (e.g., futures or indexes) and feature series (e.g., volatility)
# from a wide price panel containing many assets.
#
# It:
#   • Maps canonical tickers (ES1, TY1, EU1/RX1, CL1) to actual
#     column names found in the raw dataset
#   • Extracts clean numeric daily and weekly subpanels
#   • Extracts auxiliary feature series (like VIX)
#   • Returns the trimmed daily/weekly universes + feature panel
# -------------------------------------------------------------

from __future__ import annotations
import pandas as pd

def define_investable_universe(
    prices_daily: pd.DataFrame,
    prices_weekly: pd.DataFrame | None = None,
):
    """
    Define the investable universe and associated feature series.

    Steps:
      1. Identify canonical tradable tickers and their possible aliases.
      2. Identify auxiliary feature tickers (e.g., VIX).
      3. Extract corresponding columns from the daily and weekly price panels.
      4. Return cleaned daily/weekly universes and a feature DataFrame.

    Parameters
    ----------
    prices_daily : pd.DataFrame
        Daily wide price panel with many instruments (columns = tickers).
    prices_weekly : pd.DataFrame, optional
        Weekly wide price panel, if available (same structure).

    Returns
    -------
    tuple(pd.DataFrame, pd.DataFrame | None, pd.DataFrame, dict)
        (universe_prices_daily, universe_prices_weekly, features_daily, mapping)

    Notes
    -----
    Canonical tickers:
        - ES1 : S&P 500 E-mini futures
        - TY1 : 10-Year Treasury futures
        - EU1 : Bund future (RX1 fallback)
        - CL1 : Crude oil futures
    Feature series:
        - VIX : Volatility Index
    """
    # ---------------------------------------------------------
    # Candidate names for each canonical tradable
    # (some vendor files have variant labels)
    # ---------------------------------------------------------
    tradable_candidates = {
        "ES1": ["ES1", "ES1 Index  (L1)"],
        "TY1": ["TY1", "TY1 COMB Comdty Last Price  (R1)"],
        "EU1": ["EU1", "RX1"],  # Fallback: RX1 is Euro Bund
        "CL1": ["CL1", "WTI"],  # Fallback: generic WTI label
    }

    # Candidate names for features (e.g., volatility measures)
    feature_candidates = {
        "VIX": ["VIX", "VIX Index  (R1)", "VIX Index  (R2)",
                "VIX Index  (L2)", "VIX Index  (R4)"],
    }

    # ---------------------------------------------------------
    # Build mapping for tradables: canonical name → actual column
    # ---------------------------------------------------------
    mapping: dict[str, str] = {}
    for canon, opts in tradable_candidates.items():
        for c in opts:
            if c in prices_daily.columns:
                mapping[canon] = c
                break  # use the first match

    # ---------------------------------------------------------
    # Build mapping for feature series (e.g., VIX)
    # ---------------------------------------------------------
    feat_map: dict[str, str] = {}
    for canon, opts in feature_candidates.items():
        for c in opts:
            if c in prices_daily.columns:
                feat_map[canon] = c
                break

    # ---------------------------------------------------------
    # Extract daily universe prices using the identified mapping
    # ---------------------------------------------------------
    univ_d = pd.DataFrame(index=prices_daily.index)
    for canon, src in mapping.items():
        # Coerce column to numeric in case of mixed types
        univ_d[canon] = pd.to_numeric(prices_daily[src], errors="coerce")
    univ_d = univ_d.dropna(how="all")  # drop fully empty rows

    # ---------------------------------------------------------
    # Extract weekly universe (if provided)
    # ---------------------------------------------------------
    univ_w = None
    if prices_weekly is not None and not prices_weekly.empty:
        univ_w = pd.DataFrame(index=prices_weekly.index)
        for canon, src in mapping.items():
            # Prefer matching by actual mapped name, fallback to canonical if exists
            if src in prices_weekly.columns:
                univ_w[canon] = pd.to_numeric(prices_weekly[src], errors="coerce")
            elif canon in prices_weekly.columns:
                univ_w[canon] = pd.to_numeric(prices_weekly[canon], errors="coerce")
        univ_w = univ_w.dropna(how="all")

    # ---------------------------------------------------------
    # Extract feature series (e.g., volatility indices)
    # ---------------------------------------------------------
    feats_d = pd.DataFrame(index=prices_daily.index)
    for canon, src in feat_map.items():
        feats_d[canon] = pd.to_numeric(prices_daily[src], errors="coerce")
    feats_d = feats_d.dropna(how="all")

    # ---------------------------------------------------------
    # Return clean universe and feature sets
    # ---------------------------------------------------------
    return univ_d, univ_w, feats_d, mapping
