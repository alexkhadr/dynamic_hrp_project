from __future__ import annotations
import pandas as pd

def define_investable_universe(
    prices_daily: pd.DataFrame,
    prices_weekly: pd.DataFrame | None = None,
):
    """
    Select tradables and feature series from the wide price panel.
      Tradables: ES1, TY1, EU1 (fallback to RX1), CL1
      Features:  VIX
    Returns: (universe_prices_daily, universe_prices_weekly, features_daily, mapping)
    """
    tradable_candidates = {
        "ES1": ["ES1", "ES1 Index  (L1)"],
        "TY1": ["TY1", "TY1 COMB Comdty Last Price  (R1)"],
        "EU1": ["EU1", "RX1"],
        "CL1": ["CL1", "WTI"],
    }
    feature_candidates = {
        "VIX": ["VIX", "VIX Index  (R1)", "VIX Index  (R2)", "VIX Index  (L2)", "VIX Index  (R4)"],
    }

    mapping: dict[str, str] = {}
    for canon, opts in tradable_candidates.items():
        for c in opts:
            if c in prices_daily.columns:
                mapping[canon] = c
                break

    feat_map: dict[str, str] = {}
    for canon, opts in feature_candidates.items():
        for c in opts:
            if c in prices_daily.columns:
                feat_map[canon] = c
                break

    univ_d = pd.DataFrame(index=prices_daily.index)
    for canon, src in mapping.items():
        univ_d[canon] = pd.to_numeric(prices_daily[src], errors="coerce")
    univ_d = univ_d.dropna(how="all")

    univ_w = None
    if prices_weekly is not None and not prices_weekly.empty:
        univ_w = pd.DataFrame(index=prices_weekly.index)
        for canon, src in mapping.items():
            if src in prices_weekly.columns:
                univ_w[canon] = pd.to_numeric(prices_weekly[src], errors="coerce")
            elif canon in prices_weekly.columns:
                univ_w[canon] = pd.to_numeric(prices_weekly[canon], errors="coerce")
        univ_w = univ_w.dropna(how="all")

    feats_d = pd.DataFrame(index=prices_daily.index)
    for canon, src in feat_map.items():
        feats_d[canon] = pd.to_numeric(prices_daily[src], errors="coerce")
    feats_d = feats_d.dropna(how="all")

    return univ_d, univ_w, feats_d, mapping
