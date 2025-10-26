# -------------------------------------------------------------
# Data Standardization Utilities
# -------------------------------------------------------------
# This module provides robust helpers for cleaning, parsing,
# and standardizing raw financial data before feature or
# backtest construction.
#
# Includes:
#   • Robust datetime parsing
#   • Forward-fill and sparsity pruning
#   • Weekly resampling
#   • Column selection utilities
#   • Bond futures quote conversion (e.g., 112-29+ → 112.90625)
# -------------------------------------------------------------

from __future__ import annotations
import re
import numpy as np
import pandas as pd

# =============================================================
# ---------------------------- Helpers ------------------------
# =============================================================
def to_datetime_series(s: pd.Series) -> pd.Series:
    """
    Robustly parse mixed-format date columns into pandas datetime.
    - Coerces invalid entries to NaT
    - Removes timezone info (returns naive timestamps)
    """
    return pd.to_datetime(s, errors="coerce", utc=False).dt.tz_localize(None)


def ffill_prune(
    df: pd.DataFrame,
    max_ffill_days: int = 5,
    min_non_na_frac: float = 0.75
) -> pd.DataFrame:
    """
    Forward-fill short gaps and drop columns that remain too sparse.

    Steps:
      1. Sort by index (should be DateTimeIndex).
      2. Forward-fill up to 'max_ffill_days' consecutive NaNs.
      3. Drop columns with fewer than 'min_non_na_frac' valid observations.

    Intended for daily data where small gaps (e.g., holidays) can be tolerated.
    """
    df = df.sort_index()
    df = df.ffill(limit=max_ffill_days)
    # Keep only columns with sufficient non-NaN coverage
    keep = (df.notna().mean() >= min_non_na_frac)
    return df.loc[:, keep]


def weekly_last(df: pd.DataFrame, week_day: str = "FRI") -> pd.DataFrame:
    """
    Resample a daily DataFrame to weekly frequency,
    taking the last available observation of each week.

    Parameters
    ----------
    df : pd.DataFrame
        Daily data indexed by DateTimeIndex.
    week_day : str
        Target weekday (e.g., 'FRI' for Friday close).

    Returns
    -------
    pd.DataFrame
        Weekly data with one observation per week.
    """
    return df.resample(f"W-{week_day}").last().dropna(how="all")


def pick_first_nonnull_column(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """
    Among a list of candidate column names, return the first that:
      - exists in the DataFrame, and
      - contains at least one non-null value.

    Useful when data vendors change column suffixes over time.

    If none are valid, returns an empty Series with the same index.
    """
    for c in candidates:
        if c in df.columns:
            ser = df[c]
            if not ser.dropna().empty:
                return ser
    return pd.Series(index=df.index, dtype=float)


# =============================================================
# --------------------- Bond futures quote parsing ------------
# =============================================================
# Bond futures (e.g., U.S. Treasury futures) are quoted in "32nds":
#   112-29   = 112 + 29/32 = 112.90625
#   112-29+  = 112 + 29.5/32 = 112.921875
#   112-295  = 112 + 29.5/32 (alt encoding)
#   112-16 1/2 = 112 + 16.5/32
# These helpers convert such strings to proper decimal floats.
# =============================================================

# Precompiled regex patterns for different quoting styles
_BOND_32_RE   = re.compile(r"^(\d+)[\-\s](\d{2})(\+)?$")        # e.g., 112-29 or 112-29+
_BOND_320_RE  = re.compile(r"^(\d+)[\-\s](\d{3})$")             # e.g., 112-295 (encoded 29.5/32)
_BOND_HALF_RE = re.compile(r"^(\d+)[\-\s](\d{2})\s*(1/2)$")     # e.g., 112-16 1/2

def parse_bond_fut_price(val) -> float:
    """
    Parse a bond futures quote string into a decimal float.

    Accepts formats like:
        112-29, 112-29+, 112-295, 112-16 1/2, or already-decimal (112.90625)
    Returns:
        float value in decimal form, or np.nan if invalid.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val)

    s = str(val).strip().replace(",", "")  # remove commas, spaces, etc.

    # Case 1: Already a clean decimal (e.g., "112.90625")
    try:
        return float(s)
    except ValueError:
        pass

    # Case 2: Pattern like "112-29" or "112-29+"
    m = _BOND_32_RE.fullmatch(s)
    if m:
        pts = int(m.group(1))
        frac = int(m.group(2))
        plus = m.group(3)
        thirty2nds = frac + (0.5 if plus else 0.0)
        return pts + thirty2nds / 32.0

    # Case 3: Pattern like "112-295" (encodes 29.5/32)
    m = _BOND_320_RE.fullmatch(s)
    if m:
        pts = int(m.group(1))
        frac3 = m.group(2)
        base = int(frac3[:2])
        half = 0.5 if frac3[2] == "5" else 0.0
        return pts + (base + half) / 32.0

    # Case 4: Pattern like "112-16 1/2"
    m = _BOND_HALF_RE.fullmatch(s)
    if m:
        pts = int(m.group(1))
        frac = int(m.group(2))
        return pts + (frac + 0.5) / 32.0

    # Default: unrecognized format
    return np.nan


def to_numeric_bondaware(series: pd.Series) -> pd.Series:
    """
    Convert a Series to numeric, with fallback to bond-style parsing if needed.

    Logic:
      1. Try pd.to_numeric() (handles standard decimals)
      2. If >80% of values are still NaN, assume bond quotes and parse manually

    Returns
    -------
    pd.Series of floats (in decimal form)
    """
    s = pd.to_numeric(series, errors="coerce")

    # If majority of entries are valid numerics, return directly
    if s.notna().any() and s.isna().mean() < 0.8:
        return s

    # Otherwise, attempt parsing as bond futures quotes
    return series.map(parse_bond_fut_price)
