from __future__ import annotations
import re
import numpy as np
import pandas as pd

# ----------------------------
# Helpers
# ----------------------------
def to_datetime_series(s: pd.Series) -> pd.Series:
    """Robust date parsing for mixed/str formats (naive tz)."""
    return pd.to_datetime(s, errors="coerce", utc=False).dt.tz_localize(None)

def ffill_prune(
    df: pd.DataFrame,
    max_ffill_days: int = 5,
    min_non_na_frac: float = 0.75
) -> pd.DataFrame:
    """
    Forward-fill short gaps and drop columns that remain too sparse.
    Assumes a (nearly) daily DateTimeIndex.
    """
    df = df.sort_index()
    df = df.ffill(limit=max_ffill_days)
    keep = (df.notna().mean() >= min_non_na_frac)
    return df.loc[:, keep]

def weekly_last(df: pd.DataFrame, week_day: str = "FRI") -> pd.DataFrame:
    """Resample to weekly using the last observation on the chosen weekday (e.g., FRI)."""
    return df.resample(f"W-{week_day}").last().dropna(how="all")

def pick_first_nonnull_column(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Return the first candidate column that exists and has at least one non-null value."""
    for c in candidates:
        if c in df.columns:
            ser = df[c]
            if not ser.dropna().empty:
                return ser
    return pd.Series(index=df.index, dtype=float)

# ----------------------------
# Bond futures quote parsing
# ----------------------------
_BOND_32_RE = re.compile(r"^(\d+)[\-\s](\d{2})(\+)?$")
_BOND_320_RE = re.compile(r"^(\d+)[\-\s](\d{3})$")
_BOND_HALF_RE = re.compile(r"^(\d+)[\-\s](\d{2})\s*(1/2)$")

def parse_bond_fut_price(val) -> float:
    """
    Convert bond futures quotes to decimal.
    Accepts: 112-29, 112-29+, 112-295, 112-16 1/2, 112.90625
    Returns float or np.nan.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val).strip().replace(",", "")

    # Already decimal?
    try:
        return float(s)
    except ValueError:
        pass

    m = _BOND_32_RE.fullmatch(s)
    if m:
        pts = int(m.group(1)); frac = int(m.group(2)); plus = m.group(3)
        thirty2nds = frac + (0.5 if plus else 0.0)
        return pts + thirty2nds / 32.0

    m = _BOND_320_RE.fullmatch(s)
    if m:
        pts = int(m.group(1)); frac3 = m.group(2)
        base = int(frac3[:2]); half = 0.5 if frac3[2] == "5" else 0.0
        return pts + (base + half) / 32.0

    m = _BOND_HALF_RE.fullmatch(s)
    if m:
        pts = int(m.group(1)); frac = int(m.group(2))
        return pts + (frac + 0.5) / 32.0

    return np.nan

def to_numeric_bondaware(series: pd.Series) -> pd.Series:
    """Try numeric; if mostly NaN, parse per-element as bond quotes."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any() and s.isna().mean() < 0.8:
        return s
    return series.map(parse_bond_fut_price)
