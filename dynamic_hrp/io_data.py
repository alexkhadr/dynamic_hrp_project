# -------------------------------------------------------------
# Data Loading & Panel Builders for Cross-Asset Prices
# -------------------------------------------------------------
# Utilities to:
#   • Load raw CSVs
#   • Split OHLCV "block" style files into tidy frames
#   • Build clean, aligned daily & weekly price panels across FX, rates, equities,
#     commodities, and volatility indexes (VIX/V2X)
#   • Apply light cleaning: robust date parsing, numeric coercion, forward-fill with caps
# -------------------------------------------------------------

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from .standardize import (
    to_datetime_series,     # robust date parser returning tz-naive pandas datetime
    ffill_prune,            # forward-fill small gaps & prune overly sparse columns
    weekly_last,            # resample to weekly (take last obs on specified weekday)
    pick_first_nonnull_column,  # choose first column among candidates that has data
    to_numeric_bondaware    # convert bond-style quotes (e.g., 112-16) to decimal
)

# ---- Raw CSV loaders (light) ----
def load_raw_frames() -> dict[str, pd.DataFrame]:
    """Load all raw CSVs from the ./Data directory (paths can be edited here)."""
    # Read wide CSVs as delivered (minimal parsing here; standardization happens later)
    Energy_Comdty   = pd.read_csv("Data/AT-03_Energy_Metals_Comdty_Future_Daily_2000_2025.csv")
    Spot_FX         = pd.read_csv("Data/AT-04_Daily_Spot_Prices_G10_FX_Pairs_Daily_2000_2025.csv")
    TY1_Comdty      = pd.read_csv("Data/AT-15_NOB_Spread_Compenents_Daily_2005_2025.csv")
    RX1_Comdty      = pd.read_csv("Data/AT-16_Euro_Bond_Future_Daily_2005_2025.csv")
    Equity_Futures  = pd.read_csv("Data/AT-39_Equity_Futures_Daily_1990_2025.csv")
    Volatility_Index= pd.read_csv("Data/AT-46_Volatility_Index_Daily_1990_2025.csv")

    # Some RX1 files mistakenly include a duplicate header row as the first data row.
    # If so, promote that row to header and drop it from data.
    if RX1_Comdty.columns[0] != "Date":
        RX1_Comdty.columns = RX1_Comdty.iloc[0]
        RX1_Comdty = RX1_Comdty.iloc[1:].reset_index(drop=True)

    return {
        "Energy_Comdty": Energy_Comdty,
        "Spot_FX": Spot_FX,
        "TY1_Comdty": TY1_Comdty,
        "RX1_Comdty": RX1_Comdty,
        "Equity_Futures": Equity_Futures,
        "Volatility_Index": Volatility_Index
    }

# ---- Energy/metals OHLCV splitter (wide CSV with repeating subblocks) ----
def split_ohlcv_blocks(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Parse wide OHLCV sheets like AT-03 where columns repeat per ticker:
      [..., '<TICKER> Comdty', '<TICKER> Open', '<TICKER> Low',
       '<TICKER> Last', '<TICKER> High', '<TICKER> Volume', ...].

    We:
      1) Detect each '<...> Comdty' "anchor" column
      2) Slice the subsequent 6 columns as an OHLCV block
      3) Tidy the block into columns: Date, Open, Low, Last, High, Volume, Ticker
      4) Robustly parse dates and sort
    Returns: {ticker -> tidy DataFrame}
    """
    # Identify anchors for each commodity block
    tickers = [c for c in df.columns if "Comdty" in c]
    cleaned: dict[str, pd.DataFrame] = {}
    for t in tickers:
        base_idx = df.columns.get_loc(t)
        subcols = df.columns[base_idx: base_idx + 6]  # expected 6-col block
        temp = df[subcols].copy()
        temp.columns = ["Date", "Open", "Low", "Last", "High", "Volume"]
        # Normalize ticker label (drop suffixes sometimes present)
        temp["Ticker"] = t.replace(" COMB", "").replace(" Curncy", "")
        temp["Date"] = to_datetime_series(temp["Date"])
        temp = temp.dropna(subset=["Date"]).sort_values("Date")
        cleaned[temp["Ticker"].iloc[0]] = temp
    return cleaned

def build_last_from_ohlcv(
    x: pd.DataFrame, ticker_col: str = "Ticker", last_col: str = "Last"
) -> pd.DataFrame:
    """Pivot a tidy OHLCV table into a wide matrix of 'Last' prices indexed by Date."""
    y = x.copy()
    y["Date"] = to_datetime_series(y["Date"])
    # Remove bad dates and duplicate (Date, Ticker) rows
    y = y.dropna(subset=["Date"]).drop_duplicates(subset=["Date", ticker_col])
    # Coerce 'Last' to numeric (non-numeric → NaN)
    y[last_col] = pd.to_numeric(y[last_col], errors="coerce")
    # Wide pivot: dates x tickers
    return y.pivot(index="Date", columns=ticker_col, values=last_col).sort_index()

# ---- Specific builders for each dataset ----
def build_fx_panel(Spot_FX: pd.DataFrame) -> pd.DataFrame:
    """
    Spot FX arrives as '<PAIR> - Date' & '<PAIR> - Last Price'.
    We pair those and build a wide FX panel: EURUSD, USDJPY, ...
    Also forward-fill small gaps and drop overly sparse series.
    """
    cols = Spot_FX.columns
    pairs = []
    # Discover all currency pairs that have both a Date and matching Last Price column
    for c in cols:
        m = re.match(r"^([A-Z]{6})\s*-\s*Date$", c)
        if m:
            pair = m.group(1)
            last_col = f"{pair} - Last Price"
            if last_col in cols:
                pairs.append((pair, c, last_col))
    wide = []
    # Build per-pair small frames and align on 'date' index
    for pair, date_col, last_col in pairs:
        dfp = Spot_FX[[date_col, last_col]].copy()
        dfp.columns = ["date", pair]
        dfp["date"] = to_datetime_series(dfp["date"])
        dfp = dfp.dropna(subset=["date"]).drop_duplicates(subset=["date"]).set_index("date")
        wide.append(dfp)
    if not wide:
        return pd.DataFrame()
    # Merge all pairs horizontally; enforce numeric type; clean with ffill/prune
    fx = pd.concat(wide, axis=1).sort_index().astype(float)
    fx = ffill_prune(fx, max_ffill_days=5, min_non_na_frac=0.50)
    # Place common majors first (if present)
    order = ["EURUSD", "USDJPY", "GBPUSD", "USDCAD", "AUDUSD", "USDCHF"]
    fx = fx.reindex(columns=[c for c in order if c in fx.columns] + [c for c in fx.columns if c not in order])
    return fx

def build_ty1_series(TY1_Comdty: pd.DataFrame) -> pd.Series:
    """Build TY1 (US 10Y note future) series; handle bond-style fractional quotes."""
    df = TY1_Comdty.copy()
    df["Date"] = to_datetime_series(df["Date"])
    df = df.dropna(subset=["Date"]).drop_duplicates(subset=["Date"]).set_index("Date")
    # Try a canonical column first, then fall back to reasonable alternatives.
    col = "TY1 COMB Comdty Last Price  (R1)"
    if col not in df.columns:
        cands = [c for c in df.columns if ("TY1" in c.upper() and "LAST" in c.upper())]
        col = cands[0] if cands else next((c for c in df.columns if "US1" in c.upper() and "LAST" in c.upper()), None)
        if col is None:
            return pd.Series(dtype=float)
    ser = to_numeric_bondaware(df[col])  # converts e.g., 112-16 → 112.5
    ser.name = "TY1"
    return ser

def build_rx1_series(RX1_Comdty: pd.DataFrame) -> pd.Series:
    """Build RX1 (Bund future) series; choose a sensible 'Last Price' column."""
    df = RX1_Comdty.copy()
    df["Date"] = to_datetime_series(df["Date"])
    df = df.dropna(subset=["Date"]).drop_duplicates(subset=["Date"]).set_index("Date")
    col = "Last Price  (R2)"
    if col not in df.columns:
        # Fall back to first column containing 'Last' and 'Price'
        cands = [c for c in df.columns if ("Last" in c and "Price" in c)]
        col = cands[0] if cands else None
        if col is None:
            return pd.Series(dtype=float)
    ser = pd.to_numeric(df[col], errors="coerce")
    ser.name = "RX1"
    return ser

def build_equity_futs_panel(Equity_Futures: pd.DataFrame) -> pd.DataFrame:
    """Equity futures panel across a few canonical tickers (ES1, VG1, NK1) when available."""
    df = Equity_Futures.copy()
    df["Date"] = to_datetime_series(df["Date"])
    df = df.dropna(subset=["Date"]).drop_duplicates(subset=["Date"]).set_index("Date")
    # Map raw column names to standardized tickers
    mapping = {
        "ES1 Index  (L1)": "ES1",
        "VG1 Index  (R1)": "VG1",
        "NK1 COMB Index  (R2)": "NK1",
    }
    out = {}
    for raw, std in mapping.items():
        if raw in df.columns:
            out[std] = pd.to_numeric(df[raw], errors="coerce")
    # If nothing mapped, still return empty frame indexed on dates (useful for joins)
    return pd.DataFrame(out).sort_index() if out else pd.DataFrame(index=df.index)

def build_vol_index_panel(Volatility_Index: pd.DataFrame) -> pd.DataFrame:
    """Build VIX and V2X series by picking the first non-null among candidate columns."""
    df = Volatility_Index.copy()
    df["Date"] = to_datetime_series(df["Date"])
    df = df.dropna(subset=["Date"]).drop_duplicates(subset=["Date"]).set_index("Date")
    # Choose best VIX/V2X columns among alternatives (vendor files may vary)
    vix = pick_first_nonnull_column(df, ["VIX Index  (R1)","VIX Index  (R2)","VIX Index  (L2)","VIX Index  (R4)"])
    v2x = pick_first_nonnull_column(df, ["V2X Index  (L1)","V2X Index  (R1)","V2X Index  (L3)","V2X Index  (R3)","V2X Index  (L4)"])
    out = {}
    if not vix.dropna().empty: out["VIX"] = pd.to_numeric(vix, errors="coerce")
    if not v2x.dropna().empty: out["V2X"] = pd.to_numeric(v2x, errors="coerce")
    return pd.DataFrame(out).sort_index() if out else pd.DataFrame(index=df.index)

def build_price_panels(
    Spot_FX,
    TY1_Comdty,
    RX1_Comdty,
    Equity_Futures,
    Volatility_Index,
    GC1,
    CL1,
    convert_rx1_to_usd: bool=False
):
    """
    Assemble:
      • prices_daily  : master daily price panel across assets
      • prices_weekly : W-FRI last prices from daily panel
      • subpanels     : dict of component panels for inspection

    Steps:
      1) Build individual panels/series (FX, rates, equities, vol, energy)
      2) Optionally convert RX1 (EUR) into USD using EURUSD
      3) Outer-join everything into one daily matrix
      4) Coerce to numeric, sort by date, forward-fill with caps, prune sparse
      5) Order columns (preferred majors first), and produce weekly panel
    """
    # --- Build core subpanels
    fx = build_fx_panel(Spot_FX)
    ty1 = build_ty1_series(TY1_Comdty).to_frame()
    rx1 = build_rx1_series(RX1_Comdty).to_frame()

    # Optionally convert Bund (EUR-denominated) to USD using EURUSD spot
    if convert_rx1_to_usd and "EURUSD" in fx.columns and not rx1.empty:
        rx1 = (rx1["RX1"] * fx["EURUSD"]).to_frame(name="RX1")
    rates_panel = ty1.join(rx1, how="outer")

    eq_panel  = build_equity_futs_panel(Equity_Futures)
    vol_panel = build_vol_index_panel(Volatility_Index)

    # Gold/Crude OHLCV → 'Last' wide frames (may be empty if not provided)
    gc1_panel = build_last_from_ohlcv(GC1) if not GC1.empty else pd.DataFrame()
    cl1_panel = build_last_from_ohlcv(CL1) if not CL1.empty else pd.DataFrame()

    # If a single-column frame came through, standardize the column name to the ticker
    def _rename_singlecol(panel, target):
        return panel.rename(columns={panel.columns[0]: target}) if (not panel.empty and panel.shape[1]==1 and target not in panel.columns) else panel
    gc1_panel = _rename_singlecol(gc1_panel, "GC1")
    cl1_panel = _rename_singlecol(cl1_panel, "CL1")

    # --- Outer join all available frames into a single daily panel
    frames = [eq_panel, rates_panel, gc1_panel, cl1_panel, vol_panel, fx]
    prices_daily = None
    for f in frames:
        if f is None or f.empty:
            continue
        prices_daily = f if prices_daily is None else prices_daily.join(f, how="outer")

    # Ensure we return a DataFrame even if everything was empty
    prices_daily = pd.DataFrame() if prices_daily is None else prices_daily

    # Numeric coercion, chronological order, and conservative gap filling
    prices_daily = prices_daily.apply(pd.to_numeric, errors="coerce").sort_index()
    prices_daily = ffill_prune(prices_daily, max_ffill_days=5, min_non_na_frac=0.60)

    # Put commonly used columns first to keep things tidy
    preferred = ["ES1","TY1","RX1","CL1","GC1","VIX","V2X","EURUSD","USDJPY","GBPUSD","USDCAD","AUDUSD","USDCHF"]
    ordered = [c for c in preferred if c in prices_daily.columns]
    others  = [c for c in prices_daily.columns if c not in ordered]
    prices_daily = prices_daily.loc[:, ordered + others]

    # Weekly (W-FRI) panel for strategy inputs
    prices_weekly = weekly_last(prices_daily, week_day="FRI")

    # Provide subpanels for diagnostics or unit tests
    subpanels = {
        "fx": fx,
        "rates": rates_panel,
        "eq_futs": eq_panel,
        "vol": vol_panel,
        "energy": pd.concat([gc1_panel, cl1_panel], axis=1).sort_index()
    }
    return prices_daily, prices_weekly, subpanels
