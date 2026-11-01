"""The standard daily price schema used throughout simtrade.

Every price table is a DataFrame indexed by a tz-naive ``DatetimeIndex`` named ``date`` with
these float columns:

========== ==================================================================
open       split-adjusted open
high       split-adjusted high
low        split-adjusted low
close      split-adjusted close
adj_close  split- and dividend-adjusted close (total-return series)
volume     split-adjusted share volume (NaN if unknown)
dividends  cash dividend per share with ex-date on this row (0 if none)
splits     split ratio taking effect on this row (0 if none)
========== ==================================================================
"""

from __future__ import annotations

import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]
PRICE_FIELDS = ["open", "high", "low", "close", "adj_close"]

_ALIASES = {
    "adj close": "adj_close",
    "adjclose": "adj_close",
    "adjusted_close": "adj_close",
    "adj_close": "adj_close",
    "stock splits": "splits",
    "stock_splits": "splits",
    "split": "splits",
    "dividend": "dividends",
    "div": "dividends",
}


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a provider's price table into the standard schema.

    Missing ``adj_close`` defaults to ``close``; missing dividends/splits default to 0;
    missing volume is NaN. Rows without any price are dropped; duplicates keep the last row.
    """
    if df is None or len(df) == 0:
        return empty_frame()
    if _is_standard(df):
        return df
    out = df.copy()
    if "date" in [str(c).lower() for c in out.columns]:
        col = [c for c in out.columns if str(c).lower() == "date"][0]
        out = out.set_index(col)
    out.columns = [_ALIASES.get(str(c).strip().lower(), str(c).strip().lower().replace(" ", "_")) for c in out.columns]
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "date"

    if "close" not in out.columns:
        raise ValueError("price table needs at least a 'close' column")
    if "adj_close" not in out.columns:
        out["adj_close"] = out["close"]
    for c in ("open", "high", "low"):
        if c not in out.columns:
            out[c] = float("nan")
    if "volume" not in out.columns:
        out["volume"] = float("nan")
    for c in ("dividends", "splits"):
        if c not in out.columns:
            out[c] = 0.0

    out = out[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce").astype("float64")
    out[["dividends", "splits"]] = out[["dividends", "splits"]].fillna(0.0)
    out = out[~out[PRICE_FIELDS].isna().all(axis=1)]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _is_standard(df: pd.DataFrame) -> bool:
    """Fast path: already in the standard schema (e.g. read back from the store)."""
    idx = df.index
    return (
        list(df.columns) == PRICE_COLUMNS
        and isinstance(idx, pd.DatetimeIndex)
        and idx.tz is None
        and idx.is_monotonic_increasing
        and idx.is_unique
        and all(dt == "float64" for dt in df.dtypes)
        and bool((idx == idx.normalize()).all())
    )


def empty_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], name="date")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in PRICE_COLUMNS}, index=idx)
