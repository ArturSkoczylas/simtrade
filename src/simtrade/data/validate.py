"""Data validation and repair for daily price tables.

London-listed instruments are usually quoted in pence (``GBp``/``GBX``) and free data sources
sometimes mix pence and pounds, which shows up as spurious 100x jumps. This module converts
everything to major units (e.g. GBP) and repairs or flags the common glitches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .schema import PRICE_FIELDS, standardize

# Minor-unit currency codes -> (major currency, divisor)
MINOR_UNITS = {
    "GBP": None,
    "GBp": ("GBP", 100.0),
    "GBX": ("GBP", 100.0),
    "GBx": ("GBP", 100.0),
    "ZAc": ("ZAR", 100.0),
    "ZAC": ("ZAR", 100.0),
    "ILA": ("ILS", 100.0),
    "USd": ("USD", 100.0),
}


@dataclass
class ValidationReport:
    currency_in: Optional[str]
    currency_out: Optional[str]
    flags: List[str] = field(default_factory=list)

    def add(self, flag: str, count: int = 1) -> None:
        if count:
            self.flags.append(f"{flag}:{count}" if count != 1 else flag)


def normalize_currency(currency: Optional[str]) -> Tuple[Optional[str], float]:
    """Return ``(major_currency, divisor)`` for a quoted currency code."""
    if currency is None:
        return None, 1.0
    cur = str(currency).strip()
    if cur in MINOR_UNITS:
        conv = MINOR_UNITS[cur]
        return (cur, 1.0) if conv is None else conv
    return cur.upper(), 1.0


def _repair_100x(s: pd.Series, window: int = 21) -> Tuple[pd.Series, np.ndarray]:
    """Fix isolated values that are ~100x (or ~1/100x) the local median."""
    if s.dropna().shape[0] < 5:
        return s, np.zeros(len(s), dtype=bool)
    med = s.rolling(window, center=True, min_periods=3).median()
    ratio = s / med
    hi = (ratio > 70) & (ratio < 140)
    lo = (ratio > 1 / 140) & (ratio < 1 / 70)
    out = s.copy()
    out[hi] = out[hi] / 100.0
    out[lo] = out[lo] * 100.0
    return out, (hi | lo).to_numpy()


def validate_prices(
    df: pd.DataFrame, currency: Optional[str] = None, repair: bool = True
) -> Tuple[pd.DataFrame, ValidationReport]:
    """Validate, repair and convert a standard price table to major currency units.

    Repairs (when ``repair=True``): pence->pounds conversion, isolated 100x price glitches,
    non-positive prices, missing/inconsistent open/high/low, and dividends that are
    implausibly larger than the share price (a pence/pounds mix-up).

    Flags only (never changed): large one-day spikes that revert, very stale rows.
    """
    df = standardize(df)
    major, divisor = normalize_currency(currency)
    rep = ValidationReport(currency_in=currency, currency_out=major)
    if df.empty:
        rep.add("empty")
        return df, rep

    out = df.copy()
    if divisor != 1.0:
        out[PRICE_FIELDS + ["dividends"]] = out[PRICE_FIELDS + ["dividends"]] / divisor
        rep.add(f"converted_{currency}_to_{major}")

    if not repair:
        return out, rep

    # Non-positive prices are errors.
    bad = out[PRICE_FIELDS] <= 0
    n_bad = int(bad.to_numpy().sum())
    if n_bad:
        out[PRICE_FIELDS] = out[PRICE_FIELDS].mask(bad)
        rep.add("nonpositive_price", n_bad)

    # 100x glitches, detected on close and applied to the whole row.
    fixed_close, mask = _repair_100x(out["close"])
    if mask.any():
        rows = out.index[mask]
        factor = (fixed_close[rows] / out.loc[rows, "close"]).to_numpy()[:, None]
        out.loc[rows, PRICE_FIELDS] = out.loc[rows, PRICE_FIELDS].to_numpy() * factor
        rep.add("repaired_100x", int(mask.sum()))
    # Same for adj_close independently (its scale can differ from close).
    fixed_adj, mask_adj = _repair_100x(out["adj_close"])
    extra = mask_adj & ~mask
    if extra.any():
        out.loc[extra, "adj_close"] = fixed_adj[extra]
        rep.add("repaired_100x_adj", int(extra.sum()))

    # Fill missing close from adj_close and vice versa only where one side is missing.
    miss_close = out["close"].isna() & out["adj_close"].notna()
    if miss_close.any():
        rep.add("close_missing", int(miss_close.sum()))
    miss_adj = out["adj_close"].isna() & out["close"].notna()
    if miss_adj.any():
        out.loc[miss_adj, "adj_close"] = out.loc[miss_adj, "close"]
        rep.add("adj_close_filled", int(miss_adj.sum()))

    # Open/high/low consistency.
    miss_open = out["open"].isna() & out["close"].notna()
    if miss_open.any():
        out.loc[miss_open, "open"] = out.loc[miss_open, "close"]
        rep.add("open_filled_from_close", int(miss_open.sum()))
    oc_max = out[["open", "close"]].max(axis=1)
    oc_min = out[["open", "close"]].min(axis=1)
    hi_bad = out["high"].isna() | (out["high"] < oc_max)
    lo_bad = out["low"].isna() | (out["low"] > oc_min)
    if hi_bad.any():
        out.loc[hi_bad, "high"] = oc_max[hi_bad]
        rep.add("high_fixed", int(hi_bad.sum()))
    if lo_bad.any():
        out.loc[lo_bad, "low"] = oc_min[lo_bad]
        rep.add("low_fixed", int(lo_bad.sum()))

    # Dividends larger than half the share price are almost surely a pence/pounds mix-up.
    prev_close = out["close"].shift(1).fillna(out["close"])
    div_yield = out["dividends"] / prev_close
    huge = (out["dividends"] > 0) & (div_yield > 0.5) & (div_yield / 100 < 0.5)
    if huge.any():
        out.loc[huge, "dividends"] = out.loc[huge, "dividends"] / 100.0
        rep.add("dividend_rescaled", int(huge.sum()))
    neg_div = out["dividends"] < 0
    if neg_div.any():
        out.loc[neg_div, "dividends"] = 0.0
        rep.add("negative_dividend_removed", int(neg_div.sum()))

    # Flags only: one-day spikes (>80%) that revert the next day.
    r = out["close"].pct_change()
    spikes = (r.abs() > 0.8) & (r.shift(-1).abs() > 0.4) & (np.sign(r) != np.sign(r.shift(-1)))
    rep.add("spike_flagged", int(spikes.sum()))

    out = out[~out[PRICE_FIELDS].isna().all(axis=1)]
    return out, rep
