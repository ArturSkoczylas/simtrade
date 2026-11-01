"""Built-in price/volume features. All are causal (use data up to and including day t)."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .base import FeatureProvider


class Returns(FeatureProvider):
    """Total returns over several horizons: ``ret_<w>`` = adj_close_t / adj_close_{t-w} - 1."""

    name = "returns"

    def __init__(self, windows: Iterable[int] = (1, 5, 21)):
        self.windows = tuple(int(w) for w in windows)

    def compute(self, market) -> Dict[str, pd.DataFrame]:
        px = market.frame("adj_close").ffill()
        return {f"ret_{w}": px.pct_change(w, fill_method=None) for w in self.windows}


class Volatility(FeatureProvider):
    """Annualised rolling volatility of daily log returns: ``vol_<w>``."""

    name = "volatility"

    def __init__(self, windows: Iterable[int] = (21, 63)):
        self.windows = tuple(int(w) for w in windows)

    def compute(self, market) -> Dict[str, pd.DataFrame]:
        lr = np.log(market.frame("adj_close").ffill()).diff()
        return {f"vol_{w}": lr.rolling(w, min_periods=max(2, w // 2)).std() * np.sqrt(252) for w in self.windows}


class Momentum(FeatureProvider):
    """Classic momentum ``mom_<lookback>_<skip>``: return from t-lookback to t-skip."""

    name = "momentum"

    def __init__(self, lookback: int = 252, skip: int = 21):
        self.lookback, self.skip = int(lookback), int(skip)

    def compute(self, market) -> Dict[str, pd.DataFrame]:
        px = market.frame("adj_close").ffill()
        mom = px.shift(self.skip) / px.shift(self.lookback) - 1
        return {f"mom_{self.lookback}_{self.skip}": mom}


class Liquidity(FeatureProvider):
    """Average daily traded value ``adv_<w>`` (close x volume, base currency) and
    volume z-score ``volz_<w>`` (today's volume vs its trailing mean)."""

    name = "liquidity"

    def __init__(self, window: int = 20):
        self.window = int(window)

    def compute(self, market) -> Dict[str, pd.DataFrame]:
        w = self.window
        vol = market.frame("volume")
        dv = market.frame("close") * vol
        mean = vol.rolling(w, min_periods=max(2, w // 2)).mean()
        std = vol.rolling(w, min_periods=max(2, w // 2)).std()
        return {
            f"adv_{w}": dv.rolling(w, min_periods=max(2, w // 2)).mean(),
            f"volz_{w}": (vol - mean) / std.replace(0, np.nan),
        }
