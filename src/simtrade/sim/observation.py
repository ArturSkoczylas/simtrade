"""What an agent sees when it decides: data up to and including the current session only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from ..data.universe import Instrument


@dataclass(frozen=True)
class PortfolioState:
    """Snapshot of the portfolio at the close of the decision day."""

    cash: float
    equity: float
    shares: pd.Series
    weights: pd.Series

    @property
    def holdings(self) -> pd.Series:
        """Non-zero positions (shares)."""
        return self.shares[self.shares != 0]


class Observation:
    """Point-in-time view of the market for the decision made after the close of ``date``.

    Every accessor slices rows ``<= index``; there is no way to reach later data through it.
    DataFrames returned are read-only views (dates x tickers).
    """

    def __init__(self, market, index: int, lookback: int, portfolio: PortfolioState, features=None):
        self._market = market
        self.index = int(index)
        self.lookback = int(lookback)
        self.portfolio = portfolio
        self._features = features

    # -- basics ----------------------------------------------------------------
    @property
    def date(self) -> pd.Timestamp:
        return self._market.dates[self.index]

    @property
    def tickers(self) -> List[str]:
        return self._market.tickers

    @property
    def universe(self):
        return self._market.universe

    def instrument(self, ticker: str) -> Instrument:
        return self._market.instrument(ticker)

    def _rows(self, lookback: Optional[int]) -> slice:
        n = self.lookback if lookback is None else int(lookback)
        return slice(max(0, self.index - n + 1), self.index + 1)

    # -- price history ---------------------------------------------------------
    def history(self, field: str, lookback: Optional[int] = None) -> pd.DataFrame:
        """Last ``lookback`` rows (default: the simulation's lookback) of a price field."""
        sl = self._rows(lookback)
        return pd.DataFrame(
            self._market.array(field)[sl], index=self._market.dates[sl], columns=self.tickers, copy=False
        )

    def latest(self, field: str = "close") -> pd.Series:
        """Values of ``field`` on the decision day (NaN where the ticker did not trade)."""
        return pd.Series(self._market.array(field)[self.index], index=self.tickers, name=self.date)

    @property
    def open(self) -> pd.DataFrame:
        return self.history("open")

    @property
    def high(self) -> pd.DataFrame:
        return self.history("high")

    @property
    def low(self) -> pd.DataFrame:
        return self.history("low")

    @property
    def close(self) -> pd.DataFrame:
        return self.history("close")

    @property
    def adj_close(self) -> pd.DataFrame:
        return self.history("adj_close")

    @property
    def volume(self) -> pd.DataFrame:
        return self.history("volume")

    def returns(self, lookback: Optional[int] = None) -> pd.DataFrame:
        """Daily total returns over the window (first row dropped)."""
        n = (self.lookback if lookback is None else int(lookback)) + 1
        px = self.history("adj_close", n)
        return px.pct_change(fill_method=None).iloc[1:]

    @property
    def tradable(self) -> List[str]:
        """Tickers with a valid close on the decision day."""
        c = self._market.array("close")[self.index]
        return [t for t, ok in zip(self.tickers, np.isfinite(c)) if ok]

    def tradable_mask(self) -> np.ndarray:
        return np.isfinite(self._market.array("close")[self.index])

    # -- features ---------------------------------------------------------------
    @property
    def feature_names(self) -> List[str]:
        return [] if self._features is None else self._features.names

    def feature(self, name: str, lookback: Optional[int] = None) -> pd.DataFrame:
        if self._features is None:
            raise KeyError("no features configured")
        sl = self._rows(lookback)
        return pd.DataFrame(
            self._features.array(name)[sl], index=self._market.dates[sl], columns=self.tickers, copy=False
        )

    def features_now(self) -> pd.DataFrame:
        """All features on the decision day as a (tickers x features) table."""
        if self._features is None:
            return pd.DataFrame(index=self.tickers)
        return pd.DataFrame({n: self._features.array(n)[self.index] for n in self._features.names}, index=self.tickers)

    def __repr__(self) -> str:
        return f"Observation({self.date.date()}, equity={self.portfolio.equity:,.2f})"
