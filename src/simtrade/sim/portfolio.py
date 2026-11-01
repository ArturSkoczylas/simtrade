"""Portfolio accounting: cash plus share holdings aligned to the market's tickers."""

from __future__ import annotations

import numpy as np


class Portfolio:
    """Cash and signed share counts (negative = short) for each ticker column."""

    def __init__(self, n_assets: int, cash: float):
        self.cash = float(cash)
        self.shares = np.zeros(n_assets, dtype="float64")

    def values(self, prices: np.ndarray) -> np.ndarray:
        """Market value per asset at ``prices`` (positions without a price count as 0)."""
        held = self.shares != 0
        out = np.zeros_like(self.shares)
        out[held] = self.shares[held] * prices[held]
        return np.nan_to_num(out, nan=0.0)

    def equity(self, prices: np.ndarray) -> float:
        return float(self.cash + self.values(prices).sum())

    def weights(self, prices: np.ndarray) -> np.ndarray:
        eq = self.equity(prices)
        if eq == 0:
            return np.zeros_like(self.shares)
        return self.values(prices) / eq

    def copy(self) -> "Portfolio":
        p = Portfolio(len(self.shares), self.cash)
        p.shares = self.shares.copy()
        return p
