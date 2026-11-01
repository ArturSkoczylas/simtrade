"""A template agent you can point a YAML experiment at (see momentum.yaml / custom.yaml).

The simulator calls ``act`` after the close of every rebalance day. Return target weights
(fractions of equity) keyed by ticker; anything you leave out is sold; return None to do nothing.
"""

import numpy as np

from simtrade import Agent
from simtrade.agents import liquid_tickers


class ShortTermReversal(Agent):
    """Buy the biggest losers of the last week among liquid names (a classic reversal effect)."""

    def __init__(self, lookback: int = 5, n_positions: int = 20, pool: int = 150):
        self.lookback = lookback
        self.n_positions = n_positions
        self.pool = pool
        self.name = f"Reversal({lookback}d, {n_positions})"

    def fit(self, market, features=None):
        # Optional: learn parameters from a training window (used by walk_forward).
        pass

    def act(self, obs):
        names = liquid_tickers(obs, n=self.pool)
        px = obs.history("adj_close", self.lookback + 1)[names]
        if len(px) <= self.lookback:
            return None
        ret = (px.iloc[-1] / px.iloc[0] - 1).replace([np.inf, -np.inf], np.nan).dropna()
        losers = ret.nsmallest(self.n_positions).index
        return {t: 1.0 / len(losers) for t in losers}
