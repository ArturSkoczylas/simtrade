"""Baseline agents. Useful as benchmarks and as templates for your own agents."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from ..sim.observation import Observation
from .base import Agent


def liquid_tickers(obs: Observation, n: Optional[int] = None, min_value: float = 0.0, window: int = 20) -> list:
    """Tickers tradable today, ranked by median daily traded value over ``window`` sessions.

    Returns the top ``n`` (all if None) with median traded value >= ``min_value``. Most of the
    ~2,000 LSE instruments are thinly traded, so filtering on liquidity is usually step one.
    """
    dv = (obs.history("close", window) * obs.history("volume", window)).median()
    tradable = obs.tradable_mask()
    dv = dv[tradable]
    if dv.isna().all():
        # No volume data at all: fall back to everything tradable.
        names = list(dv.index)
        return names if n is None else names[:n]
    dv = dv.dropna()
    dv = dv[dv >= min_value].sort_values(ascending=False)
    return list(dv.index if n is None else dv.index[:n])


class CashAgent(Agent):
    """Never invests. Handy for checking interest and accounting."""

    name = "Cash"

    def act(self, obs):
        return {}


class BuyAndHold(Agent):
    """Buys once (equal weight or given weights) and never trades again."""

    def __init__(self, weights: Optional[Mapping[str, float]] = None, max_assets: Optional[int] = None):
        self.weights = dict(weights) if weights else None
        self.max_assets = max_assets
        self._done = False
        self.name = "BuyAndHold"

    def reset(self, tickers, config):
        self._done = False

    def act(self, obs):
        if self._done:
            return None
        self._done = True
        if self.weights:
            return self.weights
        names = liquid_tickers(obs, self.max_assets)
        return {t: 1.0 / len(names) for t in names} if names else {}


class EqualWeight(Agent):
    """Equal weight across the ``max_assets`` most liquid tradable names (all if None)."""

    def __init__(self, max_assets: Optional[int] = 50, min_traded_value: float = 0.0):
        self.max_assets = max_assets
        self.min_traded_value = min_traded_value
        self.name = f"EqualWeight({max_assets})"

    def act(self, obs):
        names = liquid_tickers(obs, self.max_assets, self.min_traded_value)
        return {t: 1.0 / len(names) for t in names} if names else {}


class RandomAgent(Agent):
    """Holds ``n`` random tradable names, re-drawn every rebalance. A sanity-check baseline."""

    def __init__(self, n: int = 20, seed: int = 0, liquid_pool: Optional[int] = 200):
        self.n, self.seed, self.liquid_pool = n, seed, liquid_pool
        self.rng = np.random.default_rng(seed)
        self.name = f"Random({n})"

    def reset(self, tickers, config):
        self.rng = np.random.default_rng(self.seed)

    def act(self, obs):
        pool = liquid_tickers(obs, self.liquid_pool)
        if not pool:
            return {}
        pick = self.rng.choice(pool, size=min(self.n, len(pool)), replace=False)
        return {t: 1.0 / len(pick) for t in pick}


class MomentumAgent(Agent):
    """Cross-sectional momentum: long the ``top_k`` names with the best return from
    ``t-lookback`` to ``t-skip``, among the ``liquid_pool`` most liquid names."""

    def __init__(self, lookback: int = 252, skip: int = 21, top_k: int = 20, liquid_pool: Optional[int] = 200):
        if skip >= lookback:
            raise ValueError("skip must be < lookback")
        self.lookback, self.skip, self.top_k, self.liquid_pool = lookback, skip, top_k, liquid_pool
        self.name = f"Momentum({lookback},{skip},top{top_k})"

    def act(self, obs):
        px = obs.history("adj_close", self.lookback + 1)
        if len(px) < self.lookback + 1:
            return None  # not enough history yet
        pool = liquid_tickers(obs, self.liquid_pool)
        px = px[pool]
        start, end = px.iloc[0], px.iloc[-1 - self.skip]
        mom = (end / start - 1).replace([np.inf, -np.inf], np.nan).dropna()
        winners = mom.sort_values(ascending=False).index[: self.top_k]
        return {t: 1.0 / len(winners) for t in winners} if len(winners) else {}


class FixedWeights(Agent):
    """Rebalances back to fixed weights on every rebalance day."""

    def __init__(self, weights: Mapping[str, float]):
        self.weights = dict(weights)
        self.name = "FixedWeights"

    def act(self, obs):
        return self.weights
