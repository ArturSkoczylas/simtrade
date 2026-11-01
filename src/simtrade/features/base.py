"""Feature providers: pluggable, precomputed signals the agent sees in its observation.

A provider computes full (sessions x tickers) panels once, up front. The simulator then
exposes only rows up to the current session, so a provider must be **causal**: the value on
day t may only use data from days <= t. Use :func:`check_causality` to test a provider.

This is the hook for alternative data: a Reddit/news/sentiment provider just has to return a
DataFrame per feature indexed like ``market.dates`` with ``market.tickers`` as columns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


class FeatureProvider(ABC):
    """Base class for feature providers."""

    #: Human-readable provider name.
    name: str = "feature"

    @abstractmethod
    def compute(self, market) -> Dict[str, pd.DataFrame]:
        """Return ``{feature_name: DataFrame(index=market.dates, columns=market.tickers)}``."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class FeatureSet:
    """Read-only container of computed features aligned to a :class:`MarketData`."""

    def __init__(self, arrays: Dict[str, np.ndarray], dates: pd.DatetimeIndex, tickers: Sequence[str]):
        self.dates = dates
        self.tickers = list(tickers)
        self._arrays: Dict[str, np.ndarray] = {}
        for k, a in arrays.items():
            a = np.array(a, dtype="float64", copy=True)
            if a.shape != (len(dates), len(self.tickers)):
                raise ValueError(f"feature {k!r} has shape {a.shape}, expected {(len(dates), len(self.tickers))}")
            a.setflags(write=False)
            self._arrays[k] = a

    @classmethod
    def compute(cls, providers: Iterable[FeatureProvider], market) -> "FeatureSet":
        arrays: Dict[str, np.ndarray] = {}
        for p in providers:
            out = p.compute(market)
            for name, df in out.items():
                if name in arrays:
                    raise ValueError(f"duplicate feature name {name!r} (from {p!r})")
                aligned = pd.DataFrame(df).reindex(index=market.dates, columns=market.tickers)
                arrays[name] = aligned.to_numpy(dtype="float64")
        return cls(arrays, market.dates, market.tickers)

    @property
    def names(self) -> List[str]:
        return list(self._arrays)

    def array(self, name: str) -> np.ndarray:
        return self._arrays[name]

    def frame(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(self._arrays[name], index=self.dates, columns=self.tickers, copy=False)

    def __len__(self) -> int:
        return len(self._arrays)

    def __repr__(self) -> str:
        return f"FeatureSet({self.names})"


def check_causality(
    provider: FeatureProvider,
    market,
    n_checks: int = 5,
    seed: int = 0,
    rtol: float = 1e-7,
    atol: float = 1e-10,
) -> List[str]:
    """Detect look-ahead in a feature provider.

    Features are computed on the full panel and on panels truncated at random dates; the values
    on each truncation date must match. Returns a list of problems (empty = no leak detected).
    """
    full = FeatureSet.compute([provider], market)
    rng = np.random.default_rng(seed)
    n = len(market.dates)
    if n < 3:
        return []
    cut_points = sorted(set(rng.integers(n // 4, n - 1, size=n_checks).tolist()))
    problems = []
    for i in cut_points:
        part = FeatureSet.compute([provider], market.slice(end=market.dates[i]))
        for name in full.names:
            a = full.array(name)[i]
            b = part.array(name)[i]
            same = np.isclose(a, b, rtol=rtol, atol=atol) | (np.isnan(a) & np.isnan(b))
            if not same.all():
                problems.append(f"{name}: differs at {market.dates[i].date()} when future data is removed")
    return problems
