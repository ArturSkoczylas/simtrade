"""The agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..sim.observation import Observation

TargetWeights = Union[None, Mapping[str, float], pd.Series, np.ndarray]


class Agent(ABC):
    """Base class for trading agents.

    Subclasses implement :meth:`act`, called after the close of every rebalance day with an
    :class:`~simtrade.sim.observation.Observation`. It returns **target portfolio weights**:

    * a ``dict`` / ``pd.Series`` mapping ticker -> weight (tickers not mentioned are sold to 0),
    * or a numpy array aligned with ``obs.tickers``,
    * or ``None`` to keep the current portfolio unchanged.

    Weights are fractions of equity; whatever is not invested stays in cash. Negative weights
    are shorts (only if the broker allows shorting).

    Optional hooks: :meth:`reset` before a run and :meth:`fit` for agents that learn from a
    training window (see :func:`simtrade.eval.walkforward.walk_forward`).
    """

    #: Display name used in results.
    name: Optional[str] = None

    def reset(self, tickers: Sequence[str], config) -> None:  # noqa: D401 - hook
        """Called at the start of each run."""

    def fit(self, market, features=None) -> None:
        """Optional: learn from historical data (only data before the test window is passed)."""

    @abstractmethod
    def act(self, obs: Observation) -> TargetWeights:
        """Return target weights for the next execution."""

    def __repr__(self) -> str:
        return self.name or type(self).__name__
