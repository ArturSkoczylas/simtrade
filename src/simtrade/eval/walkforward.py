"""Walk-forward evaluation: fit on a rolling training window, test on the following window."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional

import pandas as pd

from ..config import SimulationConfig
from ..features.base import FeatureProvider, FeatureSet
from .metrics import compute_metrics

if TYPE_CHECKING:  # avoid an import cycle with simtrade.sim
    from ..sim.market import MarketData
    from ..sim.results import BacktestResult


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    result: Optional["BacktestResult"] = None


@dataclass
class WalkForwardResult:
    folds: List[Fold] = field(default_factory=list)

    @property
    def equity_curve(self) -> pd.Series:
        """Out-of-sample equity, chain-linked across folds (each fold starts with the previous end value)."""
        parts, level = [], None
        for f in self.folds:
            eq = f.result.equity_curve
            scale = 1.0 if level is None else level / eq.iloc[0]
            seg = eq * scale
            if parts:
                seg = seg.iloc[1:] if seg.index[0] <= parts[-1].index[-1] else seg
            parts.append(seg)
            level = seg.iloc[-1]
        return pd.concat(parts) if parts else pd.Series(dtype=float)

    def metrics(self) -> dict:
        return compute_metrics(self.equity_curve)

    def fold_table(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            m = f.result.metrics()
            rows.append(
                {
                    "test_start": f.test_start.date(),
                    "test_end": f.test_end.date(),
                    "total_return": m.get("total_return"),
                    "sharpe": m.get("sharpe"),
                    "max_drawdown": m.get("max_drawdown"),
                }
            )
        return pd.DataFrame(rows)


def make_folds(
    dates: pd.DatetimeIndex,
    train_years: float,
    test_years: float,
    step_years: Optional[float] = None,
    start: Optional[pd.Timestamp] = None,
) -> List[Fold]:
    """Rolling folds over ``dates``: [train | test] windows advancing by ``step_years``."""
    step = pd.DateOffset(months=int(round((step_years or test_years) * 12)))
    train = pd.DateOffset(months=int(round(train_years * 12)))
    test = pd.DateOffset(months=int(round(test_years * 12)))
    t0 = pd.Timestamp(start) if start is not None else dates[0]
    folds = []
    while True:
        tr_s, tr_e = t0, t0 + train - pd.Timedelta(days=1)
        te_s, te_e = t0 + train, t0 + train + test - pd.Timedelta(days=1)
        if te_s > dates[-1]:
            break
        folds.append(Fold(tr_s, tr_e, te_s, min(te_e, dates[-1])))
        t0 = t0 + step
    return folds


def walk_forward(
    market: "MarketData",
    agent_factory: Callable[[], object],
    train_years: float = 3,
    test_years: float = 1,
    step_years: Optional[float] = None,
    config: Optional[SimulationConfig] = None,
    features: Optional[Iterable[FeatureProvider]] = None,
) -> WalkForwardResult:
    """Fit a fresh agent on each training window and test it on the next window.

    ``agent.fit(train_market, train_features)`` only ever sees data up to the training end date.
    The test simulation keeps ``config.lookback`` sessions of earlier history for observations.
    """
    from ..sim.engine import Simulation

    config = config or SimulationConfig()
    providers = list(features or [])
    feats_full = FeatureSet.compute(providers, market) if providers else None
    out = WalkForwardResult()
    for f in make_folds(market.dates, train_years, test_years, step_years):
        train_mkt = market.slice(f.train_start, f.train_end)
        if len(train_mkt.dates) < 2:
            continue
        agent = agent_factory()
        train_feats = FeatureSet.compute(providers, train_mkt) if providers else None
        agent.fit(train_mkt, train_feats)
        cfg = config.model_copy(update={"start": f.test_start.date(), "end": f.test_end.date()})
        sim = Simulation(market, cfg, feats_full)
        f.result = sim.run(agent)
        out.folds.append(f)
    return out
