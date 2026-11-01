"""The simulation engine.

Timeline for each session ``t`` (decision made after the close of ``t``)::

    close of t   agent.act(Observation up to t) -> target weights
    t+1          1. delisted holdings are cashed out at their last price
                 2. dividends with ex-date t+1 are credited (debited for shorts)
                 3. orders fill at the open of t+1 ('next_open') or its close ('next_close')
                 4. borrow fees and cash interest accrue
                 5. portfolio is marked to the close of t+1 and recorded

With ``execution='same_close'`` orders fill at the close of t right after the decision.

:class:`Simulation` is steppable (``reset / observe / submit / step``) so the same engine
drives both :meth:`Simulation.run` backtests and the Gymnasium environment.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..config import SimulationConfig
from ..features.base import FeatureProvider, FeatureSet
from .broker import Broker, ExecutionReport
from .market import MarketData
from .observation import Observation, PortfolioState
from .portfolio import Portfolio
from .results import BacktestResult

log = logging.getLogger(__name__)

Weights = Union[None, Mapping[str, float], pd.Series, np.ndarray, Sequence[float]]

_EQUITY_COLS = [
    "equity",
    "cash",
    "long_value",
    "short_value",
    "gross_exposure",
    "net_exposure",
    "n_positions",
    "traded_value",
    "turnover",
    "commission",
    "stamp_duty",
    "ptm_levy",
    "slippage",
    "borrow_cost",
    "interest",
    "dividends",
]


class Simulation:
    """A day-by-day trading simulation over a :class:`MarketData` panel."""

    def __init__(
        self,
        market: MarketData,
        config: Optional[SimulationConfig] = None,
        features: Union[None, FeatureSet, Iterable[FeatureProvider]] = None,
        benchmark: Optional[pd.Series] = None,
    ):
        self.market = market
        self.config = config or SimulationConfig()
        if features is None or isinstance(features, FeatureSet):
            self.features = features
        else:
            self.features = FeatureSet.compute(list(features), market)
        self.broker = Broker(self.config.broker, market.tickers, market.stamp_duty_mask())

        n = len(market.dates)
        if n == 0 or not market.tickers:
            raise ValueError("market data is empty")
        close = market.frame("close")
        any_price = np.isfinite(market.array("close")).any(axis=1)
        if not any_price.any():
            raise ValueError("market data has no prices")
        self.start_index = (
            market.index_of(self.config.start, "left") if self.config.start else int(np.argmax(any_price))
        )
        self.end_index = market.index_of(self.config.end, "right") if self.config.end else n - 1
        if self.start_index >= n or self.end_index < self.start_index:
            raise ValueError(
                f"no sessions between start={self.config.start} and end={self.config.end} "
                f"(data covers {market.dates[0].date()}..{market.dates[-1].date()})"
            )

        # Precomputed, causal helpers.
        self._mark_px = close.ffill().to_numpy()  # last known close at or before each row
        bc = self.config.broker
        dv = close * market.frame("volume")
        self._adv = dv.rolling(bc.adv_window, min_periods=1).mean().to_numpy()
        self._rebalance = self._rebalance_mask()
        self.benchmark = benchmark
        self.reset()

    # -- schedule -----------------------------------------------------------------
    def _rebalance_mask(self) -> np.ndarray:
        dates = self.market.dates
        n = len(dates)
        mask = np.zeros(n, dtype=bool)
        freq = self.config.rebalance
        s, e = self.start_index, self.end_index
        if isinstance(freq, int):
            mask[s : e + 1] = ((np.arange(n) - s) % freq == 0)[s : e + 1]
        elif freq == "daily":
            mask[s : e + 1] = True
        else:
            code = {"weekly": "W", "monthly": "M", "quarterly": "Q"}[freq]
            periods = dates.to_period(code)
            # Decide on the last session of each period so trades land at the start of the next.
            last = np.r_[periods[1:] != periods[:-1], True]
            mask[s : e + 1] = last[s : e + 1]
        mask[s] = True  # always allow the initial allocation
        return mask

    def is_rebalance_day(self, index: Optional[int] = None) -> bool:
        return bool(self._rebalance[self.t if index is None else index])

    # -- state --------------------------------------------------------------------
    def reset(self) -> None:
        m = len(self.market.tickers)
        self.t = self.start_index
        self.portfolio = Portfolio(m, self.config.initial_cash)
        self._pending: Optional[np.ndarray] = None
        self._records: List[dict] = []
        self._fills: List[dict] = []
        self._events: List[dict] = []
        self._positions: List[dict] = []
        self._record(self.t, ExecutionReport(), borrow=0.0, interest=0.0, dividends=0.0)

    @property
    def done(self) -> bool:
        return self.t >= self.end_index

    @property
    def date(self) -> pd.Timestamp:
        return self.market.dates[self.t]

    @property
    def equity(self) -> float:
        return self.portfolio.equity(self._mark_px[self.t])

    def portfolio_state(self) -> PortfolioState:
        px = self._mark_px[self.t]
        vals = self.portfolio.values(px)
        eq = float(self.portfolio.cash + vals.sum())
        tick = self.market.tickers
        return PortfolioState(
            cash=float(self.portfolio.cash),
            equity=eq,
            shares=pd.Series(self.portfolio.shares.copy(), index=tick, name="shares"),
            weights=pd.Series(vals / eq if eq else np.zeros_like(vals), index=tick, name="weights"),
        )

    def observe(self) -> Observation:
        """Observation for a decision after the close of the current session."""
        return Observation(self.market, self.t, self.config.lookback, self.portfolio_state(), self.features)

    # -- actions --------------------------------------------------------------------
    def to_weight_array(self, weights: Weights) -> Optional[np.ndarray]:
        """Convert agent output to a weight vector. Tickers not mentioned get weight 0."""
        if weights is None:
            return None
        m = len(self.market.tickers)
        if isinstance(weights, (pd.Series, Mapping)):
            items = weights.items()
            arr = np.zeros(m)
            unknown = []
            for k, v in items:
                j = self.market._col.get(k)
                if j is None:
                    unknown.append(k)
                    continue
                arr[j] = float(v)
            if unknown:
                log.warning("Ignoring weights for unknown tickers: %s", unknown[:10])
            return arr
        arr = np.asarray(weights, dtype="float64").ravel()
        if arr.shape[0] != m:
            raise ValueError(f"weight vector has length {arr.shape[0]}, expected {m}")
        return arr

    def submit(self, weights: Weights) -> None:
        """Submit target weights decided after the close of the current session.

        ``None`` means "no change". With ``execution='same_close'`` the orders fill immediately at
        today's close; otherwise they fill on the next session.
        """
        w = self.to_weight_array(weights)
        if w is None:
            self._pending = None
            return
        if self.config.broker.execution == "same_close":
            i = self.t
            px = self.market.array("close")[i]
            rep = self._execute(w, i, px)
            # Re-record today's row including the trades.
            last = self._records.pop()
            self._record(i, rep, borrow=last["borrow_cost"], interest=last["interest"], dividends=last["dividends"])
            self._pending = None
        else:
            self._pending = w

    def _tradable(self, i: int, px: np.ndarray) -> np.ndarray:
        vol = self.market.array("volume")[i]
        return np.isfinite(px) & (px > 0) & ~(np.isfinite(vol) & (vol <= 0))

    def _execute(self, w: np.ndarray, i: int, px: np.ndarray) -> ExecutionReport:
        decision_row = self.t  # ADV known at the decision time
        # Untradable holdings are valued at the last close known when the order fills:
        # yesterday's close for an open fill, today's close for a close fill.
        at_open = self.config.broker.execution == "next_open"
        rep = self.broker.rebalance(
            self.portfolio,
            w,
            px,
            mark=self._mark_px[max(i - 1, 0)] if at_open else self._mark_px[i],
            tradable=self._tradable(i, px),
            adv=self._adv[decision_row],
            date=self.market.dates[i],
        )
        self._fills.extend(rep.fills)
        self._events.extend(rep.events)
        if rep.fills:
            self._snapshot_positions(i)
        return rep

    def step(self) -> dict:
        """Advance one session, executing any pending orders. Returns the recorded row."""
        if self.done:
            raise RuntimeError("simulation finished; call reset()")
        i = self.t + 1
        date = self.market.dates[i]
        pf = self.portfolio
        cfg = self.config

        # 1. Delistings: holdings whose data has ended are cashed out at the last close.
        held = np.flatnonzero(pf.shares != 0)
        delisted = held[(self.market.last_valid[held] < i)]
        for j in delisted:
            last_px = self._mark_px[i, j]
            value = pf.shares[j] * last_px
            pf.cash += float(value)
            self._events.append(
                {
                    "date": date,
                    "ticker": self.market.tickers[j],
                    "event": "delisted",
                    "detail": f"cashed out {pf.shares[j]:.4f} sh @ {last_px:.4f} = {value:.2f}",
                }
            )
            pf.shares[j] = 0.0
        if len(delisted):
            self._snapshot_positions(i)

        # 2. Dividends (ex-date i): longs receive, shorts pay.
        div = self.market.array("dividends")[i]
        div_cash = float(np.sum(pf.shares * div))
        if div_cash:
            pf.cash += div_cash

        # 3. Pending orders.
        rep = ExecutionReport()
        if self._pending is not None:
            field = "open" if cfg.broker.execution == "next_open" else "close"
            px = self.market.array(field)[i]
            rep = self._execute(self._pending, i, px)
            self._pending = None

        # 4. Financing.
        mark = self._mark_px[i]
        vals = pf.values(mark)
        short_val = -vals[vals < 0].sum()
        borrow = short_val * cfg.broker.costs.borrow_rate / 252.0
        interest = max(pf.cash, 0.0) * cfg.broker.cash_interest_rate / 252.0
        pf.cash += interest - borrow

        self.t = i
        return self._record(i, rep, borrow=borrow, interest=interest, dividends=div_cash)

    # -- bookkeeping ------------------------------------------------------------------
    def _snapshot_positions(self, i: int) -> None:
        pf = self.portfolio
        px = self._mark_px[i]
        eq = pf.equity(px)
        date = self.market.dates[i]
        nz = np.flatnonzero(pf.shares != 0)
        # Replace an earlier snapshot of the same day.
        while self._positions and self._positions[-1]["date"] == date:
            self._positions.pop()
        if len(nz) == 0:
            self._positions.append({"date": date, "ticker": None, "shares": 0.0, "value": 0.0, "weight": 0.0})
        for j in nz:
            v = pf.shares[j] * px[j]
            self._positions.append(
                {
                    "date": date,
                    "ticker": self.market.tickers[j],
                    "shares": float(pf.shares[j]),
                    "value": float(v),
                    "weight": float(v / eq) if eq else 0.0,
                }
            )

    def _record(self, i: int, rep: ExecutionReport, borrow: float, interest: float, dividends: float) -> dict:
        pf = self.portfolio
        vals = pf.values(self._mark_px[i])
        eq = float(pf.cash + vals.sum())
        long_v = float(vals[vals > 0].sum())
        short_v = float(-vals[vals < 0].sum())
        row = {
            "date": self.market.dates[i],
            "equity": eq,
            "cash": float(pf.cash),
            "long_value": long_v,
            "short_value": short_v,
            "gross_exposure": (long_v + short_v) / eq if eq else np.nan,
            "net_exposure": (long_v - short_v) / eq if eq else np.nan,
            "n_positions": int(np.count_nonzero(pf.shares)),
            "traded_value": rep.traded_value,
            "turnover": rep.traded_value / eq if eq else np.nan,
            "commission": rep.commission,
            "stamp_duty": rep.stamp_duty,
            "ptm_levy": rep.ptm_levy,
            "slippage": rep.slippage,
            "borrow_cost": borrow,
            "interest": interest,
            "dividends": dividends,
        }
        self._records.append(row)
        return row

    def result(self, agent_name: str = "") -> BacktestResult:
        eq = pd.DataFrame(self._records, columns=["date"] + _EQUITY_COLS).set_index("date")
        trades = pd.DataFrame(
            self._fills,
            columns=[
                "date",
                "ticker",
                "side",
                "shares",
                "price",
                "exec_price",
                "value",
                "commission",
                "stamp_duty",
                "ptm_levy",
                "slippage",
            ],
        )
        events = pd.DataFrame(self._events, columns=["date", "ticker", "event", "detail"])
        positions = pd.DataFrame(self._positions, columns=["date", "ticker", "shares", "value", "weight"])
        bench = None
        if self.benchmark is not None:
            b = pd.Series(self.benchmark).reindex(eq.index).ffill()
            first = b.first_valid_index()
            if first is not None:
                bench = (b / b.loc[first] * self.config.initial_cash).rename("benchmark")
        cfg = self.config.model_dump(mode="json")
        return BacktestResult(eq, trades, events, positions, bench, cfg, agent_name)

    # -- convenience ---------------------------------------------------------------------
    def run(self, agent, progress: Optional[Callable[[int, int], None]] = None) -> BacktestResult:
        """Run ``agent`` from the start to the end of the simulation and return the results."""
        self.reset()
        agent.reset(self.market.tickers, self.config)
        total = self.end_index - self.start_index
        same_close = self.config.broker.execution == "same_close"
        while True:
            if self.is_rebalance_day() and (not self.done or same_close):
                self.submit(agent.act(self.observe()))
            if self.done:
                break
            self.step()
            if progress is not None:
                progress(self.t - self.start_index, total)
        return self.result(getattr(agent, "name", type(agent).__name__))


def run_backtest(
    market: MarketData,
    agent,
    config: Optional[SimulationConfig] = None,
    features: Union[None, FeatureSet, Iterable[FeatureProvider]] = None,
    benchmark: Optional[pd.Series] = None,
) -> BacktestResult:
    """One-call backtest: ``run_backtest(market, MyAgent(), SimulationConfig(...))``."""
    return Simulation(market, config, features, benchmark).run(agent)
