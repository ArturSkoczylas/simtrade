"""Backtest results container."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from ..eval.metrics import compute_metrics, drawdown, format_metrics


@dataclass
class BacktestResult:
    """Everything a run produced.

    Attributes
    ----------
    equity:
        One row per session: equity, cash, exposures, turnover and cost breakdown.
    trades:
        One row per fill.
    events:
        Delistings, rejected/capped orders, dividends, etc.
    positions:
        Holdings snapshot (long format) after every day on which positions changed.
    benchmark:
        Benchmark total-return series rebased to the initial cash, if configured.
    """

    equity: pd.DataFrame
    trades: pd.DataFrame
    events: pd.DataFrame
    positions: pd.DataFrame
    benchmark: Optional[pd.Series] = None
    config: Dict[str, Any] = field(default_factory=dict)
    agent: str = ""

    @property
    def equity_curve(self) -> pd.Series:
        return self.equity["equity"]

    @property
    def returns(self) -> pd.Series:
        return self.equity_curve.pct_change().dropna()

    def drawdown(self) -> pd.Series:
        return drawdown(self.equity_curve)

    def metrics(self) -> Dict[str, float]:
        rf = float(self.config.get("risk_free_rate", 0.0) or 0.0)
        costs = self.equity[["commission", "stamp_duty", "ptm_levy", "slippage", "borrow_cost"]].sum(axis=1)
        return compute_metrics(
            self.equity_curve, self.benchmark, risk_free_rate=rf, turnover=self.equity["turnover"], costs=costs
        )

    def summary(self) -> str:
        head = f"Agent: {self.agent}\nPeriod: {self.equity.index[0].date()} -> {self.equity.index[-1].date()}\n"
        head += f"Trades: {len(self.trades)}\n"
        return head + format_metrics(self.metrics())

    def save(self, path: Union[str, Path], plots: bool = True) -> Path:
        """Write CSV/JSON outputs (and a PNG chart if matplotlib is installed) to ``path``."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.equity.to_csv(out / "equity.csv")
        self.trades.to_csv(out / "trades.csv", index=False)
        self.events.to_csv(out / "events.csv", index=False)
        self.positions.to_csv(out / "positions.csv", index=False)
        if self.benchmark is not None:
            self.benchmark.rename("benchmark").to_csv(out / "benchmark.csv")
        with open(out / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(self.metrics(), fh, indent=2, default=float)
        with open(out / "config.json", "w", encoding="utf-8") as fh:
            json.dump({"agent": self.agent, **self.config}, fh, indent=2, default=str)
        if plots:
            try:
                from ..eval.plots import plot_result

                plot_result(self, out / "equity.png")
            except ImportError:
                pass
        return out
