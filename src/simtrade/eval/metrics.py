"""Performance metrics for equity curves."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def drawdown(equity: pd.Series) -> pd.Series:
    """Drawdown series (0 at highs, negative below)."""
    peak = equity.cummax()
    return equity / peak - 1.0


def _max_dd_duration(dd: pd.Series) -> int:
    """Longest stretch (in periods) spent below a previous high."""
    longest = cur = 0
    for under in (dd < 0).to_numpy():
        cur = cur + 1 if under else 0
        longest = max(longest, cur)
    return int(longest)


def compute_metrics(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
    turnover: Optional[pd.Series] = None,
    costs: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Standard metrics for an equity curve indexed by date.

    Returns a flat dict: total_return, cagr, volatility, sharpe, sortino, max_drawdown,
    max_drawdown_days, calmar, hit_rate, best_day, worst_day, and when given, annual turnover,
    total costs, and alpha/beta/tracking error/information ratio against ``benchmark``.
    """
    eq = pd.Series(equity).dropna()
    out: Dict[str, float] = {}
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return {"total_return": float("nan")}
    r = eq.pct_change().dropna()
    years = len(r) / periods_per_year
    total = eq.iloc[-1] / eq.iloc[0] - 1
    rf_d = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    ex = r - rf_d
    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(periods_per_year)
    dd = drawdown(eq)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else float("nan")
    out.update(
        total_return=float(total),
        cagr=float(cagr),
        volatility=float(vol),
        sharpe=float(ex.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)) if r.std(ddof=1) > 0 else float("nan"),
        sortino=float(ex.mean() * periods_per_year / downside) if downside > 0 else float("nan"),
        max_drawdown=float(dd.min()),
        max_drawdown_days=_max_dd_duration(dd),
        calmar=float(cagr / abs(dd.min())) if dd.min() < 0 and np.isfinite(cagr) else float("nan"),
        hit_rate=float((r > 0).mean()),
        best_day=float(r.max()),
        worst_day=float(r.min()),
        n_days=int(len(r)),
        years=float(years),
    )
    if turnover is not None and len(turnover):
        out["annual_turnover"] = float(pd.Series(turnover).sum() / years) if years > 0 else float("nan")
    if costs is not None and len(costs):
        out["total_costs"] = float(pd.Series(costs).sum())
        out["costs_pct_of_initial"] = float(pd.Series(costs).sum() / eq.iloc[0])
    if benchmark is not None:
        b = pd.Series(benchmark).reindex(eq.index).ffill().dropna()
        if len(b) > 2:
            br = b.pct_change().dropna()
            joined = pd.concat([r, br], axis=1, join="inner").dropna()
            if len(joined) > 2 and joined.iloc[:, 1].var() > 0:
                pr, bm = joined.iloc[:, 0], joined.iloc[:, 1]
                beta = pr.cov(bm) / bm.var()
                alpha_d = (pr - rf_d).mean() - beta * (bm - rf_d).mean()
                active = pr - bm
                te = active.std(ddof=1) * np.sqrt(periods_per_year)
                out.update(
                    benchmark_total_return=float(b.iloc[-1] / b.iloc[0] - 1),
                    beta=float(beta),
                    alpha=float(alpha_d * periods_per_year),
                    tracking_error=float(te),
                    information_ratio=float(active.mean() * periods_per_year / te) if te > 0 else float("nan"),
                    correlation=float(pr.corr(bm)),
                )
    return out


def format_metrics(m: Dict[str, float]) -> str:
    """Human-readable multi-line rendering of :func:`compute_metrics` output."""
    pct = {
        "total_return",
        "cagr",
        "volatility",
        "max_drawdown",
        "hit_rate",
        "best_day",
        "worst_day",
        "benchmark_total_return",
        "alpha",
        "tracking_error",
        "costs_pct_of_initial",
        "annual_turnover",
    }
    lines = []
    for k, v in m.items():
        if isinstance(v, float) and np.isnan(v):
            s = "n/a"
        elif k in pct:
            s = f"{v:8.2%}"
        elif isinstance(v, float):
            s = f"{v:8.3f}" if abs(v) < 1e5 else f"{v:,.0f}"
        else:
            s = f"{v:8d}" if isinstance(v, int) else str(v)
        lines.append(f"{k:<24}{s}")
    return "\n".join(lines)
