"""Static charts for backtest results (requires matplotlib)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union


def plot_result(result, path: Optional[Union[str, Path]] = None, others: Iterable = ()):
    """Equity curve (with benchmark and any other results) above a drawdown panel.

    Returns the matplotlib Figure; also saves it to ``path`` when given.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    series = [(result.agent or "agent", result.equity_curve)] + [(o.agent or "other", o.equity_curve) for o in others]
    for name, eq in series:
        ax1.plot(eq.index, eq.values, lw=1.4, label=name)
    if result.benchmark is not None:
        ax1.plot(result.benchmark.index, result.benchmark.values, lw=1.1, color="0.45", ls="--", label="benchmark")
    ax1.set_ylabel(f"Equity ({result.config.get('base_currency', '')})")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, loc="upper left")
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    for _name, eq in series:
        dd = eq / eq.cummax() - 1
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.35, lw=0)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig
