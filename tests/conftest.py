import numpy as np
import pandas as pd
import pytest

from simtrade.calendar import trading_sessions
from simtrade.data.universe import Instrument, Universe
from simtrade.sim.market import MarketData


def make_market(
    n_days=60, tickers=("AAA.L", "BBB.L", "CCC.L"), seed=0, start="2023-01-03", markets=None, volume=1e6, dividends=None
):
    """Small deterministic market. Open = previous close * 1.001; close random walk."""
    dates = trading_sessions(start, pd.Timestamp(start) + pd.Timedelta(days=int(n_days * 1.6)))[:n_days]
    rng = np.random.default_rng(seed)
    m = len(tickers)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, m)), axis=0))
    open_ = np.vstack([close[:1], close[:-1] * 1.001])
    arrays = {
        "open": open_,
        "high": np.maximum(open_, close) * 1.002,
        "low": np.minimum(open_, close) * 0.998,
        "close": close,
        "adj_close": close,
        "volume": np.full((n_days, m), volume),
        "dividends": np.zeros((n_days, m)) if dividends is None else dividends,
    }
    markets = markets or ["MAIN"] * m
    uni = Universe(Instrument(ticker=t, market=mk, country="United Kingdom") for t, mk in zip(tickers, markets))
    return MarketData(dates, list(tickers), arrays, universe=uni)


@pytest.fixture
def market():
    return make_market()
