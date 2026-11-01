"""simtrade: a daily trading simulator for London Stock Exchange equities.

Typical use::

    import simtrade as st

    store = st.DataStore()                                   # ~/.simtrade by default
    market = store.load_market(["VOD.L", "BP.L"], start="2018-01-01", lookback=252)
    result = st.run_backtest(market, st.agents.EqualWeight(), st.SimulationConfig(start="2018-01-01"))
    print(result.summary())
"""

from . import agents, data, eval, features
from .agents import Agent
from .config import BrokerConfig, CostModel, SimulationConfig
from .data import Instrument, Universe, Updater, YFinanceProvider
from .features import FeatureProvider, FeatureSet
from .sim import BacktestResult, MarketData, Observation, Simulation, run_backtest
from .store import DataStore

__version__ = "0.1.0"

__all__ = [
    "agents",
    "data",
    "eval",
    "features",
    "Agent",
    "BrokerConfig",
    "CostModel",
    "SimulationConfig",
    "Instrument",
    "Universe",
    "Updater",
    "YFinanceProvider",
    "FeatureProvider",
    "FeatureSet",
    "BacktestResult",
    "MarketData",
    "Observation",
    "Simulation",
    "run_backtest",
    "DataStore",
    "__version__",
]
