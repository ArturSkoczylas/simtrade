"""Quickstart: build a synthetic store, write an agent, run a backtest. No internet needed.

python examples/quickstart.py
"""

import simtrade as st
from simtrade.agents import liquid_tickers
from simtrade.data.sample import make_sample_store
from simtrade.features import Returns, Volatility


class LowVolatility(st.Agent):
    """Hold the 10 least volatile of the 30 most liquid stocks, equally weighted."""

    name = "LowVol"

    def act(self, obs):
        pool = liquid_tickers(obs, n=30)
        vol = obs.features_now()["vol_63"].reindex(pool).dropna()
        if vol.empty:
            return None  # not enough history yet: keep the current portfolio
        picks = vol.nsmallest(10).index
        return {t: 1 / len(picks) for t in picks}


if __name__ == "__main__":
    store = make_sample_store("quickstart-store", n_tickers=40)

    config = st.SimulationConfig(
        start="2017-01-01",
        initial_cash=100_000,
        rebalance="monthly",
        benchmark="SAMPLE_IDX.L",
        broker=st.BrokerConfig(costs=st.CostModel(commission_fixed=5.0)),  # £5 a trade + 0.5% stamp duty
    )
    market = store.load_market(start=config.start, lookback=config.lookback, universe=store.load_universe())
    benchmark = store.read_prices("SAMPLE_IDX.L")["adj_close"]

    sim = st.Simulation(market, config, features=[Returns((21,)), Volatility((63,))], benchmark=benchmark)
    result = sim.run(LowVolatility())
    print(result.summary())
    result.save("quickstart-results/lowvol")
    print("\nLast trades:\n", result.trades.tail())
