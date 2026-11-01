"""Plugging in alternative data (Reddit, news, sentiment, ...) as a feature provider.

Your signal just needs to be a daily table indexed by date with one column per ticker.
Important: the value on day t must be known by the close of day t (e.g. only posts
published before 16:30 London time), otherwise the backtest has look-ahead bias.
Run ``check_causality`` on your provider - it catches the most common mistakes.
"""

import pandas as pd

import simtrade as st
from simtrade.data.sample import make_sample_store
from simtrade.features import FeatureProvider, check_causality


class CSVSignal(FeatureProvider):
    """Loads a long CSV with columns date, ticker, value (e.g. daily Reddit mention counts)."""

    name = "csv_signal"

    def __init__(self, path: str, feature_name: str = "signal"):
        self.path, self.feature_name = path, feature_name

    def compute(self, market):
        df = pd.read_csv(self.path, parse_dates=["date"])
        wide = df.pivot_table(index="date", columns="ticker", values="value", aggfunc="sum")
        # Align to trading sessions; posts on non-trading days roll forward to the next session.
        wide = wide.reindex(wide.index.union(market.dates)).sort_index()
        wide = wide.fillna(0).rolling("3D").sum()  # causal smoothing: only past rows
        return {self.feature_name: wide.reindex(market.dates)}


class FollowTheCrowd(st.Agent):
    name = "FollowTheCrowd"

    def act(self, obs):
        s = obs.features_now()["signal"].dropna()
        top = s[s > 0].nlargest(5).index
        return {t: 1 / len(top) for t in top} if len(top) else {}


if __name__ == "__main__":
    store = make_sample_store("altdata-store", n_tickers=20)
    market = store.load_market(start="2018-01-01", lookback=60, universe=store.load_universe())

    # Fake "mentions" file for the demo.
    import numpy as np

    rng = np.random.default_rng(0)
    rows = [(d, t, rng.poisson(2)) for d in market.dates[::3] for t in market.tickers[:10]]
    pd.DataFrame(rows, columns=["date", "ticker", "value"]).to_csv("mentions.csv", index=False)

    provider = CSVSignal("mentions.csv")
    print("causality problems:", check_causality(provider, market) or "none")
    res = st.run_backtest(
        market, FollowTheCrowd(), st.SimulationConfig(start="2018-01-01", rebalance="weekly"), features=[provider]
    )
    print(res.summary())
