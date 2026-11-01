"""Train a PPO agent with stable-baselines3 and evaluate it out of sample.

    pip install "simtrade[rl]" stable-baselines3
    python examples/train_ppo.py

Uses the synthetic store so it runs anywhere; swap in ``DataStore()`` and real tickers
(e.g. ["VOD.L", "BP.L", "HSBA.L", "AZN.L", "ULVR.L"]) once you have downloaded data.
"""

from stable_baselines3 import PPO

import simtrade as st
from simtrade.data.sample import make_sample_store
from simtrade.envs import PolicyAgent, TradingEnv
from simtrade.features import Returns, Volatility

store = make_sample_store("ppo-store", n_tickers=10)
tickers = store.tickers(kind="equity")[:8]
market = store.load_market(tickers, start="2016-01-01", lookback=60)
features = [Returns((5, 21)), Volatility((21,))]

train_cfg = st.SimulationConfig(start="2016-01-01", end="2021-12-31")
test_cfg = st.SimulationConfig(start="2022-01-01", end="2024-12-31")

env = TradingEnv(market, train_cfg, features=features, window=20, turnover_penalty=0.001, episode_length=252, seed=0)
model = PPO("MlpPolicy", env, verbose=0, seed=0)
model.learn(total_timesteps=50_000)

# Out-of-sample backtest with the very same engine, costs and timing.
agent = PolicyAgent(model, env)
oos = st.run_backtest(market, agent, test_cfg, features=features)
baseline = st.run_backtest(market, st.agents.EqualWeight(None), test_cfg)
print(oos.summary())
print("\nEqual-weight baseline:\n", baseline.summary())
