"""Gymnasium environment wrapping the simulator, for reinforcement-learning agents.

Requires ``pip install simtrade[rl]``. The environment uses the same :class:`Simulation`
(same broker, costs and timing) as regular backtests, so a trained policy can be evaluated
with :func:`simtrade.run_backtest` via :class:`PolicyAgent`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("TradingEnv needs gymnasium: pip install 'simtrade[rl]'") from exc

from ..agents.base import Agent
from ..config import SimulationConfig
from ..features.base import FeatureProvider, FeatureSet
from ..sim.engine import Simulation
from ..sim.market import MarketData


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


class TradingEnv(gym.Env):
    """Portfolio-allocation environment over a fixed set of tickers.

    * **Observation** (flat ``float32`` vector): the last ``window`` daily log returns of each
      asset, the current value of every feature for each asset, and the current portfolio
      weights (assets + cash). NaNs are replaced by 0.
    * **Action**: ``n_assets + 1`` values in [-1, 1], scaled by ``action_scale`` and mapped by
      softmax to long-only weights (last = cash). A larger scale allows more concentrated portfolios.
    * **Reward**: log growth of equity over the step, minus ``turnover_penalty x turnover``.
    * **Episode**: from the config's start to end, or a random ``episode_length`` window when given.

    Keep the asset set small (tens of names); select tickers with ``market.select([...])`` first.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        market: MarketData,
        config: Optional[SimulationConfig] = None,
        features: Optional[Sequence[FeatureProvider]] = None,
        window: int = 20,
        turnover_penalty: float = 0.0,
        episode_length: Optional[int] = None,
        seed: Optional[int] = None,
        action_scale: float = 5.0,
    ):
        super().__init__()
        self.action_scale = float(action_scale)
        cfg = config or SimulationConfig()
        if cfg.broker.allow_short:
            raise ValueError("TradingEnv is long-only; set broker.allow_short=False")
        self.market = market
        self.window = int(window)
        self.turnover_penalty = float(turnover_penalty)
        self.episode_length = episode_length
        self.base_config = cfg
        self.featureset = FeatureSet.compute(list(features), market) if features else None
        self.sim = Simulation(market, cfg, self.featureset)
        self.n_assets = len(market.tickers)
        n_feat = len(self.featureset) if self.featureset is not None else 0
        self.obs_dim = self.window * self.n_assets + n_feat * self.n_assets + self.n_assets + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_assets + 1,), dtype=np.float32)
        logp = np.log(market.frame("adj_close").ffill())
        self._logret = logp.diff().to_numpy()
        self._rng = np.random.default_rng(seed)
        self._first, self._last = self.sim.start_index, self.sim.end_index

    def _obs(self) -> np.ndarray:
        t = self.sim.t
        lo = max(0, t - self.window + 1)
        r = self._logret[lo : t + 1]
        if r.shape[0] < self.window:
            r = np.vstack([np.zeros((self.window - r.shape[0], self.n_assets)), r])
        parts = [r.ravel()]
        if self.featureset is not None:
            parts += [self.featureset.array(n)[t] for n in self.featureset.names]
        st = self.sim.portfolio_state()
        w = st.weights.to_numpy()
        parts += [w, np.array([st.cash / st.equity if st.equity else 1.0])]
        return np.nan_to_num(np.concatenate(parts), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        start, end = self._first, self._last
        if self.episode_length and (self._last - self._first) > self.episode_length:
            start = int(self._rng.integers(self._first, self._last - self.episode_length))
            end = start + self.episode_length
        self.sim.start_index, self.sim.end_index = start, end
        self.sim._rebalance[:] = False
        self.sim._rebalance[start : end + 1] = True
        self.sim.reset()
        return self._obs(), {"date": str(self.sim.date.date()), "equity": self.sim.equity}

    def action_to_weights(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype="float64"), -1.0, 1.0)
        w = softmax(a * self.action_scale)
        return w[:-1]

    def step(self, action):
        before = self.sim.equity
        self.sim.submit(self.action_to_weights(action))
        row = self.sim.step()
        after = row["equity"]
        reward = float(
            np.log(max(after, 1e-12) / max(before, 1e-12)) - self.turnover_penalty * (row["turnover"] or 0.0)
        )
        terminated = self.sim.done or after <= 0
        info = {"date": str(self.sim.date.date()), "equity": after, "turnover": row["turnover"]}
        return self._obs(), reward, bool(terminated), False, info


class PolicyAgent(Agent):
    """Wraps a trained policy (anything with ``predict(obs) -> action``, e.g. a stable-baselines3
    model) so it can run in a normal backtest. The backtest must use the same ``MarketData`` the
    ``env`` was built on (same tickers and sessions); start/end dates may differ."""

    def __init__(self, policy, env: TradingEnv, deterministic: bool = True):
        self.policy, self.env, self.deterministic = policy, env, deterministic
        self.name = f"Policy({type(policy).__name__})"

    def act(self, obs):
        # Rebuild the env-style observation from the backtest's observation.
        env = self.env
        t = obs.index
        lo = max(0, t - env.window + 1)
        r = env._logret[lo : t + 1]
        if r.shape[0] < env.window:
            r = np.vstack([np.zeros((env.window - r.shape[0], env.n_assets)), r])
        parts = [r.ravel()]
        if env.featureset is not None:
            parts += [env.featureset.array(n)[t] for n in env.featureset.names]
        st = obs.portfolio
        parts += [st.weights.to_numpy(), np.array([st.cash / st.equity if st.equity else 1.0])]
        x = np.nan_to_num(np.concatenate(parts), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        out = self.policy.predict(x, deterministic=self.deterministic)
        action = out[0] if isinstance(out, tuple) else out
        return env.action_to_weights(action)
