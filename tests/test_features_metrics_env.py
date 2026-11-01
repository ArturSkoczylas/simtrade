import numpy as np
import pandas as pd
import pytest
from conftest import make_market

from simtrade import Agent, SimulationConfig, run_backtest
from simtrade.data.providers import YFinanceProvider
from simtrade.eval.metrics import compute_metrics, drawdown
from simtrade.eval.walkforward import walk_forward
from simtrade.features import FeatureProvider, Liquidity, Momentum, Returns, Volatility, check_causality


# -- features -------------------------------------------------------------------------------
def test_builtin_features_are_causal():
    mk = make_market(n_days=200)
    for p in (Returns((1, 5)), Volatility((10,)), Momentum(60, 5), Liquidity(10)):
        assert check_causality(p, mk) == [], p


class Leaky(FeatureProvider):
    name = "leaky"

    def compute(self, market):
        return {"next_ret": market.frame("close").pct_change(fill_method=None).shift(-1)}


def test_causality_check_catches_lookahead():
    mk = make_market(n_days=200)
    assert check_causality(Leaky(), mk)


def test_features_visible_in_observation_up_to_today():
    mk = make_market(n_days=50)
    seen = []

    class Look(Agent):
        def act(self, obs):
            f = obs.feature("ret_1", 3)
            seen.append((obs.date, f.index.max(), obs.features_now().columns.tolist()))
            return None

    run_backtest(mk, Look(), SimulationConfig(), features=[Returns((1,))])
    for d, fmax, cols in seen:
        assert fmax == d and cols == ["ret_1"]


# -- metrics --------------------------------------------------------------------------------
def test_metrics_on_known_curve():
    idx = pd.bdate_range("2020-01-01", periods=253)
    eq = pd.Series(100 * 1.0004 ** np.arange(253), index=idx)
    m = compute_metrics(eq)
    assert m["total_return"] == pytest.approx(1.0004**252 - 1)
    assert m["cagr"] == pytest.approx(1.0004**252 - 1, rel=1e-9)
    assert m["max_drawdown"] == 0.0
    dd = drawdown(pd.Series([100, 120, 90, 130.0]))
    assert dd.min() == pytest.approx(-0.25)


def test_metrics_beta_of_benchmark_is_one():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(1)
    b = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=idx)
    m = compute_metrics(b, benchmark=b)
    assert m["beta"] == pytest.approx(1.0)
    assert m["alpha"] == pytest.approx(0.0, abs=1e-10)


# -- walk-forward ------------------------------------------------------------------------------
def test_walk_forward_fit_never_sees_test_data():
    mk = make_market(n_days=800, tickers=tuple(f"T{i}.L" for i in range(5)))
    fitted = []

    class Fit(Agent):
        def fit(self, market, features=None):
            fitted.append(market.dates[-1])
            self.end = market.dates[-1]

        def act(self, obs):
            assert obs.date > self.end
            return {"T0.L": 1.0}

    wf = walk_forward(mk, Fit, train_years=1, test_years=0.5)
    assert len(wf.folds) >= 3
    for f, fit_end in zip(wf.folds, fitted):
        assert fit_end < f.test_start
        assert f.result.equity.index[0] >= f.test_start
    eq = wf.equity_curve
    assert eq.index.is_monotonic_increasing and not eq.index.has_duplicates


# -- yfinance parsing (no network) ----------------------------------------------------------------
def _fake_yf_frame(tickers):
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=5), name="Date")
    fields = ["Adj Close", "Close", "Dividends", "High", "Low", "Open", "Stock Splits", "Volume"]
    cols = pd.MultiIndex.from_product([tickers, fields], names=["Ticker", "Price"])
    data = np.tile(np.array([99.0, 100.0, 0.0, 101.0, 98.0, 99.5, 0.0, 1e5]), (5, len(tickers)))
    df = pd.DataFrame(data, index=idx, columns=cols)
    if len(tickers) > 1:
        df.loc[:, (tickers[1], slice(None))] = np.nan  # second ticker returned nothing
    return df


def test_yfinance_provider_parses_multiindex(monkeypatch):
    import yfinance as yf

    captured = {}

    def fake_download(**kw):
        captured.update(kw)
        return _fake_yf_frame(kw["tickers"])

    monkeypatch.setattr(yf, "download", fake_download)
    prov = YFinanceProvider(repair=False, lookup_currency=False)
    res = prov.fetch(["VOD.L", "DEAD.L"], "2024-01-01", "2024-01-05")
    assert res["VOD.L"].ok and not res["DEAD.L"].ok
    df = res["VOD.L"].data
    assert df["adj_close"].iloc[0] == 99.0 and df["close"].iloc[0] == 100.0
    assert captured["auto_adjust"] is False and captured["actions"] is True
    assert captured["end"] == "2024-01-06"  # yfinance end is exclusive
    assert prov.currency("VOD.L") == "GBp"


# -- gymnasium env ---------------------------------------------------------------------------
def test_trading_env_runs_and_matches_backtest():
    pytest.importorskip("gymnasium")
    from simtrade.envs import PolicyAgent, TradingEnv

    mk = make_market(n_days=80)
    env = TradingEnv(mk, SimulationConfig(), features=[Returns((5,))], window=5)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    total, steps, done = 0.0, 0, False
    while not done:
        obs, r, done, _trunc, info = env.step(env.action_space.sample())
        total += r
        steps += 1
    assert steps == len(mk.dates) - 1
    assert np.isclose(np.exp(total), info["equity"] / 100_000, rtol=1e-9)

    class Constant:
        def predict(self, x, deterministic=True):
            a = np.zeros(env.n_assets + 1)
            a[0] = 1.0
            return a, None

    res = run_backtest(mk, PolicyAgent(Constant(), env), SimulationConfig())
    assert res.equity["n_positions"].max() >= 1


def test_env_random_episodes():
    pytest.importorskip("gymnasium")
    from simtrade.envs import TradingEnv

    env = TradingEnv(make_market(n_days=200), episode_length=30, seed=1)
    env.reset()
    n = 0
    done = False
    while not done:
        _, _, done, _, _ = env.step(np.zeros(env.n_assets + 1))
        n += 1
    assert n == 30
