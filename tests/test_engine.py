import numpy as np
import pandas as pd
import pytest
from conftest import make_market

from simtrade import Agent, BrokerConfig, CostModel, Simulation, SimulationConfig, run_backtest
from simtrade.agents import BuyAndHold, CashAgent, EqualWeight, FixedWeights, MomentumAgent

ZERO = BrokerConfig(costs=CostModel.zero(), max_participation=None)


class Recorder(Agent):
    """Records what it sees; buys AAA on the first call."""

    def __init__(self):
        self.seen = []

    def act(self, obs):
        self.seen.append((obs.date, obs.close.index.max(), obs.returns().index.max() if len(obs.returns()) else None))
        return {"AAA.L": 1.0}


def test_observation_never_contains_future_rows(market):
    agent = Recorder()
    Simulation(market, SimulationConfig(lookback=10, broker=ZERO)).run(agent)
    assert len(agent.seen) == len(market.dates) - 1  # no decision on the final day
    for date, max_close_date, max_ret_date in agent.seen:
        assert max_close_date == date
        assert max_ret_date is None or max_ret_date <= date


def test_observation_arrays_are_read_only(market):
    sim = Simulation(market, SimulationConfig(broker=ZERO))
    obs = sim.observe()
    df = obs.close
    with pytest.raises(ValueError):
        df.iloc[-1, 0] = 0.0  # call .copy() if an agent needs a writable frame
    with pytest.raises(ValueError):
        market.array("close")[0, 0] = 1.0


def test_next_open_execution_price(market):
    cfg = SimulationConfig(broker=ZERO, initial_cash=10_000)
    sim = Simulation(market, cfg)
    sim.submit({"AAA.L": 1.0})
    sim.step()
    res = sim.result()
    fill = res.trades.iloc[0]
    j = market.col("AAA.L")
    assert fill["date"] == market.dates[1]
    assert fill["price"] == pytest.approx(market.array("open")[1, j])
    # Fully invested at open, then marked at the close.
    expected = 10_000 / market.array("open")[1, j] * market.array("close")[1, j]
    assert res.equity_curve.iloc[-1] == pytest.approx(expected)


def test_peeking_agent_cannot_profit_from_execution_timing():
    """An agent that buys whatever went up today gets no edge from today's move under next_open."""
    mk = make_market(n_days=300, tickers=tuple(f"T{i}.L" for i in range(10)), seed=3)

    class ChaseToday(Agent):
        def act(self, obs):
            r = obs.returns(1)
            if r.empty or r.iloc[-1].isna().all():
                return None
            best = r.iloc[-1].idxmax()
            return {best: 1.0}

    cfg = SimulationConfig(broker=ZERO)
    res_next = run_backtest(mk, ChaseToday(), cfg)
    res_same = run_backtest(
        mk, ChaseToday(), cfg.model_copy(update={"broker": ZERO.model_copy(update={"execution": "same_close"})})
    )
    # Random walk: no edge either way, and certainly not the near-certain gain a real look-ahead would give.
    for res in (res_next, res_same):
        eq = res.equity_curve
        assert eq.iloc[-1] / eq.iloc[0] < 2.0
    # Same-close fills one session earlier than next-open.
    assert res_same.trades["date"].min() < res_next.trades["date"].min()


def test_same_close_fills_at_todays_close(market):
    cfg = SimulationConfig(broker=BrokerConfig(execution="same_close", costs=CostModel.zero(), max_participation=None))
    sim = Simulation(market, cfg)
    sim.submit({"BBB.L": 0.5})
    res = sim.result()
    j = market.col("BBB.L")
    assert res.trades.iloc[0]["date"] == market.dates[0]
    assert res.trades.iloc[0]["price"] == pytest.approx(market.array("close")[0, j])


def test_cash_agent_keeps_cash_and_earns_interest(market):
    cfg = SimulationConfig(initial_cash=1000, broker=BrokerConfig(cash_interest_rate=0.0252, costs=CostModel.zero()))
    res = run_backtest(market, CashAgent(), cfg)
    days = len(market.dates) - 1
    assert res.equity_curve.iloc[-1] == pytest.approx(1000 * (1 + 0.0252 / 252) ** days)
    assert res.trades.empty


def test_stamp_duty_main_market_only():
    mk = make_market(tickers=("MAIN1.L", "AIM1.L"), markets=["MAIN", "AIM"])
    cfg = SimulationConfig(
        initial_cash=100_000,
        broker=BrokerConfig(costs=CostModel(slippage_bps=0, stamp_duty_rate=0.005), max_participation=None),
    )
    sim = Simulation(mk, cfg)
    sim.submit({"MAIN1.L": 0.4, "AIM1.L": 0.4})
    sim.step()
    tr = sim.result().trades.set_index("ticker")
    assert tr.loc["MAIN1.L", "stamp_duty"] == pytest.approx(0.005 * tr.loc["MAIN1.L", "value"])
    assert tr.loc["AIM1.L", "stamp_duty"] == 0.0
    # Selling never pays stamp duty.
    sim.submit({})
    sim.step()
    sells = sim.result().trades
    assert (sells[sells["side"] == "SELL"]["stamp_duty"] == 0).all()


def test_costs_reduce_equity_exactly(market):
    costs = CostModel(commission_fixed=5.0, slippage_bps=10, stamp_duty_rate=0.005)
    cfg = SimulationConfig(initial_cash=10_000, broker=BrokerConfig(costs=costs, max_participation=None))
    sim = Simulation(market, cfg)
    sim.submit({"AAA.L": 0.5})
    sim.step()
    res = sim.result()
    f = res.trades.iloc[0]
    j = market.col("AAA.L")
    shares = f["shares"]
    # Cash = initial - shares*exec_price - commission - stamp.
    assert res.equity["cash"].iloc[-1] == pytest.approx(10_000 - shares * f["exec_price"] - 5.0 - f["stamp_duty"])
    assert f["exec_price"] == pytest.approx(market.array("open")[1, j] * 1.001)
    # Buy is sized on pre-cost equity but must still fit in cash.
    assert res.equity["cash"].iloc[-1] >= -1e-6


def test_buys_scaled_when_cash_insufficient(market):
    cfg = SimulationConfig(
        initial_cash=10_000,
        broker=BrokerConfig(costs=CostModel(commission_fixed=50, stamp_duty_rate=0.005), max_participation=None),
    )
    sim = Simulation(market, cfg)
    sim.submit({"AAA.L": 0.5, "BBB.L": 0.5})
    sim.step()
    res = sim.result()
    assert res.equity["cash"].iloc[-1] >= -1e-6
    assert (res.events["event"] == "buys_scaled").any()


def test_integer_shares(market):
    cfg = SimulationConfig(
        initial_cash=1_000, broker=BrokerConfig(fractional_shares=False, costs=CostModel.zero(), max_participation=None)
    )
    sim = Simulation(market, cfg)
    sim.submit({"AAA.L": 0.33, "BBB.L": 0.33, "CCC.L": 0.33})
    sim.step()
    shares = sim.result().trades["shares"]
    assert np.allclose(shares, np.round(shares))


def test_dividends_credited_to_holders():
    n = 30
    div = np.zeros((n, 3))
    div[10, 0] = 2.0  # AAA pays 2.0/share with ex-date on row 10
    mk = make_market(n_days=n, dividends=div)
    cfg = SimulationConfig(initial_cash=10_000, broker=ZERO)
    sim = Simulation(mk, cfg)
    sim.submit({"AAA.L": 1.0})
    while not sim.done:
        sim.step()
    res = sim.result()
    shares = res.trades.iloc[0]["shares"]
    assert res.equity["dividends"].sum() == pytest.approx(2.0 * shares)
    assert res.equity["dividends"].iloc[10] == pytest.approx(2.0 * shares)


def test_delisted_position_is_cashed_out():
    mk = make_market(n_days=40)
    close = mk.array("close").copy()
    close[20:, 1] = np.nan  # BBB stops trading after row 19
    arrays = {f: mk.array(f).copy() for f in ("open", "high", "low", "adj_close", "volume", "dividends")}
    for f in ("open", "high", "low", "adj_close"):
        arrays[f][20:, 1] = np.nan
    arrays["close"] = close
    from simtrade.sim.market import MarketData

    mk2 = MarketData(mk.dates, mk.tickers, arrays, mk.universe)
    res = run_backtest(mk2, BuyAndHold({"BBB.L": 1.0}), SimulationConfig(broker=ZERO, initial_cash=1000))
    ev = res.events[res.events["event"] == "delisted"]
    assert len(ev) == 1 and ev.iloc[0]["ticker"] == "BBB.L"
    assert ev.iloc[0]["date"] == mk.dates[20]
    # After delisting, all cash and equity frozen at the last close value.
    assert res.equity["n_positions"].iloc[-1] == 0
    expected = 1000 / mk.array("open")[1, 1] * close[19, 1]
    assert res.equity_curve.iloc[-1] == pytest.approx(expected)


def test_untradable_holding_is_kept(market):
    mk = market
    vol = mk.array("volume").copy()
    vol[5, 0] = 0.0  # AAA suspended on row 5
    arrays = {f: mk.array(f) for f in ("open", "high", "low", "close", "adj_close", "dividends")}
    arrays["volume"] = vol
    from simtrade.sim.market import MarketData

    mk2 = MarketData(mk.dates, mk.tickers, arrays, mk.universe)
    sim = Simulation(mk2, SimulationConfig(broker=ZERO))
    sim.submit({"AAA.L": 1.0})
    sim.step()
    for _ in range(3):
        sim.step()
    sim.submit({})  # try to sell on row 5 (suspended)
    sim.step()
    assert sim.portfolio.shares[0] > 0
    assert (sim.result().events["event"] == "untradable").any()


def test_participation_cap_limits_order_size():
    mk = make_market(volume=100.0)  # ~100 shares/day * ~100 price = ~10k traded value per day
    cfg = SimulationConfig(initial_cash=1_000_000, broker=BrokerConfig(costs=CostModel.zero(), max_participation=0.1))
    sim = Simulation(mk, cfg)
    sim.submit({"AAA.L": 1.0})
    sim.step()
    f = sim.result().trades.iloc[0]
    adv = mk.array("close")[0, 0] * 100.0
    assert f["shares"] * f["price"] <= 0.1 * adv * 1.0001


def test_long_only_clips_negative_weights(market):
    sim = Simulation(market, SimulationConfig(broker=ZERO))
    sim.submit({"AAA.L": -0.5, "BBB.L": 0.5})
    sim.step()
    assert sim.portfolio.shares[0] == 0
    assert sim.portfolio.shares[1] > 0


def test_short_selling_when_allowed(market):
    cfg = SimulationConfig(
        broker=BrokerConfig(
            allow_short=True,
            costs=CostModel(slippage_bps=0, stamp_duty_rate=0, borrow_rate=0.05),
            max_participation=None,
        )
    )
    sim = Simulation(market, cfg)
    sim.submit({"AAA.L": -0.5, "BBB.L": 0.5})
    sim.step()
    sim.step()
    res = sim.result()
    assert sim.portfolio.shares[0] < 0
    assert res.equity["borrow_cost"].iloc[-1] > 0
    assert "SHORT" in set(res.trades["side"])


def test_leverage_cap(market):
    sim = Simulation(market, SimulationConfig(broker=ZERO))
    sim.submit({"AAA.L": 1.0, "BBB.L": 1.0})  # gross 2.0 -> scaled to 1.0
    sim.step()
    assert sim.result().equity["gross_exposure"].iloc[-1] == pytest.approx(1.0, rel=0.02)


def test_monthly_rebalance_schedule():
    mk = make_market(n_days=120)
    calls = []

    class Count(Agent):
        def act(self, obs):
            calls.append(obs.date)
            return None

    run_backtest(mk, Count(), SimulationConfig(rebalance="monthly", broker=ZERO))
    # First day plus the last session of each complete month before the final day.
    months = pd.Series(mk.dates).dt.to_period("M")
    month_ends = pd.Series(mk.dates).groupby(months).max().tolist()
    expected = sorted(set([mk.dates[0]] + [d for d in month_ends if d < mk.dates[-1]]))
    assert calls == expected


def test_reset_gives_identical_runs(market):
    sim = Simulation(market, SimulationConfig())
    a = sim.run(EqualWeight(None)).equity_curve
    b = sim.run(EqualWeight(None)).equity_curve
    pd.testing.assert_series_equal(a, b)


def test_start_end_window():
    mk = make_market(n_days=100)
    cfg = SimulationConfig(start=mk.dates[30].date(), end=mk.dates[60].date(), broker=ZERO)
    res = run_backtest(mk, EqualWeight(None), cfg)
    assert res.equity.index[0] == mk.dates[30]
    assert res.equity.index[-1] == mk.dates[60]


def test_momentum_agent_waits_for_history():
    mk = make_market(n_days=80, tickers=tuple(f"T{i}.L" for i in range(6)))
    res = run_backtest(mk, MomentumAgent(lookback=50, skip=5, top_k=2), SimulationConfig(broker=ZERO))
    first_trade = res.trades["date"].min()
    assert first_trade > mk.dates[50]
    assert res.equity["n_positions"].max() == 2


def test_fixed_weights_and_benchmark(market):
    bench = pd.Series(np.linspace(100, 110, len(market.dates)), index=market.dates)
    res = Simulation(market, SimulationConfig(), benchmark=bench).run(FixedWeights({"AAA.L": 0.6, "CCC.L": 0.4}))
    m = res.metrics()
    assert "beta" in m and "benchmark_total_return" in m
    assert m["benchmark_total_return"] == pytest.approx(0.10)
