# simtrade

A daily trading simulator for **London Stock Exchange** equities, built so you can drop in your
own trading agent (a rule, a machine-learning model or a reinforcement-learning policy) and test it
on real historical data with realistic UK costs.

- **Online data, offline simulation.** One command downloads (or incrementally updates) daily
  prices for the whole LSE into a local store. Every backtest then runs offline and reproducibly on
  that history, up to the latest session you downloaded.
- **Every LSE equity.** Main Market and AIM shares and investment trusts, loaded from the
  exchange's official instrument list, plus any custom ticker list.
- **No look-ahead by construction.** Agents see data up to the close of day *t*. Orders fill on
  day *t+1* (open by default), and the observation objects cannot reach later rows.
- **UK-realistic execution.** Prices are converted from pence to pounds, and the simulator models:
  - 0.5% stamp duty on Main Market buys (AIM exempt), commission and slippage/spread
  - liquidity caps, fractional or whole shares, optional shorting with borrow fees
  - dividends credited on the ex-date, and delisted holdings cashed out at their last price
- **One agent interface, three kinds of agent.** Rule-based, supervised-learning and RL agents all
  implement `act(observation) -> target weights`. A Gymnasium environment shares the exact same
  engine, so RL training and evaluation agree.
- **Pluggable signals.** Add Reddit, news, sentiment or any other data as a `FeatureProvider`.
  A built-in check catches features that peek into the future.

> Not investment advice. Backtests are not predictions, and free data has known gaps (see
> [Data caveats](#data-caveats)).

## Install

```bash
pip install simtrade              # core (includes yfinance for downloads)
pip install "simtrade[all]"       # + gymnasium (RL), matplotlib (charts), scipy (yfinance price repair)
```

From source: `git clone … && pip install -e ".[dev]" && pytest`.

## Try it in 30 seconds (no internet needed)

```bash
simtrade demo --dir simtrade-demo
```

This builds a synthetic LSE-like store (prices in pence, dividends, a delisting, AIM and Main
Market names) and backtests three baseline agents. Results go to `simtrade-demo/results/`:
`equity.csv`, `trades.csv`, `events.csv`, `positions.csv`, `metrics.json` and `equity.png`.

## Use real London Stock Exchange data

**1. Get the universe.** Download the *Instrument list* spreadsheet from the
[LSE reports page](https://www.londonstockexchange.com/reports?tab=instruments), then:

```bash
simtrade universe load Instrument_list.xlsx      # parses every sheet, classifies instruments
simtrade universe show                           # counts by market and type
```

`simtrade universe load --download` tries to fetch the latest file automatically. The LSE site
changes often, so the manual route is the reliable one. You can also skip the instrument list and
work with your own tickers:

```bash
simtrade universe tickers VOD.L BP.L HSBA.L AZN.L ULVR.L
```

**2. Download prices (re-run any time to update).**

```bash
simtrade data update --limit 50        # try 50 tickers first
simtrade data update                   # everything in the `lse_all` preset (~2,000 names)
simtrade data status                   # coverage, stale/delisted names, repaired data
```

The first full download takes a while; Yahoo rate-limits large requests. After that, updates only
fetch recent days. If a split or new dividend changes the adjusted history, that ticker is
re-downloaded in full. An interrupted run can simply be restarted. Data lives in `~/.simtrade`; set
`SIMTRADE_HOME` or pass `--data-dir` to change that.

**3. Run a backtest.**

```bash
simtrade backtest examples/momentum.yaml
```

## Write an agent

```python
import simtrade as st
from simtrade.agents import liquid_tickers

class MyAgent(st.Agent):
    name = "MyAgent"

    def act(self, obs):                      # called after the close of each rebalance day
        names = liquid_tickers(obs, n=100)   # 100 most traded names today
        ret = obs.history("adj_close", 64)[names].pct_change(63, fill_method=None).iloc[-1]
        top = ret.nlargest(10).index
        return {t: 0.1 for t in top}         # target weights; unlisted holdings are sold

store = st.DataStore()
config = st.SimulationConfig(start="2015-01-01", rebalance="monthly", benchmark="ISF.L")
market = store.load_market(start=config.start, lookback=config.lookback)
result = st.Simulation(market, config, benchmark=store.read_prices("ISF.L")["adj_close"]).run(MyAgent())
print(result.summary())
result.save("results/my-agent")
```

`act` may return a `dict` or `pd.Series` (ticker → weight), a numpy array aligned with
`obs.tickers`, or `None` to keep the portfolio unchanged. Weights are fractions of equity; anything
not invested stays in cash.

**What the observation offers.** All of these stop at the decision day:

| Accessor | Returns |
|---|---|
| `obs.date`, `obs.tickers`, `obs.tradable` | decision day, all tickers, tickers that traded today |
| `obs.history(field, n)` / `obs.close` / `obs.volume` … | last *n* rows of open, high, low, close, adj_close, volume or dividends |
| `obs.returns(n)` | daily total returns |
| `obs.latest(field)` | today's values |
| `obs.feature(name, n)` / `obs.features_now()` | configured features |
| `obs.portfolio` | cash, equity, shares, weights |
| `obs.instrument(t)` | market, type, currency, country, stamp-duty eligibility |

The returned DataFrames are read-only views; call `.copy()` if you want to modify one.

Agents that learn can implement `fit(market, features)`. `simtrade.eval.walk_forward` calls it on
rolling training windows and tests each fitted agent on the following window:

```python
from simtrade.eval import walk_forward
wf = walk_forward(market, lambda: MyModelAgent(), train_years=3, test_years=1)
print(wf.fold_table()); print(wf.metrics())
```

## How a simulated day works

```
close of day t   agent.act(Observation up to t)  ->  target weights
day t+1          1. holdings whose data ended (delisted) are cashed out at their last price
                 2. dividends with ex-date t+1 are credited (shorts pay them)
                 3. orders fill at the open of t+1   (or its close: execution="next_close")
                 4. borrow fees / cash interest accrue
                 5. portfolio is marked at the close of t+1
```

**Broker rules, applied in order:**

1. Weights are cleaned: negatives are clipped unless shorting is allowed, and gross exposure is
   scaled down to `max_gross_leverage`.
2. Stocks with no price or zero volume that day can't be traded and keep their current holding.
3. Each order is capped at `max_participation` × the average daily traded value known at decision
   time.
4. Sells execute before buys. If the buys don't fit in the available cash, they are scaled down.

`execution="same_close"` is available but optimistic, because the agent has already seen the
price it trades at.

**Costs** (`CostModel`):

| Parameter | Default | Notes |
|---|---|---|
| `stamp_duty_rate` | 0.5% | Buys of eligible shares only: UK-incorporated, Main Market. AIM, ETFs and most foreign companies are exempt. Override per instrument with `Instrument(stamp_duty=…)`. |
| `slippage_bps` | 5 | |
| `half_spread_bps` | 0 | |
| `commission_bps` / `commission_fixed` / `commission_min` | 0 | |
| `ptm_levy` | off | |
| `borrow_rate` | 0 | Annual rate on short market value. |

`CostModel.zero()` turns off all costs, which is useful for debugging.

## Features and alternative data

```python
from simtrade.features import FeatureProvider, check_causality

class RedditMentions(FeatureProvider):
    name = "reddit"
    def compute(self, market):
        wide = load_my_daily_mentions()            # DataFrame: dates x tickers
        return {"mentions": wide.reindex(market.dates)}

print(check_causality(RedditMentions(), market))   # [] = no look-ahead detected
sim = st.Simulation(market, config, features=[RedditMentions(), st.features.Returns((1, 5, 21))])
```

Built-in providers are `Returns`, `Volatility`, `Momentum` and `Liquidity`. A complete example is
`examples/alternative_data_feature.py`.

## Reinforcement learning

```python
from simtrade.envs import TradingEnv, PolicyAgent
env = TradingEnv(market.select(tickers), train_config, features=[...], window=20,
                 turnover_penalty=0.001, episode_length=252)
model = stable_baselines3.PPO("MlpPolicy", env).learn(100_000)
oos = st.run_backtest(market.select(tickers), PolicyAgent(model, env), test_config)
```

- **Observation:** recent log returns, features and current weights.
- **Action:** softmax portfolio weights, long-only.
- **Reward:** log equity growth minus a turnover penalty.

The environment passes Gymnasium's `check_env`. A full script is in `examples/train_ppo.py`.

## Experiments as YAML

```yaml
name: lse-momentum
universe: {preset: lse_all}            # lse_all | main | aim | lse_shares_only | everything
agent:
  class: simtrade.agents.MomentumAgent # or my_agent.py:MyAgent
  params: {lookback: 252, skip: 21, top_k: 30}
features:
  - {class: simtrade.features.Returns, params: {windows: [1, 5, 21]}}
simulation:
  start: 2012-01-01
  rebalance: monthly                   # daily | weekly | monthly | quarterly | <N sessions>
  benchmark: ISF.L
  broker: {max_participation: 0.05, costs: {commission_fixed: 5}}
```

## Data caveats

- **Survivorship bias.** Free sources only list companies that trade today. Backtests over the
  current LSE universe miss every company that went bust or was taken over, so results look
  better than reality. To plug in a provider with delisted history, implement `PriceProvider`.
- **yfinance is unofficial.** It can throttle, change or return glitches. simtrade validates
  everything it stores: it converts pence to pounds, repairs 100× glitches, fixes broken OHLC rows
  and flags spikes. Flags are listed per ticker by `simtrade data status`.
- **Corporate actions.** Prices are split-adjusted and cash dividends are credited on the ex-date
  (not the pay date). Special situations such as spin-offs and rights issues are only approximated
  by the adjusted series.
- **Currencies.** Instruments quoted in USD or EUR are converted to GBP using daily FX rates that
  are downloaded automatically.

## Project layout

```
src/simtrade/
  data/        universe (LSE list parser), providers (yfinance, CSV, synthetic), validation, updater
  store.py     Parquet-per-ticker store + manifest
  calendar.py  LSE trading calendar (exchange_calendars XLON)
  sim/         market panel, observation, broker, portfolio, engine, results
  agents/      Agent base class and baselines
  features/    FeatureProvider, built-in features, causality check
  eval/        metrics, walk-forward, plots
  envs/        Gymnasium environment
  cli.py       `simtrade` command
```

## Contributing

Issues and pull requests are welcome. Please add tests (`pytest`) for behaviour changes,
especially anything that touches timing or accounting.

## License

MIT
