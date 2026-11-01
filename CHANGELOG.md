# Changelog

## 0.1.0

First release.

- Local Parquet data store with an incremental updater: batching, retries, resumable runs, and a full
  re-download when adjustments change.
- Yahoo Finance provider. Pence-to-pound conversion, 100x glitch repair, OHLC validation.
- LSE instrument-list parser (Main Market, AIM, shares, investment trusts, ETFs, ...).
- Day-by-day simulation engine with next-open or next-close execution, UK stamp duty, commission,
  slippage, liquidity caps, dividends, delistings, shorting and FX conversion.
- Agent interface, baseline agents, feature providers with a look-ahead check, walk-forward
  evaluation, metrics and charts.
- Gymnasium environment and a policy wrapper for RL.
- `simtrade` CLI: `universe`, `data`, `backtest`, `demo`.
