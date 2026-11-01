"""Build a small synthetic data store for demos, docs and tests (no network needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from ..store import DataStore
from .providers import SyntheticProvider
from .universe import Instrument, Universe
from .updater import Updater


def make_sample_store(
    root: Union[str, Path],
    n_tickers: int = 40,
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    seed: int = 7,
) -> DataStore:
    """Create a store with ``n_tickers`` synthetic LSE-like stocks plus an equal-weighted index ``SAMPLE_IDX.L``.

    Includes Main Market and AIM names (for stamp duty), prices in pence, dividends,
    one delisting and one late listing, so every part of the pipeline is exercised.
    """
    tickers = [f"S{i:03d}.L" for i in range(n_tickers)]
    delist = {tickers[1]: "2019-06-28"} if n_tickers > 1 else {}
    list_dates = {tickers[2]: "2018-03-01"} if n_tickers > 2 else {}
    provider = SyntheticProvider(base_start=start, base_end=end, delist=delist, list_dates=list_dates, seed=seed)
    store = DataStore(root)
    uni = Universe(
        Instrument(
            ticker=t,
            tidm=t[:-2],
            name=f"Sample Company {i}",
            market="AIM" if i % 4 == 3 else "MAIN",
            instrument_type="share",
            currency="GBp",
            country="United Kingdom",
            sector=["Industrials", "Financials", "Technology", "Energy", "Consumer"][i % 5],
        )
        for i, t in enumerate(tickers)
    )
    store.save_universe(uni)
    up = Updater(store, provider, pause=0, default_start=start)
    up.update(tickers, end=end)
    _write_index(store, tickers, "SAMPLE_IDX.L")
    return store


def _write_index(store: DataStore, tickers, name: str) -> None:
    """Equal-weighted total-return index of ``tickers`` (daily rebalanced), stored as a benchmark."""
    rets = pd.concat({t: store.read_prices(t)["adj_close"] for t in tickers if store.has(t)}, axis=1, sort=True)
    daily = rets.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    level = 100.0 * (1.0 + daily).cumprod()
    df = pd.DataFrame({"open": level, "high": level, "low": level, "close": level, "adj_close": level})
    store.write_prices(name, df, kind="index", currency="GBP", provider="synthetic")
