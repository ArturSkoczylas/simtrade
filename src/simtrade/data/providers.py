"""Price providers: where daily OHLCV data comes from.

Implement :class:`PriceProvider` to plug in another source (a paid API, a database, ...).
"""

from __future__ import annotations

import logging
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..calendar import DateLike, to_timestamp, trading_sessions
from .schema import standardize

log = logging.getLogger(__name__)


@dataclass
class FetchResult:
    ticker: str
    data: Optional[pd.DataFrame]
    currency: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.data is not None and not self.data.empty and self.error is None


class PriceProvider(ABC):
    """Fetches daily price history in the standard schema (see :mod:`simtrade.data.schema`)."""

    name: str = "provider"
    #: How many tickers to request per call.
    batch_size: int = 50

    @abstractmethod
    def fetch(
        self, tickers: Sequence[str], start: Optional[DateLike] = None, end: Optional[DateLike] = None
    ) -> Dict[str, FetchResult]:
        """Return one :class:`FetchResult` per requested ticker (``start``/``end`` inclusive)."""

    def currency(self, ticker: str) -> Optional[str]:
        """Quote currency of ``ticker`` as reported by the source (e.g. 'GBp'), if known."""
        return None


class YFinanceProvider(PriceProvider):
    """Yahoo Finance via the ``yfinance`` package (free, unofficial, rate limited).

    Uses ``auto_adjust=False`` so ``close`` is split-adjusted and ``adj_close`` also includes
    dividends, and ``actions=True`` for dividends and splits. ``repair=True`` asks yfinance to
    fix 100x currency glitches (needs scipy; silently disabled if missing).
    """

    name = "yfinance"

    def __init__(
        self,
        batch_size: int = 50,
        repair: bool = True,
        timeout: float = 30.0,
        threads: bool = True,
        lookup_currency: bool = True,
        default_currency_for_l: str = "GBp",
    ):
        self.batch_size = batch_size
        self.repair = repair
        self.timeout = timeout
        self.threads = threads
        self.lookup_currency = lookup_currency
        self.default_currency_for_l = default_currency_for_l
        if repair:
            try:
                import scipy  # noqa: F401
            except ImportError:
                log.info("scipy not installed: yfinance price repair disabled")
                self.repair = False

    def fetch(self, tickers, start=None, end=None):
        import yfinance as yf

        tickers = list(dict.fromkeys(tickers))
        s = to_timestamp(start)
        e = to_timestamp(end)
        kwargs = dict(
            tickers=tickers,
            start=None if s is None else s.strftime("%Y-%m-%d"),
            end=None if e is None else (e + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),  # yfinance end is exclusive
            actions=True,
            auto_adjust=False,
            repair=self.repair,
            group_by="ticker",
            progress=False,
            threads=self.threads,
            timeout=self.timeout,
            multi_level_index=True,
        )
        if s is None:
            kwargs["period"] = "max"
            kwargs.pop("start")
        raw = yf.download(**kwargs)
        out: Dict[str, FetchResult] = {}
        for t in tickers:
            df = _extract_ticker(raw, t, single=len(tickers) == 1)
            if df is None or df.empty:
                out[t] = FetchResult(t, None, error="no data returned")
                continue
            try:
                std = standardize(df)
            except Exception as exc:  # malformed frame
                out[t] = FetchResult(t, None, error=f"bad data: {exc}")
                continue
            if std.empty:
                out[t] = FetchResult(t, None, error="no data returned")
            else:
                out[t] = FetchResult(t, std)
        return out

    def currency(self, ticker: str) -> Optional[str]:
        if not self.lookup_currency:
            return self.default_currency_for_l if ticker.endswith(".L") else None
        import yfinance as yf

        try:
            cur = yf.Ticker(ticker).fast_info.get("currency")
            if cur:
                return str(cur)
        except Exception as exc:  # pragma: no cover - network
            log.debug("currency lookup failed for %s: %s", ticker, exc)
        if ticker.endswith("=X"):
            return None
        return self.default_currency_for_l if ticker.endswith(".L") else None


def _extract_ticker(raw: Optional[pd.DataFrame], ticker: str, single: bool) -> Optional[pd.DataFrame]:
    if raw is None or len(raw) == 0:
        return None
    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        lvl0 = cols.get_level_values(0)
        lvl1 = cols.get_level_values(1)
        if ticker in lvl0:
            df = raw[ticker]
        elif ticker in lvl1:
            df = raw.xs(ticker, axis=1, level=1)
        else:
            return None
    else:
        if not single:
            return None
        df = raw
    df = df.dropna(how="all")
    price_cols = [c for c in df.columns if str(c).lower() in ("open", "high", "low", "close", "adj close")]
    if price_cols:
        df = df.dropna(how="all", subset=price_cols)
    return df


class CSVProvider(PriceProvider):
    """Reads one CSV per ticker from a directory (``<dir>/<ticker>.csv``).

    Each CSV needs a date column and at least ``close``; other schema columns are optional.
    """

    name = "csv"

    def __init__(self, directory: Union[str, Path], currency: Optional[str] = "GBP", pattern: str = "{ticker}.csv"):
        self.directory = Path(directory)
        self._currency = currency
        self.pattern = pattern
        self.batch_size = 500

    def fetch(self, tickers, start=None, end=None):
        s, e = to_timestamp(start), to_timestamp(end)
        out = {}
        for t in tickers:
            path = self.directory / self.pattern.format(ticker=t)
            if not path.exists():
                out[t] = FetchResult(t, None, error=f"missing file {path.name}")
                continue
            df = pd.read_csv(path)
            df = standardize(df)
            if s is not None:
                df = df[df.index >= s]
            if e is not None:
                df = df[df.index <= e]
            out[t] = FetchResult(t, df, currency=self._currency) if len(df) else FetchResult(t, None, error="no rows")
        return out

    def currency(self, ticker):
        return self._currency


class SyntheticProvider(PriceProvider):
    """Deterministic fake market data for demos and tests (no network needed).

    Each ticker follows a geometric random walk seeded by its name, so repeated or incremental
    fetches return identical history. Optional features exercise the data pipeline:

    * prices quoted in pence (``GBp``) like real LSE data,
    * semi-annual dividends,
    * ``delist`` - tickers whose data stops at a given date,
    * ``list_dates`` - tickers that start trading later,
    * ``suspensions`` - random missing days.
    """

    name = "synthetic"

    def __init__(
        self,
        base_start: DateLike = "2010-01-01",
        base_end: DateLike = "2024-12-31",
        currency: str = "GBp",
        dividends: bool = True,
        delist: Optional[Mapping[str, DateLike]] = None,
        list_dates: Optional[Mapping[str, DateLike]] = None,
        missing_prob: float = 0.002,
        seed: int = 7,
        currencies: Optional[Mapping[str, str]] = None,
    ):
        self.base_start = to_timestamp(base_start)
        self.base_end = to_timestamp(base_end)
        self._currency = currency
        self.dividends = dividends
        self.delist = {k: to_timestamp(v) for k, v in (delist or {}).items()}
        self.list_dates = {k: to_timestamp(v) for k, v in (list_dates or {}).items()}
        self.missing_prob = missing_prob
        self.seed = seed
        self.currencies = dict(currencies or {})
        self.batch_size = 1000
        self._sessions = trading_sessions(self.base_start, self.base_end)
        self.calls = 0

    def _series(self, ticker: str) -> pd.DataFrame:
        if ticker.endswith("=X"):  # FX: gentle random walk around 1.3
            rng = np.random.default_rng(zlib.crc32(ticker.encode()) + self.seed)
            n = len(self._sessions)
            px = 1.3 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
            return pd.DataFrame(
                {
                    "open": px,
                    "high": px,
                    "low": px,
                    "close": px,
                    "adj_close": px,
                    "volume": 0.0,
                    "dividends": 0.0,
                    "splits": 0.0,
                },
                index=self._sessions,
            )
        rng = np.random.default_rng(zlib.crc32(ticker.encode()) + self.seed)
        n = len(self._sessions)
        mu = rng.normal(0.06, 0.08) / 252
        sigma = rng.uniform(0.15, 0.45) / np.sqrt(252)
        r = rng.normal(mu, sigma, n)
        close = rng.uniform(50, 2000) * np.exp(np.cumsum(r))  # pence
        gap = rng.normal(0, sigma / 3, n)
        open_ = close * np.exp(-r + gap)  # open between previous close and close
        open_[0] = close[0]
        hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, sigma / 2, n)))
        lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, sigma / 2, n)))
        volume = np.round(rng.lognormal(np.log(rng.uniform(2e4, 5e6)), 0.5, n))
        df = pd.DataFrame(
            {"open": open_, "high": hi, "low": lo, "close": close, "volume": volume, "dividends": 0.0, "splits": 0.0},
            index=self._sessions,
        )
        if self.dividends:
            months = df.index.month
            first_in_month = np.r_[True, months[1:] != months[:-1]]
            ex = first_in_month & np.isin(months, (5, 11))
            df.loc[ex, "dividends"] = close[ex] * rng.uniform(0.005, 0.025)
        # adj_close: back-adjust for dividends (Yahoo-style multiplicative factor).
        factor = np.ones(n)
        divs = df["dividends"].to_numpy()
        for k in np.flatnonzero(divs > 0):
            if k > 0:
                factor[:k] *= 1 - divs[k] / close[k - 1]
        df["adj_close"] = close * factor
        if self.missing_prob > 0:
            drop = rng.random(n) < self.missing_prob
            drop[0] = False
            df = df[~drop]
        if ticker in self.list_dates:
            df = df[df.index >= self.list_dates[ticker]]
        if ticker in self.delist:
            df = df[df.index <= self.delist[ticker]]
        return df

    def fetch(self, tickers, start=None, end=None):
        self.calls += 1
        s, e = to_timestamp(start), to_timestamp(end)
        out = {}
        for t in tickers:
            df = self._series(t)
            if s is not None:
                df = df[df.index >= s]
            if e is not None:
                df = df[df.index <= e]
            if df.empty:
                out[t] = FetchResult(t, None, error="no data returned")
            else:
                out[t] = FetchResult(t, standardize(df), currency=self.currency(t))
        return out

    def currency(self, ticker):
        if ticker.endswith("=X"):
            return None
        return self.currencies.get(ticker, self._currency)
