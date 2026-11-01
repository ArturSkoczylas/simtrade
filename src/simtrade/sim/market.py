"""In-memory, read-only panel of market data aligned to trading sessions."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..calendar import DateLike, to_timestamp, trading_sessions
from ..data.schema import standardize
from ..data.universe import Instrument, Universe

FIELDS = ("open", "high", "low", "close", "adj_close", "volume", "dividends")


class MarketData:
    """A dense (sessions x tickers) panel for each price field.

    Arrays are read-only so agents cannot corrupt the data they observe. Missing values are NaN
    (dividends are 0). Construct it from a :class:`~simtrade.store.DataStore` with
    ``store.load_market(...)`` or from DataFrames with :meth:`from_frames` /
    :meth:`from_ticker_frames`.
    """

    def __init__(
        self,
        dates: pd.DatetimeIndex,
        tickers: Sequence[str],
        arrays: Mapping[str, np.ndarray],
        universe: Optional[Universe] = None,
        currency: str = "GBP",
    ):
        self.dates = pd.DatetimeIndex(dates, name="date")
        self.tickers: List[str] = list(tickers)
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError("duplicate tickers")
        n, m = len(self.dates), len(self.tickers)
        self._arrays: Dict[str, np.ndarray] = {}
        for f in FIELDS:
            a = arrays.get(f)
            if a is None:
                fill = 0.0 if f == "dividends" else np.nan
                if f == "adj_close" and "close" in arrays:
                    a = np.array(arrays["close"], dtype="float64", copy=True)
                else:
                    a = np.full((n, m), fill)
            a = np.array(a, dtype="float64", copy=True)
            if a.shape != (n, m):
                raise ValueError(f"field {f!r} has shape {a.shape}, expected {(n, m)}")
            if f == "dividends":
                a = np.nan_to_num(a, nan=0.0)
            a.setflags(write=False)
            self._arrays[f] = a
        self.currency = currency
        #: Tickers requested but not loaded, with the reason (filled by DataStore.load_market).
        self.skipped: Dict[str, str] = {}
        self.universe = universe if universe is not None else Universe.from_tickers(self.tickers)
        self._col = {t: j for j, t in enumerate(self.tickers)}
        valid = np.isfinite(self._arrays["close"])
        has_any = valid.any(axis=0)
        first = np.where(has_any, valid.argmax(axis=0), -1)
        last = np.where(has_any, n - 1 - valid[::-1].argmax(axis=0), -1)
        self.first_valid = first
        self.last_valid = last

    # -- construction -------------------------------------------------------
    @classmethod
    def from_frames(cls, frames: Mapping[str, pd.DataFrame], universe: Optional[Universe] = None, **kw) -> "MarketData":
        """Build from wide DataFrames ``{field: DataFrame(dates x tickers)}`` (needs 'close')."""
        if "close" not in frames:
            raise ValueError("frames must include 'close'")
        base = frames["close"]
        dates, tickers = pd.DatetimeIndex(base.index), list(base.columns)
        arrays = {
            f: df.reindex(index=dates, columns=tickers).to_numpy(dtype="float64")
            for f, df in frames.items()
            if f in FIELDS
        }
        return cls(dates, tickers, arrays, universe=universe, **kw)

    @classmethod
    def from_ticker_frames(
        cls,
        data: Mapping[str, pd.DataFrame],
        sessions: Optional[pd.DatetimeIndex] = None,
        universe: Optional[Universe] = None,
        exchange: str = "XLON",
        **kw,
    ) -> "MarketData":
        """Build from per-ticker tables in the standard schema ``{ticker: DataFrame}``.

        If ``sessions`` is None, the exchange calendar between the first and last date found is used.
        """
        std = {t: standardize(df) for t, df in data.items()}
        std = {t: df for t, df in std.items() if not df.empty}
        if sessions is None:
            if not std:
                raise ValueError("no data")
            lo = min(df.index.min() for df in std.values())
            hi = max(df.index.max() for df in std.values())
            sessions = trading_sessions(lo, hi, exchange)
        tickers = list(data.keys())
        n, m = len(sessions), len(tickers)
        arrays = {f: np.full((n, m), 0.0 if f == "dividends" else np.nan) for f in FIELDS}
        for j, t in enumerate(tickers):
            df = std.get(t)
            if df is None:
                continue
            # Dividends falling on non-sessions move to the next session.
            div = df["dividends"]
            div = div[div != 0]
            df = df.reindex(sessions)
            for f in FIELDS:
                if f == "dividends":
                    continue
                arrays[f][:, j] = df[f].to_numpy()
            if len(div):
                pos = sessions.searchsorted(div.index, side="left")
                for p, v in zip(pos, div.to_numpy()):
                    if 0 <= p < n:
                        arrays["dividends"][p, j] += v
        return cls(sessions, tickers, arrays, universe=universe, **kw)

    # -- access ---------------------------------------------------------------
    def array(self, field: str) -> np.ndarray:
        return self._arrays[field]

    def frame(self, field: str) -> pd.DataFrame:
        return pd.DataFrame(self._arrays[field], index=self.dates, columns=self.tickers, copy=False)

    def col(self, ticker: str) -> int:
        return self._col[ticker]

    def instrument(self, ticker: str) -> Instrument:
        return self.universe.get(ticker) or Instrument(ticker=ticker)

    @property
    def shape(self):
        return len(self.dates), len(self.tickers)

    def index_of(self, d: DateLike, side: str = "left") -> int:
        """Position of the first session >= d (``side='left'``) or last session <= d (``'right'``)."""
        ts = to_timestamp(d)
        if side == "left":
            return int(self.dates.searchsorted(ts, side="left"))
        return int(self.dates.searchsorted(ts, side="right")) - 1

    # -- subsetting -------------------------------------------------------------
    def slice(self, start: Optional[DateLike] = None, end: Optional[DateLike] = None) -> "MarketData":
        i0 = 0 if start is None else self.index_of(start, "left")
        i1 = len(self.dates) - 1 if end is None else self.index_of(end, "right")
        sl = slice(i0, i1 + 1)
        return MarketData(
            self.dates[sl], self.tickers, {f: a[sl] for f, a in self._arrays.items()}, self.universe, self.currency
        )

    def select(self, tickers: Iterable[str]) -> "MarketData":
        tickers = [t for t in tickers if t in self._col]
        idx = [self._col[t] for t in tickers]
        return MarketData(
            self.dates,
            tickers,
            {f: a[:, idx] for f, a in self._arrays.items()},
            self.universe.filter(tickers=tickers) if len(self.universe) else None,
            self.currency,
        )

    def stamp_duty_mask(self) -> np.ndarray:
        return np.array([self.instrument(t).stamp_duty_eligible for t in self.tickers], dtype=bool)

    def __repr__(self) -> str:
        if len(self.dates):
            span = f"{self.dates[0].date()}..{self.dates[-1].date()}"
        else:
            span = "empty"
        return f"MarketData({len(self.tickers)} tickers, {len(self.dates)} sessions, {span}, {self.currency})"
