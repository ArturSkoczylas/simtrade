"""Local on-disk data store: one Parquet file per ticker plus a JSON manifest.

Layout::

    <root>/
      manifest.json         per-ticker metadata (dates, currency, status, quality flags)
      universe.parquet      the saved investable universe (optional)
      prices/<ticker>.parquet

The store is written by :class:`~simtrade.data.updater.Updater` (online) and read by the
simulator (offline).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
from urllib.parse import quote

import numpy as np
import pandas as pd

from .calendar import DateLike, to_timestamp, trading_sessions
from .data.schema import standardize
from .data.universe import Universe

log = logging.getLogger(__name__)

DEFAULT_HOME_ENV = "SIMTRADE_HOME"


def default_root() -> Path:
    return Path(os.environ.get(DEFAULT_HOME_ENV, Path.home() / ".simtrade")).expanduser()


def fx_symbol(base: str, quote_ccy: str) -> str:
    """Yahoo FX symbol giving units of ``quote_ccy`` per 1 ``base`` (e.g. GBPUSD=X)."""
    return f"{base.upper()}{quote_ccy.upper()}=X"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DataStore:
    """Read/write access to a local price store."""

    def __init__(self, root: Optional[Union[str, Path]] = None):
        self.root = Path(root).expanduser() if root is not None else default_root()
        self.prices_dir = self.root / "prices"
        self.prices_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.json"
        self._manifest: Dict[str, dict] = self._load_manifest()
        self._batch_depth = 0
        self._dirty = False

    # -- manifest ---------------------------------------------------------------
    def _load_manifest(self) -> Dict[str, dict]:
        if self._manifest_path.exists():
            with open(self._manifest_path, encoding="utf-8") as fh:
                return json.load(fh).get("tickers", {})
        return {}

    def save_manifest(self) -> None:
        payload = {"version": 1, "updated_at": _now(), "tickers": self._manifest}
        _atomic_write_bytes(self._manifest_path, json.dumps(payload, indent=1, sort_keys=True).encode("utf-8"))
        self._dirty = False

    @contextmanager
    def batch(self):
        """Defer manifest writes until the block exits (much faster for bulk updates)."""
        self._batch_depth += 1
        try:
            yield self
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0 and self._dirty:
                self.save_manifest()

    def _touch(self) -> None:
        self._dirty = True
        if self._batch_depth == 0:
            self.save_manifest()

    def meta(self, ticker: str) -> dict:
        return dict(self._manifest.get(ticker, {}))

    def set_meta(self, ticker: str, **values) -> None:
        entry = self._manifest.setdefault(ticker, {})
        entry.update(values)
        self._touch()

    def tickers(self, kind: Optional[str] = None, status: Optional[str] = "ok") -> List[str]:
        """Tickers with stored prices (optionally filtered by ``kind`` and ``status``)."""
        out = []
        for t, m in self._manifest.items():
            if status is not None and m.get("status") != status:
                continue
            if kind is not None and m.get("kind", "equity") != kind:
                continue
            if self._path(t).exists():
                out.append(t)
        return sorted(out)

    def status_frame(self) -> pd.DataFrame:
        """One row per known ticker with dates, rows, status and quality flags."""
        if not self._manifest:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "kind",
                    "status",
                    "currency",
                    "first_date",
                    "last_date",
                    "rows",
                    "updated_at",
                    "flags",
                    "error",
                ]
            )
        df = pd.DataFrame.from_dict(self._manifest, orient="index")
        df.index.name = "ticker"
        df = df.reset_index()
        for c in ["kind", "status", "currency", "first_date", "last_date", "rows", "updated_at", "flags", "error"]:
            if c not in df.columns:
                df[c] = None
        return df

    # -- prices -----------------------------------------------------------------
    def _path(self, ticker: str) -> Path:
        return self.prices_dir / f"{quote(ticker, safe='')}.parquet"

    def has(self, ticker: str) -> bool:
        return self._path(ticker).exists()

    def write_prices(self, ticker: str, df: pd.DataFrame, **meta) -> None:
        """Replace the stored history of ``ticker`` with ``df`` (standard schema)."""
        df = standardize(df)
        path = self._path(ticker)
        fd, tmp = tempfile.mkstemp(dir=self.prices_dir, suffix=".tmp")
        os.close(fd)
        try:
            df.to_parquet(tmp, engine="pyarrow")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        entry = self._manifest.setdefault(ticker, {})
        entry.update(
            {
                "rows": int(len(df)),
                "first_date": str(df.index.min().date()) if len(df) else None,
                "last_date": str(df.index.max().date()) if len(df) else None,
                "updated_at": _now(),
                "status": "ok" if len(df) else "empty",
                "error": None,
            }
        )
        entry.setdefault("kind", "equity")
        entry.update(meta)
        self._touch()

    def read_prices(
        self, ticker: str, start: Optional[DateLike] = None, end: Optional[DateLike] = None
    ) -> pd.DataFrame:
        path = self._path(ticker)
        if not path.exists():
            raise KeyError(f"no stored prices for {ticker!r}")
        df = pd.read_parquet(path, engine="pyarrow")
        df.index = pd.DatetimeIndex(df.index, name="date")
        s, e = to_timestamp(start), to_timestamp(end)
        if s is not None:
            df = df[df.index >= s]
        if e is not None:
            df = df[df.index <= e]
        return df

    def last_date(self, ticker: str) -> Optional[pd.Timestamp]:
        d = self._manifest.get(ticker, {}).get("last_date")
        return pd.Timestamp(d) if d else None

    def delete(self, ticker: str) -> None:
        p = self._path(ticker)
        if p.exists():
            p.unlink()
        self._manifest.pop(ticker, None)
        self._touch()

    # -- universe ---------------------------------------------------------------
    def save_universe(self, universe: Universe) -> Path:
        path = self.root / "universe.parquet"
        df = universe.to_frame().astype({"stamp_duty": "object"})
        df.to_parquet(path, engine="pyarrow", index=False)
        return path

    def load_universe(self) -> Universe:
        path = self.root / "universe.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"No universe saved in {self.root}. Run `simtrade universe load <instrument_list.xlsx>` first."
            )
        return Universe.from_frame(pd.read_parquet(path, engine="pyarrow"))

    def has_universe(self) -> bool:
        return (self.root / "universe.parquet").exists()

    # -- loading a simulation panel -------------------------------------------------
    def load_market(
        self,
        tickers: Optional[Iterable[str]] = None,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        lookback: int = 0,
        base_currency: str = "GBP",
        use_adjusted_prices: bool = False,
        universe: Optional[Universe] = None,
        exchange: str = "XLON",
        min_rows: int = 1,
    ):
        """Load stored prices into a :class:`~simtrade.sim.market.MarketData` panel.

        ``lookback`` extra sessions before ``start`` are included so agents and features have
        history on the first trading day. Prices in other currencies are converted to
        ``base_currency`` with stored FX rates (tickers without FX data are skipped with a warning).
        """
        from .sim.market import MarketData

        if tickers is None:
            if universe is not None:
                tickers = universe.tickers
            else:
                tickers = self.tickers(kind="equity")
        tickers = [t for t in dict.fromkeys(tickers)]
        if universe is None and self.has_universe():
            universe = self.load_universe()

        s, e = to_timestamp(start), to_timestamp(end)
        load_from = None
        if s is not None:
            # Calendar days generously covering `lookback` sessions.
            load_from = s - pd.Timedelta(days=int(lookback * 1.6) + 10)

        fx_cache: Dict[str, Optional[pd.Series]] = {}
        frames: Dict[str, pd.DataFrame] = {}
        skipped: Dict[str, str] = {}
        for t in tickers:
            if not self.has(t):
                skipped[t] = "not in store"
                continue
            df = self.read_prices(t, load_from, e)
            if len(df) < min_rows:
                skipped[t] = "no rows in range"
                continue
            cur = (self._manifest.get(t, {}) or {}).get("currency")
            if cur and cur.upper() != base_currency.upper():
                if cur not in fx_cache:
                    fx_cache[cur] = self._load_fx(base_currency, cur)
                fx = fx_cache[cur]
                if fx is None:
                    skipped[t] = f"no FX {fx_symbol(base_currency, cur)}"
                    continue
                rate = fx.reindex(df.index).ffill().bfill()
                for c in ("open", "high", "low", "close", "adj_close", "dividends"):
                    df[c] = df[c] / rate
            if use_adjusted_prices:
                factor = (df["adj_close"] / df["close"]).where(lambda x: np.isfinite(x) & (x > 0), 1.0)
                for c in ("open", "high", "low", "close"):
                    df[c] = df[c] * factor
                df["dividends"] = 0.0
            frames[t] = df
        if skipped:
            n_fx = sum(1 for v in skipped.values() if v.startswith("no FX"))
            log.warning("Skipped %d tickers (%d for missing FX); see MarketData.skipped", len(skipped), n_fx)
        if not frames:
            raise ValueError("No price data found for the requested tickers/date range.")

        lo = min(df.index.min() for df in frames.values())
        hi = max(df.index.max() for df in frames.values())
        if e is not None:
            hi = min(hi, e)
        sessions = trading_sessions(lo, hi, exchange)
        if s is not None and lookback >= 0:
            i0 = max(0, int(sessions.searchsorted(s, side="left")) - lookback)
            sessions = sessions[i0:]
        market = MarketData.from_ticker_frames(frames, sessions=sessions, universe=universe, currency=base_currency)
        market.skipped = skipped
        return market

    def _load_fx(self, base: str, quote_ccy: str) -> Optional[pd.Series]:
        sym = fx_symbol(base, quote_ccy)
        if not self.has(sym):
            return None
        s = self.read_prices(sym)["close"].dropna()
        return s if len(s) else None

    def __repr__(self) -> str:
        return f"DataStore({str(self.root)!r}, {len(self._manifest)} tickers)"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
