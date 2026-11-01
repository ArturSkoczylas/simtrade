"""Download and incrementally update the local price store."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..calendar import DateLike, last_completed_session, to_timestamp
from ..store import DataStore, fx_symbol
from .providers import FetchResult, PriceProvider
from .validate import normalize_currency, validate_prices

log = logging.getLogger(__name__)


@dataclass
class UpdateReport:
    new: List[str] = field(default_factory=list)
    updated: List[str] = field(default_factory=list)
    refreshed: List[str] = field(default_factory=list)  # full history re-downloaded (adjustments changed)
    unchanged: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"new: {len(self.new)}, updated: {len(self.updated)}, refreshed: {len(self.refreshed)}, "
            f"unchanged: {len(self.unchanged)}, failed: {len(self.failed)}"
        )

    def merge(self, other: "UpdateReport") -> None:
        self.new += other.new
        self.updated += other.updated
        self.refreshed += other.refreshed
        self.unchanged += other.unchanged
        self.failed.update(other.failed)


class Updater:
    """Fetches prices from a :class:`PriceProvider` into a :class:`DataStore`.

    * New tickers get their full history from ``default_start``.
    * Known tickers are fetched from ``last_date - overlap_days``. If the overlapping rows
      disagree (a split or dividend re-adjusted the history) the whole history is re-downloaded.
    * Requests are batched with a pause between batches and retried with backoff on errors.
    * The manifest is saved after every batch, so an interrupted update can simply be rerun.
    * Prices are validated, repaired and converted from pence to pounds before storing.
    """

    def __init__(
        self,
        store: DataStore,
        provider: PriceProvider,
        batch_size: Optional[int] = None,
        pause: float = 1.0,
        max_retries: int = 3,
        overlap_days: int = 10,
        default_start: DateLike = "2000-01-01",
        base_currency: str = "GBP",
        progress: Optional[Callable[[int, int, str], None]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.store = store
        self.provider = provider
        self.batch_size = batch_size or getattr(provider, "batch_size", 50)
        self.pause = pause
        self.max_retries = max_retries
        self.overlap_days = overlap_days
        self.default_start = to_timestamp(default_start)
        self.base_currency = base_currency
        self.progress = progress
        self._sleep = sleep

    # -- public ------------------------------------------------------------------
    def update(
        self,
        tickers: Iterable[str],
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        full: bool = False,
        kind: str = "equity",
        fx: bool = True,
    ) -> UpdateReport:
        """Bring ``tickers`` up to date (through ``end``, default: last completed LSE session)."""
        tickers = list(dict.fromkeys(tickers))
        end_ts = to_timestamp(end) if end is not None else last_completed_session()
        start_ts = to_timestamp(start) if start is not None else self.default_start
        report = UpdateReport()

        fresh, incremental = [], {}
        for t in tickers:
            last = self.store.last_date(t)
            if full or last is None or not self.store.has(t):
                fresh.append(t)
            elif last >= end_ts:
                report.unchanged.append(t)
            else:
                incremental[t] = last

        # New / full downloads share a start date and can be batched together.
        batches = [(b, start_ts) for b in _chunks(fresh, self.batch_size)]
        # Incremental downloads: group by fetch start.
        by_start: Dict[pd.Timestamp, List[str]] = {}
        for t, last in incremental.items():
            by_start.setdefault(last - pd.Timedelta(days=self.overlap_days), []).append(t)
        for s, group in sorted(by_start.items()):
            batches += [(b, s) for b in _chunks(group, self.batch_size)]

        done, total = 0, len(fresh) + len(incremental)
        to_refresh: List[str] = []
        with self.store.batch():
            for k, (batch, s) in enumerate(batches):
                results = self._fetch_with_retry(batch, s, end_ts)
                for t in batch:
                    res = results.get(t) or FetchResult(t, None, error="missing from response")
                    outcome = self._apply(t, res, incremental.get(t), kind, start_ts)
                    if outcome == "refresh":
                        to_refresh.append(t)
                    elif outcome == "new":
                        report.new.append(t)
                    elif outcome == "updated":
                        report.updated.append(t)
                    elif outcome == "unchanged":
                        report.unchanged.append(t)
                    else:
                        report.failed[t] = outcome
                done += len(batch)
                self.store.save_manifest()
                if self.progress:
                    self.progress(done, total, batch[-1])
                if self.pause and k < len(batches) - 1:
                    self._sleep(self.pause)

            for batch in _chunks(to_refresh, self.batch_size):
                results = self._fetch_with_retry(batch, start_ts, end_ts)
                for t in batch:
                    res = results.get(t) or FetchResult(t, None, error="missing from response")
                    outcome = self._apply(t, res, None, kind, start_ts, replace=True)
                    if outcome in ("new", "updated"):
                        report.refreshed.append(t)
                    else:
                        report.failed[t] = outcome
                self.store.save_manifest()

        if fx and kind != "fx":
            report.merge(self.update_fx(end=end_ts))
        return report

    def update_fx(self, end: Optional[DateLike] = None) -> UpdateReport:
        """Fetch FX rates for every non-base currency present in the store."""
        currencies = set()
        for t in self.store.tickers(kind="equity"):
            cur = self.store.meta(t).get("currency")
            if cur and cur.upper() != self.base_currency.upper():
                currencies.add(cur.upper())
        if not currencies:
            return UpdateReport()
        symbols = [fx_symbol(self.base_currency, c) for c in sorted(currencies)]
        return self.update(symbols, end=end, kind="fx", fx=False)

    # -- internals ---------------------------------------------------------------
    def _fetch_with_retry(self, batch: Sequence[str], start, end) -> Dict[str, FetchResult]:
        delay = max(self.pause, 1.0)
        for attempt in range(self.max_retries + 1):
            try:
                return self.provider.fetch(batch, start, end)
            except Exception as exc:
                if attempt == self.max_retries:
                    log.error("batch of %d failed after %d retries: %s", len(batch), attempt, exc)
                    return {t: FetchResult(t, None, error=f"{type(exc).__name__}: {exc}") for t in batch}
                log.warning("fetch failed (%s); retrying in %.0fs", exc, delay)
                self._sleep(delay)
                delay *= 2
        return {}

    def _apply(
        self, t: str, res: FetchResult, last: Optional[pd.Timestamp], kind: str, start_ts, replace: bool = False
    ) -> str:
        meta = self.store.meta(t)
        quote_ccy = res.currency or meta.get("quote_currency")
        if quote_ccy is None and kind != "fx":
            quote_ccy = self.provider.currency(t)
        if not res.ok:
            if last is not None:
                self.store.set_meta(t, last_checked=pd.Timestamp.now().isoformat(timespec="seconds"))
                return "unchanged"
            self.store.set_meta(
                t, status="error" if res.error and "no data" not in res.error else "empty", error=res.error, kind=kind
            )
            return res.error or "no data"
        df, vrep = validate_prices(res.data, None if kind == "fx" else quote_ccy)
        if df.empty:
            if last is not None:
                return "unchanged"
            self.store.set_meta(t, status="empty", error="empty after validation", kind=kind)
            return "empty after validation"
        major, _ = normalize_currency(quote_ccy)
        common = dict(
            kind=kind, provider=self.provider.name, quote_currency=quote_ccy, currency=None if kind == "fx" else major
        )

        if last is None or replace:
            self.store.write_prices(t, df, flags=vrep.flags, **common)
            return "new" if not replace else "updated"

        old = self.store.read_prices(t)
        overlap = old.index.intersection(df.index)
        if len(overlap):
            a = old.loc[overlap, ["close", "adj_close"]].to_numpy()
            b = df.loc[overlap, ["close", "adj_close"]].to_numpy()
            ok = np.isclose(a, b, rtol=2e-4, atol=1e-8) | (np.isnan(a) & np.isnan(b))
            if not ok.all():
                return "refresh"
        new_rows = df[df.index > old.index.max()]
        if (new_rows["splits"] > 0).any():
            return "refresh"
        if new_rows.empty:
            self.store.set_meta(t, last_checked=pd.Timestamp.now().isoformat(timespec="seconds"))
            return "unchanged"
        merged = pd.concat([old[old.index < df.index.min()], df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        flags = sorted(set(meta.get("flags") or []) | set(vrep.flags))
        self.store.write_prices(t, merged, flags=flags, **common)
        return "updated"


def _chunks(seq: Sequence[str], n: int) -> List[List[str]]:
    seq = list(seq)
    return [seq[i : i + n] for i in range(0, len(seq), max(1, n))]
