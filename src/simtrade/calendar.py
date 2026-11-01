"""Exchange trading calendar helpers (London Stock Exchange by default)."""

from __future__ import annotations

import logging
from datetime import date, datetime
from functools import lru_cache
from typing import Optional, Union

import pandas as pd

log = logging.getLogger(__name__)

DateLike = Union[str, date, datetime, pd.Timestamp]

DEFAULT_EXCHANGE = "XLON"
# exchange_calendars only builds sessions inside the calendar's bounds; ask for a wide range.
_CAL_START = pd.Timestamp("1990-01-01")


def to_timestamp(d: Optional[DateLike]) -> Optional[pd.Timestamp]:
    if d is None:
        return None
    ts = pd.Timestamp(d)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


@lru_cache(maxsize=8)
def _get_calendar(exchange: str):
    import exchange_calendars as xcals

    end = pd.Timestamp.today().normalize() + pd.DateOffset(years=1)
    return xcals.get_calendar(exchange, start=_CAL_START, end=end)


def trading_sessions(start: DateLike, end: DateLike, exchange: str = DEFAULT_EXCHANGE) -> pd.DatetimeIndex:
    """Trading sessions between ``start`` and ``end`` (inclusive), tz-naive.

    Falls back to Monday-Friday business days if the exchange calendar is unavailable
    or the range is outside its bounds.
    """
    s, e = to_timestamp(start), to_timestamp(end)
    if s > e:
        return pd.DatetimeIndex([], name="date")
    try:
        cal = _get_calendar(exchange)
        lo = max(s, cal.first_session.tz_localize(None) if cal.first_session.tzinfo else cal.first_session)
        hi = min(e, cal.last_session.tz_localize(None) if cal.last_session.tzinfo else cal.last_session)
        parts = []
        if s < lo:
            parts.append(pd.bdate_range(s, lo - pd.Timedelta(days=1)))
        if lo <= hi:
            sess = cal.sessions_in_range(lo, hi)
            sess = sess.tz_localize(None) if sess.tz is not None else sess
            parts.append(pd.DatetimeIndex(sess))
        if e > hi:
            parts.append(pd.bdate_range(hi + pd.Timedelta(days=1), e))
        out = parts[0].append(parts[1:]) if parts else pd.DatetimeIndex([])
    except Exception as exc:  # pragma: no cover - depends on optional package state
        log.warning("Exchange calendar %s unavailable (%s); using business days.", exchange, exc)
        out = pd.bdate_range(s, e)
    out = pd.DatetimeIndex(out).normalize().unique().sort_values()
    out.name = "date"
    return out


def last_completed_session(now: Optional[DateLike] = None, exchange: str = DEFAULT_EXCHANGE) -> pd.Timestamp:
    """The most recent session whose close has passed (LSE closes 16:30 London time)."""
    now_ts = pd.Timestamp.now(tz="Europe/London") if now is None else pd.Timestamp(now)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("Europe/London")
    else:
        now_ts = now_ts.tz_convert("Europe/London")
    today = now_ts.tz_localize(None).normalize()
    sessions = trading_sessions(today - pd.Timedelta(days=14), today, exchange)
    closed = today + pd.Timedelta(hours=16, minutes=35)
    if len(sessions) and sessions[-1] == today and now_ts.tz_localize(None) < closed:
        sessions = sessions[:-1]
    return sessions[-1]
