"""Investable universes: lists of instruments with the metadata the simulator needs.

The main entry point for London is :func:`load_lse_instrument_list`, which parses the
"Instrument list" spreadsheet published by the London Stock Exchange. Because the exact
layout of that file changes from time to time, the parser locates the header row and maps
columns by name rather than position.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Union

import pandas as pd

log = logging.getLogger(__name__)

# Instrument types used by simtrade.
SHARE = "share"
INVESTMENT_TRUST = "investment_trust"
ETF = "etf"
ETP = "etp"
DEPOSITARY_RECEIPT = "depositary_receipt"
OTHER = "other"

EQUITY_TYPES = (SHARE, INVESTMENT_TRUST)

UK_COUNTRIES = {"united kingdom", "uk", "gb", "gbr", "great britain", "england", "england and wales", "scotland"}


@dataclass(frozen=True)
class Instrument:
    """One tradable instrument.

    ``ticker`` is the symbol used by the price provider (for Yahoo, LSE tickers end in ``.L``).
    """

    ticker: str
    tidm: Optional[str] = None
    name: Optional[str] = None
    isin: Optional[str] = None
    market: Optional[str] = None  # "MAIN", "AIM", ...
    instrument_type: str = SHARE
    currency: Optional[str] = None
    country: Optional[str] = None
    sector: Optional[str] = None
    stamp_duty: Optional[bool] = None  # None = derive from market/type/country

    @property
    def stamp_duty_eligible(self) -> bool:
        """Approximate UK stamp duty / SDRT eligibility on purchases.

        Applies to shares (incl. investment trusts) of UK-incorporated companies on the Main
        Market. AIM shares have been exempt since April 2014; ETFs and most foreign-incorporated
        companies are exempt. Unknown country on the Main Market is treated as eligible
        (the conservative choice). Override per instrument with ``stamp_duty``.
        """
        if self.stamp_duty is not None:
            return bool(self.stamp_duty)
        if self.instrument_type not in EQUITY_TYPES:
            return False
        if (self.market or "").upper() != "MAIN":
            return False
        if self.country is None or str(self.country).strip() == "":
            return True
        return str(self.country).strip().lower() in UK_COUNTRIES


class Universe:
    """An ordered collection of :class:`Instrument` objects keyed by ticker."""

    def __init__(self, instruments: Iterable[Instrument] = ()):
        self._by_ticker: dict[str, Instrument] = {}
        for inst in instruments:
            self._by_ticker.setdefault(inst.ticker, inst)

    # -- container protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self._by_ticker)

    def __iter__(self) -> Iterator[Instrument]:
        return iter(self._by_ticker.values())

    def __contains__(self, ticker: object) -> bool:
        return ticker in self._by_ticker

    def __repr__(self) -> str:
        return f"Universe({len(self)} instruments)"

    def get(self, ticker: str) -> Optional[Instrument]:
        return self._by_ticker.get(ticker)

    @property
    def tickers(self) -> List[str]:
        return list(self._by_ticker)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_tickers(cls, tickers: Sequence[str], **defaults) -> "Universe":
        return cls(Instrument(ticker=t, **defaults) for t in tickers)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Universe":
        names = {f.name for f in fields(Instrument)}
        insts = []
        for rec in df.to_dict("records"):
            kw = {k: (None if _isna(v) else v) for k, v in rec.items() if k in names}
            if kw.get("instrument_type") is None:
                kw["instrument_type"] = SHARE
            if kw.get("stamp_duty") is not None:
                kw["stamp_duty"] = bool(kw["stamp_duty"])
            insts.append(Instrument(**kw))
        return cls(insts)

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> "Universe":
        """Load a universe from a CSV with at least a ``ticker`` column."""
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "ticker" not in df.columns:
            raise ValueError(f"{path}: CSV needs a 'ticker' column")
        return cls.from_frame(df)

    def to_frame(self) -> pd.DataFrame:
        cols = [f.name for f in fields(Instrument)]
        return pd.DataFrame([asdict(i) for i in self], columns=cols)

    # -- filtering ----------------------------------------------------------
    def filter(
        self,
        markets: Optional[Iterable[str]] = None,
        types: Optional[Iterable[str]] = None,
        currencies: Optional[Iterable[str]] = None,
        countries: Optional[Iterable[str]] = None,
        tickers: Optional[Iterable[str]] = None,
    ) -> "Universe":
        """Return a new universe keeping instruments that match every given criterion."""
        mk = {m.upper() for m in markets} if markets else None
        ty = set(types) if types else None
        cu = {c.upper() for c in currencies} if currencies else None
        co = {c.lower() for c in countries} if countries else None
        tk = set(tickers) if tickers else None
        out = []
        for i in self:
            if mk is not None and (i.market or "").upper() not in mk:
                continue
            if ty is not None and i.instrument_type not in ty:
                continue
            if cu is not None and (i.currency or "").upper() not in cu:
                continue
            if co is not None and (i.country or "").lower() not in co:
                continue
            if tk is not None and i.ticker not in tk:
                continue
            out.append(i)
        return Universe(out)

    def summary(self) -> pd.DataFrame:
        df = self.to_frame()
        if df.empty:
            return df
        return df.groupby(["market", "instrument_type"], dropna=False).size().rename("count").reset_index()


PRESETS = {
    # name: (markets, types)
    "lse_all": (("MAIN", "AIM"), EQUITY_TYPES),
    "main": (("MAIN",), EQUITY_TYPES),
    "aim": (("AIM",), EQUITY_TYPES),
    "lse_shares_only": (("MAIN", "AIM"), (SHARE,)),
    "everything": (None, None),
}


def apply_preset(universe: Universe, preset: str) -> Universe:
    """Apply a named preset: ``lse_all`` (default: Main Market + AIM ordinary shares and
    investment trusts), ``main``, ``aim``, ``lse_shares_only`` or ``everything``."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    markets, types = PRESETS[preset]
    return universe.filter(markets=markets, types=types)


# ---------------------------------------------------------------------------
# London Stock Exchange instrument list
# ---------------------------------------------------------------------------


def tidm_to_yahoo(tidm: str) -> str:
    """Convert an LSE TIDM to a Yahoo Finance symbol.

    ``VOD`` -> ``VOD.L``; ``BP.`` -> ``BP.L`` (trailing dot dropped); ``BT.A`` -> ``BT-A.L``.
    """
    t = str(tidm).strip().upper()
    t = t.rstrip(".")
    t = t.replace(".", "-").replace("/", "-")
    return f"{t}.L"


_COLUMN_ALIASES = {
    "tidm": ["tidm", "ticker", "epic", "mnemonic", "symbol"],
    "issuer": ["issuer name", "company name", "issuer", "company"],
    "instrument_name": ["instrument name", "security name", "instrument description", "description", "name"],
    "isin": ["isin", "isin code"],
    "mifir": ["mifir identifier code", "mifir identifier", "mifir code"],
    "mifir_desc": ["mifir identifier description", "mifir description"],
    "market": ["lse market", "market"],
    "segment": ["market segment code", "market segment", "segment"],
    "sector_code": ["market sector code", "market sector"],
    "listing": ["fca listing category", "listing category", "uk listing category", "listing segment"],
    "currency": ["trading currency", "currency", "price currency"],
    "country": ["country of incorporation", "country of register", "country"],
    "sector": ["icb industry", "icb super-sector name", "icb supersector", "industry", "ftse industry", "sector"],
}


def _norm(s: object) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _isna(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and pd.isna(v)) or (not isinstance(v, (list, tuple)) and pd.isna(v))
    except (TypeError, ValueError):
        return False


def _find_header(raw: pd.DataFrame, max_rows: int = 40) -> Optional[int]:
    for r in range(min(max_rows, len(raw))):
        cells = {_norm(v) for v in raw.iloc[r].tolist() if not _isna(v)}
        if "tidm" in cells or "epic" in cells:
            return r
    return None


def _map_columns(cols: Sequence[str]) -> dict:
    normed = {_norm(c): c for c in cols}
    mapping = {}
    for key, aliases in _COLUMN_ALIASES.items():
        for a in aliases:
            if a in normed and normed[a] not in mapping.values():
                mapping[key] = normed[a]
                break
    return mapping


def _classify(mifir: str, mifir_desc: str, listing: str, segment: str, sheet: str, name: str) -> str:
    m = (mifir or "").upper()
    text = " ".join([mifir_desc or "", listing or "", segment or "", sheet or "", name or ""]).lower()
    if m in ("ETFS",) or "exchange traded fund" in text or re.search(r"\betfs?\b", text):
        return ETF
    if m in ("ETCS", "ETNS") or re.search(r"\bet[cn]s?\b|exchange traded (commodit|note|product)", text):
        return ETP
    if m == "DPRS" or "depositary receipt" in text or re.search(r"\bgdr\b|\badr\b", text):
        return DEPOSITARY_RECEIPT
    is_shareish = m in ("SHRS", "") or "share" in text or "equity" in text
    if m and m not in ("SHRS", "OTHR"):
        return OTHER
    if any(
        k in text for k in ("closed-ended", "closed ended", "investment trust", "investment compan", "investment entit")
    ):
        return INVESTMENT_TRUST
    if any(k in text for k in ("bond", "debt", "gilt", "warrant", "preference", "loan note", "covered")):
        return OTHER
    return SHARE if is_shareish else OTHER


def _normalize_market(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().upper()
    if "AIM" in s:
        return "AIM"
    if "MAIN" in s:
        return "MAIN"
    return s or None


def parse_lse_instrument_frames(frames: dict) -> Universe:
    """Parse already-loaded sheets ``{sheet_name: raw DataFrame without header}``."""
    records = []
    found_any = False
    for sheet, raw in frames.items():
        if raw is None or raw.empty:
            continue
        h = _find_header(raw)
        if h is None:
            continue
        found_any = True
        df = raw.iloc[h + 1 :].copy()
        df.columns = [str(c) for c in raw.iloc[h].tolist()]
        df = df.loc[:, [c for c in df.columns if c and c.lower() != "nan"]]
        cmap = _map_columns(df.columns)
        if "tidm" not in cmap:
            continue

        def col(key, row, cmap=cmap):
            c = cmap.get(key)
            if c is None:
                return None
            v = row.get(c)
            return None if _isna(v) else str(v).strip()

        for row in df.to_dict("records"):
            tidm = col("tidm", row)
            if not tidm:
                continue
            name = col("issuer", row) or col("instrument_name", row)
            inst_name = col("instrument_name", row) or ""
            itype = _classify(
                col("mifir", row) or "",
                col("mifir_desc", row) or "",
                col("listing", row) or "",
                col("segment", row) or "",
                str(sheet),
                inst_name,
            )
            records.append(
                Instrument(
                    ticker=tidm_to_yahoo(tidm),
                    tidm=tidm,
                    name=name,
                    isin=col("isin", row),
                    market=_normalize_market(col("market", row)) or _normalize_market(str(sheet)),
                    instrument_type=itype,
                    currency=col("currency", row),
                    country=col("country", row),
                    sector=col("sector", row),
                )
            )
    if not found_any:
        raise ValueError(
            "Could not find a header row containing 'TIDM' in any sheet. Is this the LSE 'Instrument list' spreadsheet?"
        )
    # De-duplicate (sheets can overlap); the first occurrence wins.
    uni = Universe(records)
    log.info("Parsed %d instruments from LSE instrument list", len(uni))
    return uni


def load_lse_instrument_list(path: Union[str, Path]) -> Universe:
    """Parse the LSE "Instrument list" spreadsheet (``.xlsx``) into a :class:`Universe`.

    Download the latest file from the London Stock Exchange website (Reports ->
    Instrument list) and pass its path. Every sheet is scanned; rows are classified as
    shares, investment trusts, ETFs, ETPs, depositary receipts or other.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frames = {path.stem: pd.read_csv(path, header=None, dtype=str)}
    else:
        frames = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    return parse_lse_instrument_frames(frames)


def download_lse_instrument_list(dest: Union[str, Path], url: Optional[str] = None, timeout: float = 30.0) -> Path:
    """Best-effort download of the latest LSE instrument list.

    If ``url`` is given it is downloaded directly. Otherwise the LSE reports page is searched
    for a link to the instrument list. The LSE website changes often; if this fails, download
    the file manually and use :func:`load_lse_instrument_list`.
    """
    import requests

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (simtrade)"}
    if url is None:
        pages = [
            "https://www.londonstockexchange.com/reports?tab=instruments",
            "https://www.londonstockexchange.com/reports",
        ]
        found = None
        for page in pages:
            try:
                html = requests.get(page, headers=headers, timeout=timeout).text
            except Exception as exc:  # pragma: no cover - network
                log.debug("fetching %s failed: %s", page, exc)
                continue
            links = re.findall(r'https?://[^"\'\s>]+?\.xlsx', html) + [
                "https://docs.londonstockexchange.com" + m
                for m in re.findall(r'(/sites/default/files/[^"\'\s>]+?\.xlsx)', html)
            ]
            links = [link for link in links if "instrument" in link.lower()]
            if links:
                found = links[0]
                break
        if not found:
            raise RuntimeError(
                "Could not find the instrument list link automatically. Download it manually from "
                "https://www.londonstockexchange.com/reports?tab=instruments and run "
                "`simtrade universe load <file.xlsx>` (or pass --url)."
            )
        url = found
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest
