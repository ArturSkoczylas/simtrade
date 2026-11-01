import numpy as np
import pandas as pd
import pytest

from simtrade.calendar import trading_sessions
from simtrade.data.providers import FetchResult, PriceProvider, SyntheticProvider
from simtrade.data.schema import PRICE_COLUMNS, standardize
from simtrade.data.universe import (
    Instrument,
    Universe,
    apply_preset,
    load_lse_instrument_list,
    parse_lse_instrument_frames,
    tidm_to_yahoo,
)
from simtrade.data.updater import Updater
from simtrade.data.validate import normalize_currency, validate_prices
from simtrade.store import DataStore


# -- calendar -------------------------------------------------------------------------
def test_xlon_calendar_skips_uk_holidays():
    s = trading_sessions("2024-12-20", "2025-01-03")
    for holiday in ("2024-12-25", "2024-12-26", "2025-01-01"):
        assert pd.Timestamp(holiday) not in s
    assert pd.Timestamp("2024-12-24") in s
    assert pd.Timestamp("2024-05-27") not in trading_sessions("2024-05-20", "2024-05-31")  # spring bank holiday


# -- schema / validation -------------------------------------------------------------------
def test_standardize_renames_and_fills():
    df = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03", "2024-01-03"],
            "Close": [1.0, 2.0, 2.5],
            "Adj Close": [1.0, 2.0, 2.5],
            "Stock Splits": [0, 0, 0],
        }
    )
    out = standardize(df)
    assert list(out.columns) == PRICE_COLUMNS
    assert len(out) == 2 and out["close"].iloc[-1] == 2.5  # duplicate keeps last
    assert out["dividends"].eq(0).all()


def test_pence_converted_to_pounds():
    idx = pd.bdate_range("2024-01-01", periods=10)
    df = pd.DataFrame(
        {
            "open": 250.0,
            "high": 255.0,
            "low": 245.0,
            "close": 250.0,
            "adj_close": 250.0,
            "volume": 1e5,
            "dividends": 0.0,
            "splits": 0.0,
        },
        index=idx,
    )
    df.iloc[5, df.columns.get_loc("dividends")] = 5.0
    out, rep = validate_prices(df, "GBp")
    assert rep.currency_out == "GBP"
    assert out["close"].iloc[0] == pytest.approx(2.5)
    assert out["dividends"].iloc[5] == pytest.approx(0.05)
    assert normalize_currency("GBX") == ("GBP", 100.0)
    assert normalize_currency("USD") == ("USD", 1.0)


def test_100x_glitch_repaired():
    idx = pd.bdate_range("2024-01-01", periods=40)
    close = np.full(40, 2.50)
    close[20] = 250.0  # one row quoted in pence by mistake
    df = pd.DataFrame({"close": close, "open": close, "high": close, "low": close, "adj_close": close}, index=idx)
    out, rep = validate_prices(df, "GBP")
    assert out["close"].iloc[20] == pytest.approx(2.5)
    assert any(f.startswith("repaired_100x") for f in rep.flags)


def test_ohlc_consistency_and_bad_dividends():
    idx = pd.bdate_range("2024-01-01", periods=10)
    df = pd.DataFrame(
        {"open": np.nan, "high": 1.0, "low": 3.0, "close": 2.0, "adj_close": 2.0, "dividends": 0.0}, index=idx
    )
    df.iloc[3, df.columns.get_loc("dividends")] = 50.0  # pence-sized dividend in a pounds series
    out, rep = validate_prices(df, "GBP")
    assert (out["high"] >= out[["open", "close"]].max(axis=1)).all()
    assert (out["low"] <= out[["open", "close"]].min(axis=1)).all()
    assert out["dividends"].iloc[3] == pytest.approx(0.5)


# -- universe ---------------------------------------------------------------------------------
def test_tidm_to_yahoo():
    assert tidm_to_yahoo("VOD") == "VOD.L"
    assert tidm_to_yahoo("BP.") == "BP.L"
    assert tidm_to_yahoo("BT.A") == "BT-A.L"
    assert tidm_to_yahoo(" rr. ") == "RR.L"


def _instrument_sheet():
    header = [
        "TIDM",
        "Issuer Name",
        "Instrument Name",
        "ISIN",
        "MiFIR Identifier Code",
        "MiFIR Identifier Description",
        "ICB Industry",
        "Country of Incorporation",
        "Trading Currency",
        "LSE Market",
        "FCA Listing Category",
    ]
    rows = [
        [
            "VOD",
            "VODAFONE GROUP PLC",
            "ORD USD0.2095",
            "GB00BH4HKS39",
            "SHRS",
            "Shares",
            "Telecommunications",
            "United Kingdom",
            "GBX",
            "MAIN MARKET",
            "Equity shares (commercial companies)",
        ],
        [
            "BP.",
            "BP PLC",
            "ORD USD0.25",
            "GB0007980591",
            "SHRS",
            "Shares",
            "Energy",
            "United Kingdom",
            "GBX",
            "MAIN MARKET",
            "Equity shares (commercial companies)",
        ],
        [
            "SMT",
            "SCOTTISH MORTGAGE INV TST PLC",
            "ORD 5P",
            "GB00BLDYK618",
            "SHRS",
            "Shares",
            "Financials",
            "United Kingdom",
            "GBX",
            "MAIN MARKET",
            "Closed-ended investment funds",
        ],
        [
            "ABC",
            "ABC MINING PLC",
            "ORD 1P",
            "GB0000000001",
            "SHRS",
            "Shares",
            "Basic Materials",
            "Jersey",
            "GBX",
            "AIM",
            "",
        ],
        [
            "ISF",
            "ISHARES PLC",
            "ISHARES CORE FTSE 100 UCITS ETF",
            "IE0005042456",
            "ETFS",
            "Exchange Traded Funds",
            "",
            "Ireland",
            "GBX",
            "MAIN MARKET",
            "",
        ],
        [
            "GDR1",
            "FOREIGN CO",
            "GDR EACH REPR 1 SHS",
            "US0000000001",
            "DPRS",
            "Depositary Receipts",
            "Industrials",
            "Kazakhstan",
            "USD",
            "MAIN MARKET",
            "",
        ],
    ]
    title = [["List of instruments", None, None, None, None, None, None, None, None, None, None], [None] * 11]
    return pd.DataFrame(title + [header] + rows)


def test_parse_lse_instrument_list_finds_header_and_classifies(tmp_path):
    path = tmp_path / "instruments.xlsx"
    with pd.ExcelWriter(path) as xl:
        _instrument_sheet().to_excel(xl, sheet_name="1.0 All Equity", header=False, index=False)
        pd.DataFrame([["Notes only"]]).to_excel(xl, sheet_name="Notes", header=False, index=False)
    uni = load_lse_instrument_list(path)
    assert len(uni) == 6
    vod = uni.get("VOD.L")
    assert vod.market == "MAIN" and vod.instrument_type == "share" and vod.stamp_duty_eligible
    assert uni.get("BP.L") is not None
    assert uni.get("SMT.L").instrument_type == "investment_trust"
    abc = uni.get("ABC.L")
    assert abc.market == "AIM" and not abc.stamp_duty_eligible
    assert uni.get("ISF.L").instrument_type == "etf" and not uni.get("ISF.L").stamp_duty_eligible
    assert uni.get("GDR1.L").instrument_type == "depositary_receipt"
    lse = apply_preset(uni, "lse_all")
    assert set(lse.tickers) == {"VOD.L", "BP.L", "SMT.L", "ABC.L"}
    assert set(apply_preset(uni, "aim").tickers) == {"ABC.L"}


def test_parse_rejects_unrelated_file():
    with pytest.raises(ValueError):
        parse_lse_instrument_frames({"x": pd.DataFrame([["a", "b"], [1, 2]])})


def test_universe_roundtrip(tmp_path):
    store = DataStore(tmp_path)
    uni = Universe([Instrument("A.L", market="AIM"), Instrument("B.L", market="MAIN", stamp_duty=False)])
    store.save_universe(uni)
    back = store.load_universe()
    assert back.tickers == ["A.L", "B.L"]
    assert back.get("B.L").stamp_duty is False and not back.get("B.L").stamp_duty_eligible


# -- store + updater -----------------------------------------------------------------------------
class CountingSynthetic(SyntheticProvider):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.requests = []

    def fetch(self, tickers, start=None, end=None):
        self.requests.append((tuple(tickers), start, end))
        return super().fetch(tickers, start, end)


def test_updater_full_then_incremental(tmp_path):
    store = DataStore(tmp_path)
    prov = CountingSynthetic(base_start="2020-01-01", base_end="2021-12-31", missing_prob=0)
    up = Updater(store, prov, pause=0, default_start="2020-01-01")
    rep = up.update(["X.L", "Y.L"], end="2021-06-30")
    assert set(rep.new) == {"X.L", "Y.L"}
    df1 = store.read_prices("X.L")
    assert df1.index.max() == pd.Timestamp("2021-06-30")
    assert store.meta("X.L")["currency"] == "GBP"
    assert df1["close"].max() < 100  # converted from pence (synthetic prices are 50..2000p)

    rep2 = up.update(["X.L", "Y.L"], end="2021-12-31")
    assert set(rep2.updated) == {"X.L", "Y.L"}
    df2 = store.read_prices("X.L")
    assert df2.index.max() == pd.Timestamp("2021-12-31")
    # Incremental request started shortly before the previous last date, not from scratch.
    _, start, _ = prov.requests[-1]
    assert pd.Timestamp(start) > pd.Timestamp("2021-06-01")
    # Overlap rows identical to a fresh full download.
    fresh = standardize(prov.fetch(["X.L"], None, "2021-12-31")["X.L"].data)
    np.testing.assert_allclose(df2["close"].to_numpy(), fresh["close"].to_numpy() / 100, rtol=1e-9)

    rep3 = up.update(["X.L"], end="2021-12-31")
    assert rep3.unchanged == ["X.L"]


def test_updater_refreshes_when_history_readjusted(tmp_path):
    store = DataStore(tmp_path)
    prov = SyntheticProvider(base_start="2020-01-01", base_end="2020-12-31", missing_prob=0)
    up = Updater(store, prov, pause=0, default_start="2020-01-01")
    up.update(["X.L"], end="2020-06-30")

    class Readjusted(PriceProvider):
        name = "readjusted"

        def fetch(self, tickers, start=None, end=None):
            res = prov.fetch(tickers, start, end)
            for r in res.values():
                r.data = r.data.copy()
                r.data["adj_close"] *= 0.97  # a new dividend re-scaled history
            return res

        def currency(self, t):
            return "GBp"

    up2 = Updater(store, Readjusted(), pause=0, default_start="2020-01-01")
    rep = up2.update(["X.L"], end="2020-12-31")
    assert rep.refreshed == ["X.L"]
    df = store.read_prices("X.L")
    assert df.index.min() == pd.Timestamp("2020-01-02") and df.index.max() == pd.Timestamp("2020-12-31")


def test_updater_records_failures_and_retries(tmp_path):
    store = DataStore(tmp_path)
    calls = {"n": 0}

    class Flaky(PriceProvider):
        name = "flaky"

        def fetch(self, tickers, start=None, end=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("rate limited")
            return {t: FetchResult(t, None, error="no data returned") for t in tickers}

    up = Updater(store, Flaky(), pause=0, max_retries=3, sleep=lambda s: None)
    rep = up.update(["GONE.L"], end="2024-01-31")
    assert calls["n"] == 3
    assert "GONE.L" in rep.failed
    assert store.meta("GONE.L")["status"] == "empty"


def test_fx_conversion_for_foreign_currency(tmp_path):
    store = DataStore(tmp_path)
    prov = SyntheticProvider(
        base_start="2020-01-01", base_end="2020-12-31", missing_prob=0, currencies={"USDCO.L": "USD"}
    )
    up = Updater(store, prov, pause=0, default_start="2020-01-01")
    up.update(["GBCO.L", "USDCO.L"], end="2020-12-31")
    assert store.has("GBPUSD=X")
    raw = store.read_prices("USDCO.L")
    mk = store.load_market(["GBCO.L", "USDCO.L"])
    fx = store.read_prices("GBPUSD=X")["close"]
    j = mk.col("USDCO.L")
    d = mk.dates[50]
    assert mk.array("close")[50, j] == pytest.approx(raw.loc[d, "close"] / fx.loc[d])


def test_load_market_includes_lookback_before_start(tmp_path):
    store = DataStore(tmp_path)
    prov = SyntheticProvider(base_start="2020-01-01", base_end="2021-12-31", missing_prob=0)
    Updater(store, prov, pause=0, default_start="2020-01-01").update(["A.L"], end="2021-12-31", fx=False)
    mk = store.load_market(["A.L"], start="2021-01-04", lookback=20)
    i = mk.index_of("2021-01-04")
    assert i == 20
    assert mk.dates[-1] == pd.Timestamp("2021-12-31")


def test_status_frame(tmp_path):
    store = DataStore(tmp_path)
    store.write_prices("A.L", pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"])))
    st = store.status_frame()
    assert st.loc[st.ticker == "A.L", "last_date"].iloc[0] == "2024-01-03"
    assert DataStore(tmp_path).meta("A.L")["rows"] == 2  # manifest persisted
