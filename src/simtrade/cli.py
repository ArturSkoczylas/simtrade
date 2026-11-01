"""Command-line interface: ``simtrade --help``."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer

from . import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="simtrade: daily trading simulator for London Stock Exchange equities.",
)
universe_app = typer.Typer(no_args_is_help=True, help="Manage the investable universe.")
data_app = typer.Typer(no_args_is_help=True, help="Download and inspect price data.")
app.add_typer(universe_app, name="universe")
app.add_typer(data_app, name="data")

DataDir = typer.Option(
    None, "--data-dir", "-d", envvar="SIMTRADE_HOME", help="Data store directory (default ~/.simtrade)."
)


def _store(data_dir: Optional[Path]):
    from .store import DataStore

    return DataStore(data_dir)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logging.")):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )


@app.command()
def version():
    """Print the installed version."""
    typer.echo(__version__)


# -- universe -------------------------------------------------------------------------
@universe_app.command("load")
def universe_load(
    path: Optional[Path] = typer.Argument(None, help="LSE instrument list (.xlsx). Omit with --download."),
    download: bool = typer.Option(False, "--download", help="Try to download the latest list from the LSE site."),
    url: Optional[str] = typer.Option(None, "--url", help="Direct URL of the instrument list file."),
    data_dir: Optional[Path] = DataDir,
):
    """Load the London Stock Exchange instrument list into the store."""
    from .data.universe import download_lse_instrument_list, load_lse_instrument_list

    store = _store(data_dir)
    if path is None:
        if not (download or url):
            typer.echo("Give a file path, or use --download / --url.", err=True)
            raise typer.Exit(2)
        path = store.root / "lse_instrument_list.xlsx"
        typer.echo("Downloading instrument list ...")
        download_lse_instrument_list(path, url=url)
    uni = load_lse_instrument_list(path)
    store.save_universe(uni)
    typer.echo(f"Saved {len(uni)} instruments to {store.root / 'universe.parquet'}")
    typer.echo(uni.summary().to_string(index=False))


@universe_app.command("tickers")
def universe_tickers(
    tickers: List[str] = typer.Argument(..., help="Yahoo tickers, e.g. VOD.L BP.L"),
    market: str = typer.Option("MAIN", help="Market for these tickers (MAIN or AIM; affects stamp duty)."),
    data_dir: Optional[Path] = DataDir,
):
    """Create/extend the universe from a plain list of tickers."""
    from .data.universe import Instrument, Universe

    store = _store(data_dir)
    existing = store.load_universe() if store.has_universe() else Universe()
    new = Universe(list(existing) + [Instrument(ticker=t, market=market.upper()) for t in tickers])
    store.save_universe(new)
    typer.echo(f"Universe now has {len(new)} instruments.")


@universe_app.command("show")
def universe_show(
    preset: str = typer.Option("lse_all", help="Preset filter: lse_all, main, aim, lse_shares_only, everything."),
    limit: int = typer.Option(20, help="Rows to print."),
    data_dir: Optional[Path] = DataDir,
):
    """Summarise the stored universe."""
    from .data.universe import apply_preset

    uni = apply_preset(_store(data_dir).load_universe(), preset)
    typer.echo(f"{len(uni)} instruments after preset '{preset}'")
    typer.echo(uni.summary().to_string(index=False))
    df = uni.to_frame()[["ticker", "name", "market", "instrument_type", "currency", "country"]]
    typer.echo(df.head(limit).to_string(index=False))


# -- data -----------------------------------------------------------------------------
@data_app.command("update")
def data_update(
    tickers: Optional[str] = typer.Option(None, "--tickers", "-t", help="Comma-separated tickers (default: universe)."),
    preset: str = typer.Option("lse_all", help="Universe preset when --tickers is not given."),
    start: Optional[str] = typer.Option(None, help="History start for new tickers (default 2000-01-01)."),
    end: Optional[str] = typer.Option(None, help="Last date (default: last completed LSE session)."),
    full: bool = typer.Option(False, help="Re-download full history for every ticker."),
    benchmark: Optional[str] = typer.Option("ISF.L", help="Also fetch this benchmark ('' to skip)."),
    batch_size: int = typer.Option(50, help="Tickers per request."),
    pause: float = typer.Option(1.0, help="Seconds to wait between requests."),
    limit: Optional[int] = typer.Option(None, help="Only the first N tickers (for a quick try)."),
    no_currency_lookup: bool = typer.Option(
        False, help="Assume .L tickers are quoted in pence instead of asking Yahoo."
    ),
    data_dir: Optional[Path] = DataDir,
):
    """Download new data or bring the store up to date (incremental)."""
    from .data.providers import YFinanceProvider
    from .data.universe import apply_preset
    from .data.updater import Updater

    store = _store(data_dir)
    if tickers:
        names = [t.strip() for t in tickers.split(",") if t.strip()]
    else:
        if not store.has_universe():
            typer.echo("No universe yet. Run `simtrade universe load <file.xlsx>` or pass --tickers.", err=True)
            raise typer.Exit(2)
        names = apply_preset(store.load_universe(), preset).tickers
    if limit:
        names = names[:limit]
    provider = YFinanceProvider(batch_size=batch_size, lookup_currency=not no_currency_lookup)

    def progress(done, total, last):
        typer.echo(f"\r  {done}/{total} tickers ({last})", nl=False)

    up = Updater(
        store,
        provider,
        batch_size=batch_size,
        pause=pause,
        progress=progress,
        **({"default_start": start} if start else {}),
    )
    typer.echo(f"Updating {len(names)} tickers into {store.root} ...")
    report = up.update(names, end=end, full=full)
    if benchmark:
        report.merge(up.update([benchmark], end=end, kind="index", fx=False))
    typer.echo("")
    typer.echo(report.summary())
    if report.failed:
        shown = list(report.failed.items())[:15]
        typer.echo("Failures (first 15):")
        for t, err in shown:
            typer.echo(f"  {t}: {err}")


@data_app.command("status")
def data_status(
    stale_days: int = typer.Option(5, help="Flag tickers whose last date is this many sessions behind."),
    data_dir: Optional[Path] = DataDir,
):
    """Summarise what is stored: counts, date coverage, stale and broken tickers."""
    import pandas as pd

    from .calendar import last_completed_session, trading_sessions

    store = _store(data_dir)
    df = store.status_frame()
    if df.empty:
        typer.echo(f"Store {store.root} is empty.")
        return
    typer.echo(f"Store: {store.root}")
    typer.echo(df.groupby(["kind", "status"], dropna=False).size().rename("count").to_string())
    ok = df[df["status"] == "ok"].copy()
    if not ok.empty:
        latest = last_completed_session()
        ok["last_date"] = pd.to_datetime(ok["last_date"])
        sess = trading_sessions(ok["last_date"].min(), latest)
        behind = ok["last_date"].map(lambda d: len(sess) - 1 - int(sess.searchsorted(d)))
        stale = ok[behind >= stale_days]
        typer.echo(f"Latest completed session: {latest.date()}")
        typer.echo(f"Coverage: {ok['first_date'].min()} .. {ok['last_date'].max().date()}")
        typer.echo(f"Stale (>= {stale_days} sessions behind; possibly delisted or suspended): {len(stale)}")
        flagged = ok[ok["flags"].map(lambda f: bool(f) if isinstance(f, list) else False)]
        typer.echo(f"Tickers with repaired/flagged data: {len(flagged)}")


# -- simulation -------------------------------------------------------------------------
@app.command()
def backtest(
    config: Path = typer.Argument(..., exists=True, help="Experiment YAML file."),
    data_dir: Optional[Path] = DataDir,
    out: Optional[Path] = typer.Option(None, help="Override output directory."),
):
    """Run a backtest described by a YAML experiment file."""
    from .experiment import ExperimentConfig, run_experiment

    cfg = ExperimentConfig.from_yaml(config)
    if data_dir is not None:
        cfg.data_dir = str(data_dir)
    if out is not None:
        cfg.output_dir = str(out)
    result, market = run_experiment(cfg, base_dir=config.parent.resolve())
    typer.echo(repr(market))
    if market.skipped:
        typer.echo(f"Skipped {len(market.skipped)} tickers without usable data.")
    typer.echo(result.summary())
    if cfg.output_dir:
        typer.echo(f"Results written to {Path(cfg.output_dir) / cfg.name}")


@app.command()
def demo(
    directory: Path = typer.Option(Path("simtrade-demo"), "--dir", help="Where to create the demo store and results."),
    n_tickers: int = typer.Option(40, help="Number of synthetic stocks."),
):
    """Run an offline demo on synthetic data: builds a store, then backtests three agents."""
    from .agents import EqualWeight, MomentumAgent, RandomAgent
    from .config import SimulationConfig
    from .data.sample import make_sample_store
    from .eval.metrics import format_metrics
    from .sim.engine import Simulation

    store_dir = directory / "store"
    typer.echo(f"Building synthetic store in {store_dir} ...")
    store = make_sample_store(store_dir, n_tickers=n_tickers)
    cfg = SimulationConfig(start="2017-01-01", rebalance="monthly", benchmark="SAMPLE_IDX.L")
    market = store.load_market(start=cfg.start, lookback=cfg.lookback, universe=store.load_universe())
    bench = store.read_prices("SAMPLE_IDX.L")["adj_close"]
    typer.echo(repr(market))
    for agent in (EqualWeight(max_assets=None), MomentumAgent(top_k=10), RandomAgent(n=10)):
        res = Simulation(market, cfg, benchmark=bench).run(agent)
        path = res.save(directory / "results" / agent.name.split("(")[0].lower())
        m = res.metrics()
        typer.echo(f"\n== {agent.name} ==  saved to {path}")
        typer.echo(
            format_metrics(
                {
                    k: m[k]
                    for k in ("total_return", "cagr", "sharpe", "max_drawdown", "annual_turnover", "total_costs")
                    if k in m
                }
            )
        )


def run():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
