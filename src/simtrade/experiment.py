"""YAML-configured experiments: universe + agent + features + simulation settings."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .config import SimulationConfig
from .data.universe import Instrument, Universe, apply_preset
from .store import DataStore


class ObjectSpec(BaseModel):
    """An importable class plus constructor arguments, e.g. ``{class: simtrade.agents.MomentumAgent, params: {...}}``.

    ``class`` accepts ``package.module.Class``, ``package.module:Class`` or ``path/to/file.py:Class``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    cls: str = Field(alias="class")
    params: Dict[str, Any] = Field(default_factory=dict)

    def build(self, base_dir: Optional[Path] = None):
        return load_object(self.cls, base_dir)(**self.params)


class UniverseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: Optional[str] = Field("lse_all", description="Preset applied to the stored universe.")
    tickers: Optional[List[str]] = Field(None, description="Explicit ticker list (overrides preset).")
    csv: Optional[str] = Field(None, description="CSV with a 'ticker' column (overrides preset).")
    markets: Optional[List[str]] = None
    types: Optional[List[str]] = None
    max_tickers: Optional[int] = Field(None, description="Keep only the first N (handy for quick tests).")


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "experiment"
    data_dir: Optional[str] = None
    universe: UniverseSpec = Field(default_factory=UniverseSpec)
    agent: ObjectSpec
    features: List[ObjectSpec] = Field(default_factory=list)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    output_dir: Optional[str] = "results"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        with open(path, encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))


def load_object(spec: str, base_dir: Optional[Path] = None):
    """Import ``module.Class``, ``module:Class`` or ``file.py:Class``."""
    if ":" in spec:
        mod, name = spec.rsplit(":", 1)
    else:
        mod, name = spec.rsplit(".", 1)
    if mod.endswith(".py"):
        path = Path(mod)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        modname = f"simtrade_user_{path.stem}"
        sp = importlib.util.spec_from_file_location(modname, path)
        if sp is None or sp.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(module)
    else:
        module = importlib.import_module(mod)
    return getattr(module, name)


def resolve_universe(store: DataStore, spec: UniverseSpec, base_dir: Optional[Path] = None) -> Universe:
    if spec.tickers:
        stored = store.load_universe() if store.has_universe() else Universe()
        uni = Universe([stored.get(t) or Instrument(ticker=t) for t in spec.tickers])
    elif spec.csv:
        p = Path(spec.csv)
        uni = Universe.from_csv(p if p.is_absolute() or base_dir is None else base_dir / p)
    else:
        uni = store.load_universe() if store.has_universe() else Universe.from_tickers(store.tickers(kind="equity"))
        if spec.preset and store.has_universe():
            uni = apply_preset(uni, spec.preset)
    if spec.markets or spec.types:
        uni = uni.filter(markets=spec.markets, types=spec.types)
    if spec.max_tickers:
        uni = Universe(list(uni)[: spec.max_tickers])
    return uni


def run_experiment(cfg: ExperimentConfig, base_dir: Optional[Path] = None, save: bool = True):
    """Load data, build agent and features, run the simulation and (optionally) save results."""
    from .sim.engine import Simulation

    store = DataStore(cfg.data_dir)
    uni = resolve_universe(store, cfg.universe, base_dir)
    sim_cfg = cfg.simulation
    market = store.load_market(
        uni.tickers,
        start=sim_cfg.start,
        end=sim_cfg.end,
        lookback=sim_cfg.lookback,
        base_currency=sim_cfg.base_currency,
        use_adjusted_prices=sim_cfg.use_adjusted_prices,
        universe=uni,
    )
    bench = None
    if sim_cfg.benchmark:
        if store.has(sim_cfg.benchmark):
            bench = store.read_prices(sim_cfg.benchmark)["adj_close"]
        else:
            import logging

            logging.getLogger(__name__).warning(
                "benchmark %s not in store; run `simtrade data update --tickers %s`",
                sim_cfg.benchmark,
                sim_cfg.benchmark,
            )
    agent = cfg.agent.build(base_dir)
    features = [f.build(base_dir) for f in cfg.features]
    sim = Simulation(market, sim_cfg, features or None, benchmark=bench)
    result = sim.run(agent)
    if save and cfg.output_dir:
        out = Path(cfg.output_dir)
        if not out.is_absolute() and base_dir is not None:
            out = base_dir / out
        result.save(out / cfg.name)
    return result, market
