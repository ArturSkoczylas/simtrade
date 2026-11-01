import json

import yaml
from typer.testing import CliRunner

from simtrade.cli import app
from simtrade.data.sample import make_sample_store

runner = CliRunner()


def test_demo_command(tmp_path):
    res = runner.invoke(app, ["demo", "--dir", str(tmp_path), "--n-tickers", "12"])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "results" / "equalweight" / "metrics.json").exists()
    m = json.loads((tmp_path / "results" / "momentum" / "metrics.json").read_text())
    assert "sharpe" in m


def test_backtest_from_yaml_with_custom_agent_file(tmp_path):
    store_dir = tmp_path / "store"
    make_sample_store(store_dir, n_tickers=10, start="2018-01-01", end="2020-12-31")
    (tmp_path / "my_agent.py").write_text(
        "from simtrade import Agent\n"
        "class Top3(Agent):\n"
        "    name = 'Top3'\n"
        "    def act(self, obs):\n"
        "        f = obs.features_now()['ret_21'].dropna().nlargest(3)\n"
        "        return {t: 1/3 for t in f.index}\n"
    )
    cfg = {
        "name": "top3",
        "data_dir": str(store_dir),
        "universe": {"preset": "lse_all"},
        "agent": {"class": "my_agent.py:Top3"},
        "features": [{"class": "simtrade.features.Returns", "params": {"windows": [21]}}],
        "simulation": {
            "start": "2019-01-01",
            "rebalance": "weekly",
            "benchmark": "SAMPLE_IDX.L",
            "broker": {"costs": {"commission_fixed": 3.0}},
        },
        "output_dir": "out",
    }
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(cfg))
    res = runner.invoke(app, ["backtest", str(path)])
    assert res.exit_code == 0, res.output
    out = tmp_path / "out" / "top3"
    assert (out / "trades.csv").exists() and (out / "equity.png").exists()
    m = json.loads((out / "metrics.json").read_text())
    assert "information_ratio" in m


def test_status_and_universe_commands(tmp_path):
    make_sample_store(tmp_path, n_tickers=5, start="2020-01-01", end="2020-12-31")
    res = runner.invoke(app, ["data", "status", "--data-dir", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert "equity" in res.output
    res = runner.invoke(app, ["universe", "show", "--data-dir", str(tmp_path)])
    assert res.exit_code == 0 and "5 instruments" in res.output
    res = runner.invoke(app, ["universe", "tickers", "VOD.L", "BP.L", "--data-dir", str(tmp_path)])
    assert res.exit_code == 0 and "7 instruments" in res.output
