"""End-to-end tests of the command-line interface."""

import json
from pathlib import Path

import pytest

from grid_gmm.cli import main
from grid_gmm.config import load_config_from_json


def _out(tmp_path: Path) -> list[str]:
    return [
        "--csv",
        str(tmp_path / "r.csv"),
        "--plot",
        str(tmp_path / "r.png"),
        "--runs",
        "10",
        "--connections",
        "1",
        "5",
    ]


def test_default_run_writes_outputs(tmp_path: Path, capsys):
    assert main(_out(tmp_path)) == 0
    assert (tmp_path / "r.csv").exists()
    assert (tmp_path / "r.meta.json").exists()
    assert (tmp_path / "r.png").exists()
    assert "December" in capsys.readouterr().out


def test_invalid_override_is_rejected(tmp_path: Path, capsys):
    assert main([*_out(tmp_path), "--mu-ratio", "1.5"]) == 1
    assert "mu_ratio must be between 0 and 1" in capsys.readouterr().err


def test_overrides_and_save_config(tmp_path: Path):
    cfg = tmp_path / "cfg.json"
    assert main([*_out(tmp_path), "--p-max", "5", "--name", "Tiny", "--save-config", str(cfg)]) == 0
    component, _ = load_config_from_json(cfg)
    assert (component.p_max, component.name) == (5.0, "Tiny")

    assert main([*_out(tmp_path), "--config", str(cfg)]) == 0
    meta = json.loads((tmp_path / "r.meta.json").read_text())
    assert meta["component"]["name"] == "Tiny"


@pytest.mark.parametrize(
    "extra",
    [
        ["--config", "a.json", "--from-year-profile", "b.csv"],
        ["--simultaneity", "0.5"],
        ["--max-p-on", "0.9"],
        ["--timezone", "Europe/Berlin"],
        ["--from-year-profile", "b.csv", "--p-max", "5"],
    ],
)
def test_conflicting_options_exit_with_usage_error(tmp_path: Path, extra):
    with pytest.raises(SystemExit) as exc_info:
        main([*_out(tmp_path), *extra])
    assert exc_info.value.code == 2


def test_fit_from_year_profile(tmp_path: Path, profile_csv: Path, capsys):
    cfg = tmp_path / "fit.json"
    args = [*_out(tmp_path), "--from-year-profile", str(profile_csv), "--simultaneity", "0.5", "--max-p-on", "0.9"]
    assert main([*args, "--name", "Fitted", "--save-config", str(cfg)]) == 0
    component, factors = load_config_from_json(cfg)
    assert component.name == "Fitted"
    assert factors.find_critical_point().p_on == pytest.approx(0.9)
    assert "Derived Rusck P_max" in capsys.readouterr().out


def test_convert_w_profile(tmp_path: Path):
    src = tmp_path / "w.csv"
    src.write_text("ts,w\n2000-01-01 00:00:00+00:00,500\n2000-01-01 01:00:00+00:00,600\n")
    dst = tmp_path / "kw.csv"
    assert main(["--convert-w-profile", str(src), str(dst), "--value-column", "w"]) == 0
    assert len(dst.read_text().splitlines()) == 9


def test_save_example_config(tmp_path: Path):
    path = tmp_path / "example.json"
    assert main(["--save-example-config", str(path)]) == 0
    assert load_config_from_json(path)[0].p_max == 11.0
