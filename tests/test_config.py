"""Tests for JSON configuration serialization."""

import json
from pathlib import Path

import numpy as np
import pytest

from grid_gmm.config import CONFIG_SCHEMA_VERSION, load_config_from_json, save_config_to_json
from grid_gmm.example_data import create_example_component, create_example_profiles


def test_config_round_trip(tmp_path: Path):
    component, factors = create_example_component(), create_example_profiles()
    path = tmp_path / "cfg.json"
    save_config_to_json(component, factors, path)

    data = json.loads(path.read_text())
    assert data["schema_version"] == CONFIG_SCHEMA_VERSION
    assert "mu_kw" not in data["component"]  # derived values are kept apart

    loaded_component, loaded_factors = load_config_from_json(path)
    assert loaded_component == component
    for attr in ("month_factors", "weekday_factors", "weekend_factors"):
        np.testing.assert_array_equal(getattr(loaded_factors, attr), getattr(factors, attr))


def test_derived_section_is_ignored_on_load(tmp_path: Path):
    path = tmp_path / "cfg.json"
    save_config_to_json(create_example_component(), create_example_profiles(), path)
    data = json.loads(path.read_text())
    data["derived"]["mu_kw"] = 999.0
    path.write_text(json.dumps(data))
    assert load_config_from_json(path)[0].mu == pytest.approx(3.85)


def test_legacy_config_without_schema_version(tmp_path: Path):
    profiles = create_example_profiles()
    legacy = {
        "component": {"name": "Old", "p_max": 5.0, "mu_ratio": 0.5, "sigma_ratio": 0.1, "mu_kw": 2.5},
        "factors": {
            "month_factors": profiles.month_factors.tolist(),
            "weekday_factors": profiles.weekday_factors.tolist(),
            "weekend_factors": profiles.weekend_factors.tolist(),
        },
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))
    component, _ = load_config_from_json(path)
    assert (component.name, component.p_max) == ("Old", 5.0)


def test_newer_schema_version_is_rejected(tmp_path: Path):
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"schema_version": CONFIG_SCHEMA_VERSION + 1, "component": {}, "factors": {}}))
    with pytest.raises(ValueError, match="newer than supported"):
        load_config_from_json(path)


def test_committed_result_configs_load():
    for path in sorted(Path(__file__).resolve().parents[1].glob("results/*.json")):
        if not path.name.endswith(".meta.json"):
            load_config_from_json(path)
