"""JSON (de)serialization of model configurations (component + factor profiles)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from grid_gmm.factors import FactorProfiles
from grid_gmm.model import GaussianComponent

#: Version of the JSON layout written by :func:`save_config_to_json`.
#: Files without ``schema_version`` are treated as the legacy version 0 layout.
CONFIG_SCHEMA_VERSION = 1


def config_to_dict(component: GaussianComponent, factors: FactorProfiles) -> dict[str, Any]:
    """Serialize a configuration. The ``derived`` section is informational and ignored when loading."""
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "component": {
            "name": component.name,
            "p_max": component.p_max,
            "mu_ratio": component.mu_ratio,
            "sigma_ratio": component.sigma_ratio,
        },
        "factors": {
            "month_factors": factors.month_factors.tolist(),
            "weekday_factors": factors.weekday_factors.tolist(),
            "weekend_factors": factors.weekend_factors.tolist(),
        },
        "derived": {
            "_note": "Informational only; recomputed from 'component' and ignored when loading.",
            "mu_kw": component.mu,
            "sigma_kw": component.sigma,
            "mean_active_kw": component.mean,
        },
    }


def config_from_dict(data: dict[str, Any]) -> tuple[GaussianComponent, FactorProfiles]:
    """Deserialize a configuration written by :func:`config_to_dict` (or the legacy layout)."""
    version = int(data.get("schema_version", 0))
    if version > CONFIG_SCHEMA_VERSION:
        raise ValueError(f"Config schema_version {version} is newer than supported version {CONFIG_SCHEMA_VERSION}")

    c_data = data["component"]
    f_data = data["factors"]
    component = GaussianComponent(
        p_max=float(c_data["p_max"]),
        mu_ratio=float(c_data["mu_ratio"]),
        sigma_ratio=float(c_data["sigma_ratio"]),
        name=c_data.get("name", "Custom Load"),
    )
    factors = FactorProfiles(
        month_factors=f_data["month_factors"],
        weekday_factors=f_data["weekday_factors"],
        weekend_factors=f_data["weekend_factors"],
    )
    return component, factors


def save_config_to_json(
    component: GaussianComponent,
    factors: FactorProfiles,
    filepath: str | Path,
) -> None:
    """Save a component and factor profiles to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_to_dict(component, factors), f, indent=2)
        f.write("\n")


def load_config_from_json(filepath: str | Path) -> tuple[GaussianComponent, FactorProfiles]:
    """Load a component and factor profiles from a JSON file."""
    with open(filepath, encoding="utf-8") as f:
        return config_from_dict(json.load(f))
