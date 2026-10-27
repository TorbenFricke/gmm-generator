"""Example input dataset generation and serialization for grid usage simulation."""

import json
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np

from grid_gmm.factors import FactorProfiles
from grid_gmm.model import GaussianComponent


def create_example_month_factors() -> np.ndarray:
    """Generate 12 seasonal monthly factors (higher in winter, lower in summer)."""
    return np.array([
        1.25,  # Jan
        1.20,  # Feb
        1.10,  # Mar
        0.95,  # Apr
        0.85,  # May
        0.75,  # Jun
        0.70,  # Jul
        0.72,  # Aug
        0.85,  # Sep
        1.00,  # Oct
        1.18,  # Nov
        1.28,  # Dec (peak month)
    ], dtype=float)


def _generate_smooth_daily_profile(peaks_and_valleys: list) -> np.ndarray:
    """Interpolate a 96-step 15-min profile from key hourly checkpoints."""
    # checkpoints: list of (hour_float, value)
    hours = np.array([p[0] for p in peaks_and_valleys])
    vals = np.array([p[1] for p in peaks_and_valleys])
    
    # 96 intervals: 0.0, 0.25, 0.5, ..., 23.75 hours
    interval_hours = np.linspace(0.0, 23.75, 96)
    
    # Linear interpolation wrapping 24h
    interpolated = np.interp(interval_hours, hours, vals)
    return np.round(interpolated, 4)


def create_example_weekday_factors() -> np.ndarray:
    """Generate 96 weekday 15-minute interval factors with morning and evening peaks."""
    checkpoints = [
        (0.0, 0.10),
        (5.0, 0.08),
        (6.5, 0.22),
        (7.75, 0.36),   # Morning peak ~07:45
        (9.0, 0.25),
        (12.0, 0.20),  # Lunch
        (15.0, 0.18),
        (17.0, 0.28),
        (19.25, 0.48), # Evening peak ~19:15
        (21.0, 0.38),
        (22.5, 0.22),
        (23.75, 0.12),
    ]
    return _generate_smooth_daily_profile(checkpoints)


def create_example_weekend_factors() -> np.ndarray:
    """Generate 96 weekend 15-minute interval factors with late morning and evening spread."""
    checkpoints = [
        (0.0, 0.11),
        (6.5, 0.09),
        (9.0, 0.26),
        (11.0, 0.36),  # Late morning weekend peak
        (13.5, 0.30),
        (16.0, 0.24),
        (18.5, 0.42),  # Evening peak
        (21.0, 0.36),
        (22.5, 0.24),
        (23.75, 0.14),
    ]
    return _generate_smooth_daily_profile(checkpoints)


def create_example_profiles() -> FactorProfiles:
    """Create the default FactorProfiles object with realistic example factors."""
    return FactorProfiles(
        month_factors=create_example_month_factors(),
        weekday_factors=create_example_weekday_factors(),
        weekend_factors=create_example_weekend_factors(),
    )


def create_example_component() -> GaussianComponent:
    """Create the default Gaussian load component."""
    return GaussianComponent(
        p_max=11.0,        # 11 kW maximum power
        mu_ratio=0.35,     # mu = 3.85 kW (35% of P_max)
        sigma_ratio=0.15,  # sigma = 1.65 kW (15% of P_max)
        name="Household Connected Load",
    )


def save_config_to_json(
    component: GaussianComponent,
    factors: FactorProfiles,
    filepath: Union[str, Path],
) -> None:
    """Save any GaussianComponent and FactorProfiles configuration to a JSON file."""
    data = {
        "component": {
            "name": component.name,
            "p_max": component.p_max,
            "mu_ratio": component.mu_ratio,
            "sigma_ratio": component.sigma_ratio,
            "mu_kw": component.mu,
            "sigma_kw": component.sigma,
        },
        "factors": {
            "month_factors": factors.month_factors.tolist(),
            "weekday_factors": factors.weekday_factors.tolist(),
            "weekend_factors": factors.weekend_factors.tolist(),
        },
    }

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_example_config(filepath: Union[str, Path]) -> None:
    """Export the example configuration to a JSON file."""
    comp = create_example_component()
    profiles = create_example_profiles()
    save_config_to_json(comp, profiles, filepath)


def load_config_from_json(filepath: Union[str, Path]) -> tuple[GaussianComponent, FactorProfiles]:
    """Load component and factors from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

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
