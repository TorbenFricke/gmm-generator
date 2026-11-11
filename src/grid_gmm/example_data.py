"""Built-in example component and factor profiles."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from grid_gmm.config import save_config_to_json
from grid_gmm.constants import INTERVAL_HOURS, INTERVALS_PER_DAY
from grid_gmm.factors import FactorProfiles
from grid_gmm.model import GaussianComponent

#: (hour of day, factor) checkpoints, linearly interpolated to 15-minute intervals.
Checkpoints = Sequence[tuple[float, float]]

EXAMPLE_MONTH_FACTORS = (
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
)

EXAMPLE_WEEKDAY_CHECKPOINTS: Checkpoints = (
    (0.0, 0.10),
    (5.0, 0.08),
    (6.5, 0.22),
    (7.75, 0.36),  # Morning peak ~07:45
    (9.0, 0.25),
    (12.0, 0.20),  # Lunch
    (15.0, 0.18),
    (17.0, 0.28),
    (19.25, 0.48),  # Evening peak ~19:15
    (21.0, 0.38),
    (22.5, 0.22),
    (23.75, 0.12),
)

EXAMPLE_WEEKEND_CHECKPOINTS: Checkpoints = (
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
)


def interpolate_daily_profile(checkpoints: Checkpoints) -> np.ndarray:
    """Linearly interpolate (hour, value) checkpoints to the 96 interval starts of a day."""
    hours, values = zip(*checkpoints)
    interval_hours = np.arange(INTERVALS_PER_DAY) * INTERVAL_HOURS
    return np.round(np.interp(interval_hours, hours, values), 4)


def create_example_month_factors() -> np.ndarray:
    """12 seasonal month factors (higher in winter, lower in summer)."""
    return np.array(EXAMPLE_MONTH_FACTORS, dtype=float)


def create_example_weekday_factors() -> np.ndarray:
    """96 weekday interval factors with morning and evening peaks."""
    return interpolate_daily_profile(EXAMPLE_WEEKDAY_CHECKPOINTS)


def create_example_weekend_factors() -> np.ndarray:
    """96 weekend interval factors with a late-morning and an evening peak."""
    return interpolate_daily_profile(EXAMPLE_WEEKEND_CHECKPOINTS)


def create_example_profiles() -> FactorProfiles:
    """The default example FactorProfiles."""
    return FactorProfiles(
        month_factors=create_example_month_factors(),
        weekday_factors=create_example_weekday_factors(),
        weekend_factors=create_example_weekend_factors(),
    )


def create_example_component() -> GaussianComponent:
    """The default example load component (11 kW connection)."""
    return GaussianComponent(
        p_max=11.0,
        mu_ratio=0.35,  # mu = 3.85 kW
        sigma_ratio=0.15,  # sigma = 1.65 kW
        name="Household Connected Load",
    )


def export_example_config(filepath: str | Path) -> None:
    """Write the example configuration to a JSON file."""
    save_config_to_json(create_example_component(), create_example_profiles(), filepath)
