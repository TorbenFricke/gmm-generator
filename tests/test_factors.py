"""Unit tests for FactorProfiles and critical point calculation."""

import numpy as np
import pytest
from grid_gmm.factors import FactorProfiles
from grid_gmm.example_data import create_example_profiles


def test_factor_profiles_dimensions():
    # Invalid months count
    with pytest.raises(ValueError, match="Expected 12 month factors"):
        FactorProfiles(
            month_factors=[1.0] * 10,
            weekday_factors=[1.0] * 96,
            weekend_factors=[1.0] * 96,
        )

    # Invalid weekday count
    with pytest.raises(ValueError, match="Expected 96 weekday interval factors"):
        FactorProfiles(
            month_factors=[1.0] * 12,
            weekday_factors=[1.0] * 50,
            weekend_factors=[1.0] * 96,
        )

    # Invalid weekend count
    with pytest.raises(ValueError, match="Expected 96 weekend interval factors"):
        FactorProfiles(
            month_factors=[1.0] * 12,
            weekday_factors=[1.0] * 96,
            weekend_factors=[1.0] * 90,
        )


def test_critical_point_detection():
    # Construct synthetic known factors
    month_factors = [1.0] * 12
    month_factors[11] = 2.0  # Dec is highest (month 12)

    weekday_factors = [0.1] * 96
    weekday_factors[76] = 0.5  # interval 76 (19:00) is highest weekday

    weekend_factors = [0.1] * 96
    weekend_factors[40] = 0.4  # lower than weekday peak

    profiles = FactorProfiles(
        month_factors=month_factors,
        weekday_factors=weekday_factors,
        weekend_factors=weekend_factors,
    )

    critical = profiles.find_critical_point()
    assert critical.month == 12
    assert critical.month_name == "December"
    assert critical.day_type == "weekday"
    assert critical.interval_idx == 76
    assert critical.time_str == "19:00"
    assert critical.p_on == pytest.approx(1.0)  # clipped 2.0 * 0.5 = 1.0


def test_example_profiles_critical_point():
    profiles = create_example_profiles()
    cp = profiles.find_critical_point()
    assert 1 <= cp.month <= 12
    assert cp.day_type in ("weekday", "weekend")
    assert 0 <= cp.interval_idx < 96
    assert 0.0 < cp.p_on <= 1.0
