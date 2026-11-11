"""Unit tests for FactorProfiles and critical point calculation."""

import numpy as np
import pytest

from grid_gmm.constants import DayType
from grid_gmm.example_data import create_example_profiles
from grid_gmm.factors import FactorProfiles, interval_to_time_str


def _profiles(month=1.0, weekday=0.1, weekend=0.1):
    return FactorProfiles([month] * 12, [weekday] * 96, [weekend] * 96)


@pytest.mark.parametrize(
    ("months", "weekday", "weekend", "message"),
    [
        (10, 96, 96, "Expected 12 month factors"),
        (12, 50, 96, "Expected 96 weekday interval factors"),
        (12, 96, 90, "Expected 96 weekend interval factors"),
    ],
)
def test_factor_profiles_dimensions(months, weekday, weekend, message):
    with pytest.raises(ValueError, match=message):
        FactorProfiles([1.0] * months, [1.0] * weekday, [1.0] * weekend)


def test_factor_profiles_reject_negative_and_nonfinite():
    with pytest.raises(ValueError, match="Month factors cannot be negative"):
        FactorProfiles([-1.0] + [1.0] * 11, [1.0] * 96, [1.0] * 96)
    with pytest.raises(ValueError, match="must be finite"):
        FactorProfiles([np.nan] + [1.0] * 11, [1.0] * 96, [1.0] * 96)


def test_factor_arrays_are_read_only_copies():
    source = np.ones(12)
    profiles = FactorProfiles(source, [1.0] * 96, [1.0] * 96)
    source[0] = 99.0
    assert profiles.month_factors[0] == 1.0
    with pytest.raises(ValueError):
        profiles.month_factors[0] = 5.0


def test_critical_point_detection():
    month_factors = [1.0] * 12
    month_factors[11] = 2.0  # December
    weekday_factors = [0.1] * 96
    weekday_factors[76] = 0.5  # 19:00
    weekend_factors = [0.1] * 96
    weekend_factors[40] = 0.4

    cp = FactorProfiles(month_factors, weekday_factors, weekend_factors).find_critical_point()
    assert cp.month == 12
    assert cp.month_name == "December"
    assert cp.day_type is DayType.WEEKDAY
    assert cp.interval_idx == 76
    assert cp.time_str == "19:00"
    assert cp.p_on == pytest.approx(1.0)  # clipped 2.0 * 0.5
    assert cp.month_factor == 2.0
    assert cp.interval_factor == 0.5


def test_critical_point_weekend_and_tie_break():
    weekend = [0.1] * 96
    weekend[10] = 0.9
    cp = FactorProfiles([1.0] * 12, [0.1] * 96, weekend).find_critical_point()
    assert cp.day_type is DayType.WEEKEND
    assert cp.interval_idx == 10

    tie = _profiles().find_critical_point()
    assert (tie.day_type, tie.month, tie.interval_idx) == (DayType.WEEKDAY, 1, 0)


def test_calculate_p_on():
    profiles = _profiles(month=2.0, weekday=0.2, weekend=0.7)
    assert profiles.calculate_p_on(1, 0, "weekday") == pytest.approx(0.4)
    assert profiles.calculate_p_on(1, 0, "Weekend") == pytest.approx(1.0)  # clipped
    assert profiles.calculate_p_on(1, 0, DayType.WEEKDAY) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="Unknown day_type"):
        profiles.calculate_p_on(1, 0, "holiday")
    with pytest.raises(ValueError, match="Month must be 1-12"):
        profiles.calculate_p_on(13, 0, "weekday")


def test_interval_to_time_str():
    assert interval_to_time_str(0) == "00:00"
    assert interval_to_time_str(77) == "19:15"
    assert interval_to_time_str(95) == "23:45"
    with pytest.raises(ValueError):
        interval_to_time_str(96)


def test_example_profiles_critical_point():
    cp = create_example_profiles().find_critical_point()
    assert (cp.month, cp.day_type, cp.time_str) == (12, DayType.WEEKDAY, "19:15")
    assert 0.0 < cp.p_on <= 1.0
