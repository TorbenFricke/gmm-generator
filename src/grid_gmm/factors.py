"""Factor profiles and critical-time calculation for the activation probability p(on)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from grid_gmm.constants import (
    INTERVAL_MINUTES,
    INTERVALS_PER_DAY,
    MONTH_NAMES,
    MONTHS_PER_YEAR,
    DayType,
)

#: Order of the day-type axis in stacked (day_type, month, interval) arrays.
_DAY_TYPE_ORDER = (DayType.WEEKDAY, DayType.WEEKEND)


def interval_to_time_str(interval_idx: int) -> str:
    """Convert an interval index of the day (0-95) to an 'HH:MM' string."""
    if not (0 <= interval_idx < INTERVALS_PER_DAY):
        raise ValueError(f"Interval index must be 0-{INTERVALS_PER_DAY - 1}, got {interval_idx}")
    hours, minutes = divmod(interval_idx * INTERVAL_MINUTES, 60)
    return f"{hours:02d}:{minutes:02d}"


@dataclass(frozen=True)
class CriticalPoint:
    """The point in time with the highest activation probability p(on)."""

    month: int  # 1 to 12
    day_type: DayType
    interval_idx: int  # 0 to 95
    p_on: float
    month_factor: float
    interval_factor: float

    @property
    def month_name(self) -> str:
        return MONTH_NAMES[self.month - 1]

    @property
    def time_str(self) -> str:
        return interval_to_time_str(self.interval_idx)


@dataclass(frozen=True, eq=False)  # eq=False: element-wise array comparison has no single truth value
class FactorProfiles:
    """Separable p(on) model: ``p(on) = month_factor[m] * interval_factor[day_type][t]``, clipped to [0, 1].

    Attributes:
        month_factors: 12 factors for January (index 0) to December (index 11).
        weekday_factors: 96 factors for 15-minute intervals of a weekday (00:00 to 23:45).
        weekend_factors: 96 factors for 15-minute intervals of a weekend day (00:00 to 23:45).

    The arrays are copied and made read-only on construction.
    """

    month_factors: np.ndarray
    weekday_factors: np.ndarray
    weekend_factors: np.ndarray

    def __init__(
        self,
        month_factors: ArrayLike,
        weekday_factors: ArrayLike,
        weekend_factors: ArrayLike,
    ) -> None:
        specs = (
            ("month_factors", month_factors, MONTHS_PER_YEAR, "month factors"),
            ("weekday_factors", weekday_factors, INTERVALS_PER_DAY, "weekday interval factors"),
            ("weekend_factors", weekend_factors, INTERVALS_PER_DAY, "weekend interval factors"),
        )
        for attr, values, expected_len, label in specs:
            arr = np.array(values, dtype=float)
            if arr.shape != (expected_len,):
                raise ValueError(f"Expected {expected_len} {label}, got {arr.size}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{label.capitalize()} must be finite")
            if np.any(arr < 0):
                raise ValueError(f"{label.capitalize()} cannot be negative")
            arr.setflags(write=False)
            object.__setattr__(self, attr, arr)

    def interval_factors(self, day_type: DayType | str) -> np.ndarray:
        """Return the 96 interval factors for the given day type."""
        if DayType.parse(day_type) is DayType.WEEKDAY:
            return self.weekday_factors
        return self.weekend_factors

    def calculate_p_on(self, month: int, interval_idx: int, day_type: DayType | str) -> float:
        """Calculate p(on) for a month (1-12), interval (0-95) and day type, clipped to [0, 1]."""
        if not (1 <= month <= MONTHS_PER_YEAR):
            raise ValueError(f"Month must be 1-12, got {month}")
        if not (0 <= interval_idx < INTERVALS_PER_DAY):
            raise ValueError(f"Interval must be 0-{INTERVALS_PER_DAY - 1}, got {interval_idx}")

        raw = self.month_factors[month - 1] * self.interval_factors(day_type)[interval_idx]
        return float(np.clip(raw, 0.0, 1.0))

    def raw_products(self) -> np.ndarray:
        """Unclipped factor products with shape (day_type, month, interval) = (2, 12, 96)."""
        return np.stack([np.outer(self.month_factors, self.interval_factors(dt)) for dt in _DAY_TYPE_ORDER])

    def find_critical_point(self) -> CriticalPoint:
        """Identify the (month, day type, interval) that maximizes p(on).

        Ties are resolved in favour of weekdays, then earlier months, then earlier intervals.
        """
        products = self.raw_products()
        day_idx, month_idx, interval_idx = (int(i) for i in np.unravel_index(np.argmax(products), products.shape))
        day_type = _DAY_TYPE_ORDER[day_idx]

        return CriticalPoint(
            month=month_idx + 1,
            day_type=day_type,
            interval_idx=interval_idx,
            p_on=float(np.clip(products[day_idx, month_idx, interval_idx], 0.0, 1.0)),
            month_factor=float(self.month_factors[month_idx]),
            interval_factor=float(self.interval_factors(day_type)[interval_idx]),
        )
