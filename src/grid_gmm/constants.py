"""Shared domain constants and small enums used across the package."""

from __future__ import annotations

from enum import Enum

from scipy.stats import norm

MONTHS_PER_YEAR = 12
MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

#: Temporal resolution of the factor profiles.
INTERVAL_MINUTES = 15
INTERVALS_PER_HOUR = 60 // INTERVAL_MINUTES
INTERVALS_PER_DAY = 24 * INTERVALS_PER_HOUR
INTERVAL_HOURS = INTERVAL_MINUTES / 60

#: Standard-normal 99th percentile (Rusck's design quantile), ~2.32635.
Z_99 = float(norm.ppf(0.99))

#: Profiles whose peak exceeds this value are assumed to be in W rather than kW.
KW_PEAK_THRESHOLD = 20.0


class DayType(str, Enum):
    """Day category used to select the interval factor profile."""

    WEEKDAY = "weekday"
    WEEKEND = "weekend"

    @classmethod
    def parse(cls, value: DayType | str) -> DayType:
        """Accept a DayType or a case-insensitive string ('weekday' / 'weekend')."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            raise ValueError(f"Unknown day_type '{value}', must be 'weekday' or 'weekend'") from None

    @property
    def label(self) -> str:
        return self.value.capitalize()
