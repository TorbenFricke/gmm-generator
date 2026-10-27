"""Factor profiles and critical time calculation for grid load activation probability p(on)."""

from dataclasses import dataclass
from typing import Sequence, Tuple, Union
import numpy as np


@dataclass
class CriticalPoint:
    """Represents the most critical point in time (highest p(on))."""

    month: int  # 1 to 12
    month_name: str
    day_type: str  # "weekday" or "weekend"
    interval_idx: int  # 0 to 95
    time_str: str  # e.g., "19:15"
    p_on: float
    month_factor: float
    interval_factor: float


@dataclass
class FactorProfiles:
    """Holds the 12 month factors and 96 15-minute interval factors for weekdays and weekends.

    Attributes:
        month_factors: 12 factors for months Jan (index 0) to Dec (index 11).
        weekday_factors: 96 factors for 15-minute intervals across a weekday (00:00 to 23:45).
        weekend_factors: 96 factors for 15-minute intervals across a weekend day (00:00 to 23:45).
    """

    month_factors: Sequence[float]
    weekday_factors: Sequence[float]
    weekend_factors: Sequence[float]

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

    def __post_init__(self):
        self.month_factors = np.asarray(self.month_factors, dtype=float)
        self.weekday_factors = np.asarray(self.weekday_factors, dtype=float)
        self.weekend_factors = np.asarray(self.weekend_factors, dtype=float)

        if len(self.month_factors) != 12:
            raise ValueError(f"Expected 12 month factors, got {len(self.month_factors)}")
        if len(self.weekday_factors) != 96:
            raise ValueError(f"Expected 96 weekday interval factors, got {len(self.weekday_factors)}")
        if len(self.weekend_factors) != 96:
            raise ValueError(f"Expected 96 weekend interval factors, got {len(self.weekend_factors)}")

        if np.any(self.month_factors < 0):
            raise ValueError("Month factors cannot be negative")
        if np.any(self.weekday_factors < 0):
            raise ValueError("Weekday interval factors cannot be negative")
        if np.any(self.weekend_factors < 0):
            raise ValueError("Weekend interval factors cannot be negative")

    @staticmethod
    def interval_to_time_str(interval_idx: int) -> str:
        """Convert a 15-minute interval index (0-95) to 'HH:MM' string."""
        if not (0 <= interval_idx < 96):
            raise ValueError(f"Interval index must be 0-95, got {interval_idx}")
        total_minutes = interval_idx * 15
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours:02d}:{minutes:02d}"

    def calculate_p_on(self, month: int, interval_idx: int, day_type: str) -> float:
        """Calculate p(on) for a specific month, interval, and day type clipped to [0, 1].

        Args:
            month: 1 to 12
            interval_idx: 0 to 95
            day_type: 'weekday' or 'weekend'
        """
        if not (1 <= month <= 12):
            raise ValueError(f"Month must be 1-12, got {month}")
        if not (0 <= interval_idx < 96):
            raise ValueError(f"Interval must be 0-95, got {interval_idx}")

        m_factor = self.month_factors[month - 1]
        if day_type.lower() == "weekday":
            i_factor = self.weekday_factors[interval_idx]
        elif day_type.lower() == "weekend":
            i_factor = self.weekend_factors[interval_idx]
        else:
            raise ValueError(f"Unknown day_type '{day_type}', must be 'weekday' or 'weekend'")

        return float(np.clip(m_factor * i_factor, 0.0, 1.0))

    def find_critical_point(self) -> CriticalPoint:
        """Identify the most critical point in time ahead of the Monte Carlo simulation.

        The critical point is the (month, day_type, interval) that maximizes p(on),
        representing the highest probability of peak grid demand.

        Returns:
            CriticalPoint dataclass with the peak time and p(on).
        """
        # Outer product: 12 x 96 for weekday and weekend
        p_weekday = np.outer(self.month_factors, self.weekday_factors)
        p_weekend = np.outer(self.month_factors, self.weekend_factors)

        max_weekday = np.max(p_weekday)
        max_weekend = np.max(p_weekend)

        if max_weekday >= max_weekend:
            day_type = "weekday"
            idx_2d = np.unravel_index(np.argmax(p_weekday), p_weekday.shape)
            raw_p = float(p_weekday[idx_2d])
            m_factor = float(self.month_factors[idx_2d[0]])
            i_factor = float(self.weekday_factors[idx_2d[1]])
        else:
            day_type = "weekend"
            idx_2d = np.unravel_index(np.argmax(p_weekend), p_weekend.shape)
            raw_p = float(p_weekend[idx_2d])
            m_factor = float(self.month_factors[idx_2d[0]])
            i_factor = float(self.weekend_factors[idx_2d[1]])

        month = int(idx_2d[0] + 1)
        interval_idx = int(idx_2d[1])
        p_on = float(np.clip(raw_p, 0.0, 1.0))

        return CriticalPoint(
            month=month,
            month_name=self.MONTH_NAMES[month - 1],
            day_type=day_type,
            interval_idx=interval_idx,
            time_str=self.interval_to_time_str(interval_idx),
            p_on=p_on,
            month_factor=m_factor,
            interval_factor=i_factor,
        )
