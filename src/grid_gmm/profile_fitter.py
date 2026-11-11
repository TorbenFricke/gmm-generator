"""Derive Gaussian model parameters and factor profiles from an annual load profile.

Pipeline:
    1. :func:`load_year_profile_csv` reads a ``timestamp, value`` CSV into a :class:`YearProfile`
       (unit detection, timezone conversion, resampling to 15-minute intervals).
    2. :func:`derive_component` derives the :class:`GaussianComponent` from the profile peak (Rusck).
    3. :func:`decompose_factors` decomposes the profile into month and interval factors.
    4. :func:`fit_from_profile` / :func:`fit_gm_from_year_profile` combine the steps.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from pathlib import Path
from typing import Optional, Union
from zoneinfo import ZoneInfo

import numpy as np
from dateutil import parser as date_parser

from grid_gmm.constants import (
    INTERVAL_HOURS,
    INTERVAL_MINUTES,
    INTERVALS_PER_DAY,
    KW_PEAK_THRESHOLD,
    MONTHS_PER_YEAR,
    Z_99,
)
from grid_gmm.factors import FactorProfiles
from grid_gmm.model import GaussianComponent

PathLike = Union[str, Path]
ValueColumn = Optional[Union[str, int]]

#: Resolutions (minutes) that can be expanded to the 15-minute model resolution.
SUPPORTED_RESOLUTIONS = (15, 30, 60)
#: Column picked automatically when the CSV has a header but no column is specified.
DEFAULT_VALUE_COLUMN_NAME = "consumption_kw"

_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
)


@dataclass(frozen=True, eq=False)
class YearProfile:
    """A load profile at 15-minute resolution, as aligned NumPy arrays.

    Attributes:
        months: Month of each interval (1-12).
        is_weekend: True for Saturday/Sunday.
        interval_idx: Interval of the day (0-95).
        power_kw: Average power in kW.
        source_resolution_minutes: Resolution of the source data before expansion.
        source_unit: "kW" or "W" (as detected from the profile peak).
    """

    months: np.ndarray
    is_weekend: np.ndarray
    interval_idx: np.ndarray
    power_kw: np.ndarray
    source_resolution_minutes: int = INTERVAL_MINUTES
    source_unit: str = "kW"

    def __len__(self) -> int:
        return len(self.power_kw)

    @property
    def peak_kw(self) -> float:
        return float(np.max(self.power_kw))

    @property
    def energy_kwh(self) -> float:
        return float(np.sum(self.power_kw) * INTERVAL_HOURS)


@dataclass(frozen=True)
class ProfileFitSummary:
    """Summary of parameters derived from the annual load profile."""

    p_peak: float
    simultaneity: float
    p_max: float
    mu_ratio: float
    sigma_ratio: float
    peak_p_on: float
    annual_energy_kwh: float
    total_records: int
    source_unit: str = "kW"
    source_resolution_minutes: int = INTERVAL_MINUTES


# --------------------------------------------------------------------------------------
# CSV parsing
# --------------------------------------------------------------------------------------


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse a timestamp using ISO 8601, a list of common formats, or dateutil as fallback."""
    ts_str = ts_str.strip()
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        pass

    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            pass

    try:
        return date_parser.parse(ts_str)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Unable to parse timestamp: '{ts_str}'") from exc


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _resolve_value_column(header: list[str] | None, value_column: ValueColumn) -> int:
    """Return the index of the value column; raises if a named column does not exist."""
    if value_column is None:
        if header and DEFAULT_VALUE_COLUMN_NAME in header:
            return header.index(DEFAULT_VALUE_COLUMN_NAME)
        return 1
    if isinstance(value_column, int) or str(value_column).isdigit():
        return int(value_column)
    if header is None:
        raise ValueError(f"Column '{value_column}' requested, but the CSV has no header row")
    if value_column not in header:
        raise ValueError(f"Column '{value_column}' not found in CSV header {header}")
    return header.index(value_column)


def _read_timestamp_value_rows(path: Path, value_column: ValueColumn) -> tuple[list[datetime], np.ndarray]:
    """Read (timestamp, value) pairs from column 0 and the selected value column."""
    if not path.is_file():
        raise FileNotFoundError(f"Profile file not found: {path}")

    with open(path, encoding="utf-8-sig", newline="") as f:
        numbered_rows = [(line_no, row) for line_no, row in enumerate(csv.reader(f), start=1) if row]

    if not numbered_rows:
        raise ValueError(f"CSV file is empty: {path}")

    first_row = numbered_rows[0][1]
    has_header = len(first_row) < 2 or not _is_number(first_row[1])
    header = [cell.strip() for cell in first_row] if has_header else None
    col_idx = _resolve_value_column(header, value_column)
    data_rows = numbered_rows[1:] if has_header else numbered_rows

    timestamps: list[datetime] = []
    values: list[float] = []
    for line_no, row in data_rows:
        if len(row) <= col_idx:
            continue
        ts_str, val_str = row[0].strip(), row[col_idx].strip()
        if not ts_str or not val_str:
            continue
        try:
            timestamps.append(_parse_timestamp(ts_str))
            values.append(float(val_str))
        except ValueError as exc:
            raise ValueError(f"Error parsing row {line_no} ({row}): {exc}") from exc

    if not values:
        raise ValueError(f"No valid data rows found in {path}")
    return timestamps, np.array(values, dtype=float)


def _infer_resolution_minutes(timestamps: list[datetime]) -> int:
    """Infer the sampling resolution from the median spacing of consecutive timestamps."""
    if len(timestamps) < 2:
        return INTERVAL_MINUTES
    deltas = [(b - a).total_seconds() / 60 for a, b in zip(timestamps, timestamps[1:])]
    positive = [d for d in deltas if d > 0]
    if not positive:
        raise ValueError("Timestamps must be increasing to infer the profile resolution")
    resolution = int(round(float(np.median(positive))))
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError(f"Unsupported profile resolution of {resolution} min; supported: {SUPPORTED_RESOLUTIONS}")
    return resolution


def _normalize_to_kw(values: np.ndarray, threshold: float = KW_PEAK_THRESHOLD) -> tuple[np.ndarray, str]:
    """Heuristic unit detection: a peak above ``threshold`` means the values are in W."""
    peak = float(np.max(values))
    if peak > threshold:
        warnings.warn(
            f"Profile peak {peak:.1f} exceeds {threshold:g}; assuming values are in W and converting to kW.",
            stacklevel=3,
        )
        return values / 1000.0, "W"
    return values, "kW"


def _resolve_timezone(timezone: str | tzinfo | None) -> tzinfo | None:
    if timezone is None or isinstance(timezone, tzinfo):
        return timezone
    return ZoneInfo(timezone)


def load_year_profile_csv(
    filepath: PathLike,
    value_column: ValueColumn = None,
    timezone: str | tzinfo | None = None,
) -> YearProfile:
    """Load a ``timestamp, value`` CSV into a 15-minute :class:`YearProfile`.

    Args:
        filepath: CSV path. Column 0 holds timestamps; an optional header row is detected.
        value_column: Column name or index of the power values. Defaults to ``consumption_kw`` if
            present in the header, otherwise column 1.
        timezone: IANA zone (e.g. ``"Europe/Berlin"``). Timezone-aware timestamps are converted to it
            before deriving month, weekday and time of day. Naive timestamps are taken as local time.
            ``None`` uses the wall-clock time exactly as written in the file.

    Values are interpreted as kW unless the peak exceeds :data:`KW_PEAK_THRESHOLD`, in which case
    they are treated as W and divided by 1000. Negative values are clipped to 0. Hourly and
    30-minute data are expanded to 15-minute intervals (constant power within the hour).
    """
    path = Path(filepath)
    timestamps, values = _read_timestamp_value_rows(path, value_column)
    resolution = _infer_resolution_minutes(timestamps)

    tz = _resolve_timezone(timezone)
    if tz is not None:
        timestamps = [ts.astimezone(tz) if ts.tzinfo is not None else ts for ts in timestamps]

    negatives = int(np.sum(values < 0))
    if negatives:
        warnings.warn(f"{negatives} negative power value(s) clipped to 0.0 in {path}", stacklevel=2)
        values = np.clip(values, 0.0, None)

    power_kw, unit = _normalize_to_kw(values)

    base_interval = np.array([ts.hour * 4 + ts.minute // INTERVAL_MINUTES for ts in timestamps], dtype=int)
    expansion = resolution // INTERVAL_MINUTES
    if np.any(base_interval % expansion):
        raise ValueError(f"Timestamps of {resolution}-minute data must be aligned to {resolution}-minute boundaries")

    sub_steps = np.tile(np.arange(expansion), len(timestamps))
    return YearProfile(
        months=np.repeat([ts.month for ts in timestamps], expansion).astype(int),
        is_weekend=np.repeat([ts.weekday() >= 5 for ts in timestamps], expansion).astype(bool),
        interval_idx=np.repeat(base_interval, expansion) + sub_steps,
        power_kw=np.repeat(power_kw, expansion).astype(float),
        source_resolution_minutes=resolution,
        source_unit=unit,
    )


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------


def derive_component(
    p_peak: float,
    simultaneity: float,
    max_p_on: float = 1.0,
    mu_ratio: float | None = None,
    sigma_ratio: float | None = None,
    name: str = "Fitted Household Connection",
) -> GaussianComponent:
    """Derive the load component from the diversified profile peak following Rusck.

    - ``P_max = P_peak / s``
    - ``mu_ratio = s / max_p_on`` (so that ``mu * max_p_on = P_peak``)
    - ``sigma_ratio = (1 - mu_ratio) / z_0.99`` (so that ``mu + z_0.99 * sigma = P_max``)

    Note that the design relations use the location ``mu``; the truncated mean
    (:attr:`GaussianComponent.mean`) deviates slightly when the truncation is asymmetric.
    """
    if not (0.0 < simultaneity <= 1.0):
        raise ValueError(f"Simultaneity must be in (0, 1], got {simultaneity}")
    if not (0.0 < max_p_on <= 1.0):
        raise ValueError(f"max_p_on must be in (0, 1], got {max_p_on}")
    if p_peak <= 0:
        raise ValueError("Maximum power consumption in year profile must be strictly positive")

    eff_mu_ratio = simultaneity / max_p_on if mu_ratio is None else float(mu_ratio)
    if not (0.0 < eff_mu_ratio <= 1.0):
        raise ValueError(f"mu_ratio must be in (0, 1], got {eff_mu_ratio}. Check that simultaneity <= max_p_on.")

    eff_sigma_ratio = max(0.0, 1.0 - eff_mu_ratio) / Z_99 if sigma_ratio is None else float(sigma_ratio)

    return GaussianComponent(
        p_max=p_peak / simultaneity,
        mu_ratio=eff_mu_ratio,
        sigma_ratio=eff_sigma_ratio,
        name=name,
    )


def _relative_group_means(values: np.ndarray, groups: np.ndarray, n_groups: int, reference: float) -> np.ndarray:
    """Mean of ``values`` per group divided by ``reference``; empty groups get a neutral factor of 1."""
    sums = np.bincount(groups, weights=values, minlength=n_groups)
    counts = np.bincount(groups, minlength=n_groups)
    factors = np.ones(n_groups, dtype=float)
    present = counts > 0
    factors[present] = sums[present] / counts[present] / reference
    return factors


def decompose_factors(profile: YearProfile, peak_p_on: float) -> FactorProfiles:
    """Decompose a profile into month and weekday/weekend interval factors.

    ``p(on, t) = peak_p_on * P(t) / P_peak`` is averaged per month and per interval (relative to
    the annual mean). Interval factors are then scaled so ``max(month_factor * interval_factor)``
    equals ``peak_p_on``.
    """
    p_on_series = np.clip(peak_p_on * profile.power_kw / profile.peak_kw, 0.0, 1.0)
    mean_p_on = float(np.mean(p_on_series))

    month_factors = _relative_group_means(p_on_series, profile.months - 1, MONTHS_PER_YEAR, mean_p_on)
    weekday = ~profile.is_weekend
    weekday_raw = _relative_group_means(
        p_on_series[weekday], profile.interval_idx[weekday], INTERVALS_PER_DAY, mean_p_on
    )
    weekend_raw = _relative_group_means(
        p_on_series[~weekday], profile.interval_idx[~weekday], INTERVALS_PER_DAY, mean_p_on
    )

    raw = FactorProfiles(month_factors, weekday_raw, weekend_raw)
    raw_max = float(np.max(raw.raw_products()))
    scale = peak_p_on / raw_max if raw_max > 0 else 1.0

    return FactorProfiles(month_factors, weekday_raw * scale, weekend_raw * scale)


def fit_from_profile(
    profile: YearProfile,
    simultaneity: float,
    max_p_on: float = 1.0,
    mu_ratio: float | None = None,
    sigma_ratio: float | None = None,
    component_name: str = "Fitted Household Connection",
) -> tuple[GaussianComponent, FactorProfiles, ProfileFitSummary]:
    """Fit component and factor profiles from an already loaded :class:`YearProfile`."""
    p_peak = profile.peak_kw
    component = derive_component(p_peak, simultaneity, max_p_on, mu_ratio, sigma_ratio, component_name)
    factors = decompose_factors(profile, peak_p_on=max_p_on)

    summary = ProfileFitSummary(
        p_peak=p_peak,
        simultaneity=simultaneity,
        p_max=component.p_max,
        mu_ratio=component.mu_ratio,
        sigma_ratio=component.sigma_ratio,
        peak_p_on=float(max_p_on),
        annual_energy_kwh=profile.energy_kwh,
        total_records=len(profile),
        source_unit=profile.source_unit,
        source_resolution_minutes=profile.source_resolution_minutes,
    )
    return component, factors, summary


def fit_gm_from_year_profile(
    csv_path: PathLike,
    simultaneity: float,
    max_p_on: float = 1.0,
    mu_ratio: float | None = None,
    sigma_ratio: float | None = None,
    component_name: str = "Fitted Household Connection",
    value_column: ValueColumn = None,
    timezone: str | tzinfo | None = None,
) -> tuple[GaussianComponent, FactorProfiles, ProfileFitSummary]:
    """Load a year profile CSV and fit component and factors matching Rusck's simultaneity model.

    - At the profile peak, the activation probability is ``max_p_on``; elsewhere
      ``p(on, t) = max_p_on * P(t) / P_peak``.
    - The component follows :func:`derive_component`, so the fleet expectation
      ``mu * max_p_on`` equals ``P_peak`` and the single-connection 99th percentile reaches ``P_max``.

    See :func:`load_year_profile_csv` for ``value_column`` and ``timezone``.
    """
    # Validate cheap arguments before parsing a potentially large file
    derive_component(1.0, simultaneity, max_p_on, mu_ratio, sigma_ratio, component_name)
    profile = load_year_profile_csv(csv_path, value_column=value_column, timezone=timezone)
    return fit_from_profile(profile, simultaneity, max_p_on, mu_ratio, sigma_ratio, component_name)


# --------------------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------------------


def convert_w_profile_to_kw_csv(
    input_path: PathLike,
    output_path: PathLike,
    value_column: ValueColumn = "admd_scaled_w_per_hp",
    expand_to_15min: bool = True,
) -> Path:
    """Convert a profile CSV with power in W to a standardized ``timestamp, consumption_kw`` CSV.

    Args:
        input_path: Source CSV (timestamps in column 0).
        output_path: Target CSV path.
        value_column: Column name or index holding the power in W. A missing column raises.
        expand_to_15min: Repeat each reading of coarser (30/60-minute) data as 15-minute rows.

    Timezone offsets in the source timestamps are preserved in the output.
    """
    out_p = Path(output_path)
    timestamps, watts = _read_timestamp_value_rows(Path(input_path), value_column)
    expansion = _infer_resolution_minutes(timestamps) // INTERVAL_MINUTES if expand_to_15min else 1

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", DEFAULT_VALUE_COLUMN_NAME])
        for ts, w_val in zip(timestamps, watts):
            kw = f"{w_val / 1000.0:.6f}"
            for step in range(expansion):
                sub_ts = ts + timedelta(minutes=INTERVAL_MINUTES * step)
                writer.writerow([sub_ts.isoformat(sep=" "), kw])

    return out_p
