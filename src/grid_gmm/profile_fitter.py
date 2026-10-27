"""Module for extracting Gaussian Model parameters and factor profiles from an annual load profile."""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union
import warnings
import numpy as np

try:
    from dateutil import parser as date_parser
except ImportError:
    date_parser = None

from grid_gmm.factors import FactorProfiles
from grid_gmm.model import GaussianComponent


@dataclass
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


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse timestamp string using ISO format or dateutil parser."""
    ts_str = ts_str.strip()
    # Try ISO8601 first (fastest)
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        pass

    # Common formats
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
    ):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            pass

    if date_parser is not None:
        return date_parser.parse(ts_str)

    raise ValueError(f"Unable to parse timestamp: '{ts_str}'")


def load_year_profile_csv(
    filepath: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a two-column CSV containing timestamp and consumption in kW.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Tuple of (months, is_weekends, interval_indices, consumptions) as NumPy arrays:
        - months: 1 to 12
        - is_weekends: boolean (True for Saturday/Sunday)
        - interval_indices: 0 to 95 (15-minute interval index of day)
        - consumptions: float power in kW
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Profile file not found: {path}")

    months: List[int] = []
    is_weekends: List[bool] = []
    interval_indices: List[int] = []
    consumptions: List[float] = []

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"CSV file is empty: {path}")

        # Check if first row is header
        start_row = 0
        try:
            float(first_row[1])
            has_header = False
            rows = [first_row]
        except (ValueError, IndexError):
            has_header = True
            rows = []

        # Read remaining rows
        for row in reader:
            if row and len(row) >= 2:
                rows.append(row)

    if not rows:
        raise ValueError(f"No valid data rows found in {path}")

    for row_idx, row in enumerate(rows, start=2 if has_header else 1):
        if len(row) < 2:
            continue
        ts_str, val_str = row[0].strip(), row[1].strip()
        if not ts_str or not val_str:
            continue

        try:
            dt = _parse_timestamp(ts_str)
            val = float(val_str)
        except Exception as e:
            raise ValueError(f"Error parsing row {row_idx} ({row}): {e}") from e

        if val < 0:
            warnings.warn(f"Negative power value ({val} kW) encountered at row {row_idx}, clipping to 0.0.")
            val = 0.0

        interval_idx = dt.hour * 4 + (dt.minute // 15)
        is_weekend = dt.weekday() >= 5  # 5=Saturday, 6=Sunday

        months.append(dt.month)
        is_weekends.append(is_weekend)
        interval_indices.append(interval_idx)
        consumptions.append(val)

    if len(consumptions) == 0:
        raise ValueError(f"No valid readings parsed from {path}")

    intervals_arr = np.array(interval_indices, dtype=int)
    # Check if data is hourly (all intervals are multiples of 4, e.g. 0, 4, 8, ...)
    if len(intervals_arr) > 10 and np.all(intervals_arr % 4 == 0):
        # Auto-expand each hourly reading into four 15-minute intervals
        exp_months = []
        exp_weekends = []
        exp_intervals = []
        exp_consumptions = []
        for m, w, idx, c in zip(months, is_weekends, intervals_arr, consumptions):
            for sub_i in range(4):
                exp_months.append(m)
                exp_weekends.append(w)
                exp_intervals.append(idx + sub_i)
                exp_consumptions.append(c)
        return (
            np.array(exp_months, dtype=int),
            np.array(exp_weekends, dtype=bool),
            np.array(exp_intervals, dtype=int),
            np.array(exp_consumptions, dtype=float),
        )

    return (
        np.array(months, dtype=int),
        np.array(is_weekends, dtype=bool),
        intervals_arr,
        np.array(consumptions, dtype=float),
    )


def fit_gm_from_year_profile(
    csv_path: Union[str, Path],
    simultaneity: float,
    max_p_on: float = 1.0,
    mu_ratio: Optional[float] = None,
    sigma_ratio: Optional[float] = None,
    component_name: str = "Fitted Household Connection",
) -> Tuple[GaussianComponent, FactorProfiles, ProfileFitSummary]:
    """Derive Gaussian Model parameters and factor profiles from an annual load profile matching Rusck's simultaneity model.

    In this formulation:
    - At the profile's peak demand P_peak, activation probability is max_p_on (defaults to 1.0 = 100%):
        p(on)_peak = max_p_on
    - For all time steps t:
        p(on, t) = max_p_on * (P(t) / P_peak)
    - Connection capacity matches Rusck's definition (P_peak = s * P_max):
        P_max = P_peak / s
    - Active load parameters match the desired simultaneity:
        mu_ratio = s / max_p_on  (so mu * max_p_on = P_peak)
        sigma_ratio = (1.0 - mu_ratio) / 2.32635  (matches Rusck's 99th percentile curve: mu + 2.33*sigma = P_max)
    - Fleet average load as N -> infinity is:
        E[P_peak] = p(on)_peak * mu = max_p_on * (P_peak / max_p_on) = P_peak
    - Single connection peak at N = 1 (99th percentile) is:
        P_1 = mu + 2.326 * sigma = s * P_max + (1 - s) * P_max = P_max

    Args:
        csv_path: Path to two-column CSV (timestamp, consumption_kw).
        simultaneity: Target simultaneity factor s in (0, 1].
        max_p_on: Maximum activation probability p(on) at peak demand (default: 1.0).
        mu_ratio: Median active power as ratio of P_max (defaults to s / max_p_on).
        sigma_ratio: Standard deviation as ratio of P_max (defaults to (1 - mu_ratio) / 2.32635).
        component_name: Identifier for the created GaussianComponent.

    Returns:
        Tuple of (GaussianComponent, FactorProfiles, ProfileFitSummary).
    """
    if not (0.0 < simultaneity <= 1.0):
        raise ValueError(f"Simultaneity must be in (0, 1], got {simultaneity}")
    if not (0.0 < max_p_on <= 1.0):
        raise ValueError(f"max_p_on must be in (0, 1], got {max_p_on}")

    months, is_weekends, interval_indices, consumptions = load_year_profile_csv(csv_path)

    p_peak = float(np.max(consumptions))
    if p_peak <= 0:
        raise ValueError("Maximum power consumption in year profile must be strictly positive")

    # 1. Connection capacity from Rusck: P_max = P_peak / simultaneity
    p_max = float(p_peak / simultaneity)

    # 2. Derive mu_ratio and sigma_ratio matching simultaneity and max_p_on
    if mu_ratio is None:
        eff_mu_ratio = round(float(simultaneity / max_p_on), 5)  # mu * max_p_on = p_peak
    else:
        eff_mu_ratio = float(mu_ratio)

    if sigma_ratio is None:
        # Match Rusck's 99th percentile spread (z_0.99 = 2.32635): mu + z * sigma = P_max
        margin = max(0.0, 1.0 - eff_mu_ratio)
        eff_sigma_ratio = round(float(margin / 2.32635), 4)
    else:
        eff_sigma_ratio = float(sigma_ratio)

    if not (0.0 < eff_mu_ratio <= 1.0):
        raise ValueError(f"mu_ratio must be in (0, 1], got {eff_mu_ratio}. Check that simultaneity <= max_p_on.")
    if eff_sigma_ratio < 0.0:
        raise ValueError(f"sigma_ratio must be non-negative, got {eff_sigma_ratio}")

    # 3. Peak activation probability is fixed to max_p_on at peak demand
    peak_p_on = float(max_p_on)

    # Continuous activation probability series: p(on, t) = max_p_on * (P(t) / P_peak)
    p_on_series = np.clip(peak_p_on * (consumptions / p_peak), 0.0, 1.0)
    mean_p_on_overall = float(np.mean(p_on_series))
    if mean_p_on_overall == 0:
        mean_p_on_overall = 1e-6

    # 1. Month factors: mean p(on) in month m relative to annual mean
    month_factors = np.ones(12, dtype=float)
    for m in range(1, 13):
        mask_m = months == m
        if np.any(mask_m):
            month_factors[m - 1] = float(np.mean(p_on_series[mask_m])) / mean_p_on_overall
        else:
            month_factors[m - 1] = 1.0

    # 2. Interval factors for weekday and weekend
    weekday_factors_raw = np.zeros(96, dtype=float)
    weekend_factors_raw = np.zeros(96, dtype=float)

    mask_weekday = ~is_weekends
    mask_weekend = is_weekends

    for idx in range(96):
        # Weekday
        mask_w_idx = mask_weekday & (interval_indices == idx)
        if np.any(mask_w_idx):
            weekday_factors_raw[idx] = float(np.mean(p_on_series[mask_w_idx])) / mean_p_on_overall
        else:
            weekday_factors_raw[idx] = 1.0

        # Weekend
        mask_h_idx = mask_weekend & (interval_indices == idx)
        if np.any(mask_h_idx):
            weekend_factors_raw[idx] = float(np.mean(p_on_series[mask_h_idx])) / mean_p_on_overall
        else:
            weekend_factors_raw[idx] = 1.0

    # 3. Scale factor matrices so max(f_month * f_interval) == peak_p_on
    p_weekday_outer = np.outer(month_factors, weekday_factors_raw)
    p_weekend_outer = np.outer(month_factors, weekend_factors_raw)
    raw_max_prod = max(float(np.max(p_weekday_outer)), float(np.max(p_weekend_outer)))

    if raw_max_prod > 0:
        scale = peak_p_on / raw_max_prod
    else:
        scale = 1.0

    weekday_factors = np.round(weekday_factors_raw * scale, 5)
    weekend_factors = np.round(weekend_factors_raw * scale, 5)
    month_factors = np.round(month_factors, 5)

    # 4. Construct objects
    component = GaussianComponent(
        p_max=p_max,
        mu_ratio=eff_mu_ratio,
        sigma_ratio=eff_sigma_ratio,
        name=component_name,
    )

    factor_profiles = FactorProfiles(
        month_factors=month_factors,
        weekday_factors=weekday_factors,
        weekend_factors=weekend_factors,
    )

    # Annual energy assuming 15-minute readings: power * 0.25 h
    annual_kwh = float(np.sum(consumptions) * 0.25)

    summary = ProfileFitSummary(
        p_peak=p_peak,
        simultaneity=simultaneity,
        p_max=p_max,
        mu_ratio=eff_mu_ratio,
        sigma_ratio=eff_sigma_ratio,
        peak_p_on=peak_p_on,
        annual_energy_kwh=annual_kwh,
        total_records=len(consumptions),
    )

    return component, factor_profiles, summary


def convert_w_profile_to_kw_csv(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    value_column: Optional[str] = "admd_scaled_w_per_hp",
    expand_to_15min: bool = True,
) -> Path:
    """Convert a profile CSV with power in Watts to a standardized 2-column CSV (timestamp, consumption_kw).

    Args:
        input_path: Path to source CSV.
        output_path: Target path for the converted CSV.
        value_column: Column name in source CSV (e.g. 'admd_scaled_w_per_hp' or 'archetype_w').
                      Defaults to 'admd_scaled_w_per_hp'.
        expand_to_15min: If True, expands hourly data to 15-minute intervals.

    Returns:
        Path to the saved CSV.
    """
    in_p = Path(input_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with open(in_p, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Input file is empty: {in_p}")

        col_idx = 1
        if value_column and value_column in header:
            col_idx = header.index(value_column)
        elif len(header) >= 3:
            # Default to column 2 (admd_scaled_w_per_hp) if present
            col_idx = 2

        out_rows = [["timestamp", "consumption_kw"]]
        for row in reader:
            if len(row) <= col_idx:
                continue
            ts_str = row[0].strip()
            w_val = float(row[col_idx].strip())
            kw_val = round(w_val / 1000.0, 6)

            dt = _parse_timestamp(ts_str)

            if expand_to_15min:
                for sub_i in range(4):
                    sub_dt = dt + timedelta(minutes=15 * sub_i)
                    out_rows.append([sub_dt.strftime("%Y-%m-%d %H:%M:%S"), f"{kw_val:.6f}"])
            else:
                out_rows.append([dt.strftime("%Y-%m-%d %H:%M:%S"), f"{kw_val:.6f}"])

    with open(out_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)

    return out_p
