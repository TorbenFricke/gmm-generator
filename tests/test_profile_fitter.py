"""Unit tests for profile_fitter module."""

import csv
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pytest

from grid_gmm.profile_fitter import (
    fit_gm_from_year_profile,
    load_year_profile_csv,
)
from grid_gmm.simulation import MonteCarloEngine


def _create_synthetic_csv(path: Path, num_days: int = 14) -> None:
    """Helper to generate a realistic two-column CSV for testing."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    rows = [["timestamp", "consumption_kw"]]

    # Generate 15-min intervals for num_days
    total_steps = num_days * 96
    for step in range(total_steps):
        dt = start_time + timedelta(minutes=15 * step)
        # Power base pattern: evening peak around 19:00 (step % 96 == 76)
        interval_idx = step % 96
        base = 0.5
        if 28 <= interval_idx <= 36:  # 07:00 - 09:00
            base = 1.8
        elif 72 <= interval_idx <= 84:  # 18:00 - 21:00
            base = 3.5  # peak
        # Add small deterministic variation
        power = base + 0.1 * np.sin(step)
        rows.append([dt.strftime("%Y-%m-%d %H:%M:%S"), f"{power:.3f}"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def test_load_year_profile_csv(tmp_path: Path):
    csv_file = tmp_path / "test_profile.csv"
    _create_synthetic_csv(csv_file, num_days=7)

    months, is_weekends, intervals, consumptions = load_year_profile_csv(csv_file)

    assert len(months) == 7 * 96
    assert len(is_weekends) == 7 * 96
    assert len(intervals) == 7 * 96
    assert len(consumptions) == 7 * 96

    assert np.all((months >= 1) & (months <= 12))
    assert np.all((intervals >= 0) & (intervals < 96))
    assert np.all(consumptions >= 0.0)
    assert np.any(is_weekends)  # 7 days includes Saturday and Sunday


def test_fit_gm_from_year_profile(tmp_path: Path):
    csv_file = tmp_path / "test_profile.csv"
    _create_synthetic_csv(csv_file, num_days=14)

    simultaneity = 0.35
    mu_ratio = 0.5
    sigma_ratio = 0.2

    # Test with max_p_on=0.90
    component, factors, summary = fit_gm_from_year_profile(
        csv_path=csv_file,
        simultaneity=simultaneity,
        max_p_on=0.90,
    )

    # 1. P_max = P_peak / simultaneity
    assert summary.p_peak > 0
    assert component.p_max == pytest.approx(summary.p_peak / simultaneity)
    assert component.mu_ratio == pytest.approx(simultaneity / 0.90, rel=1e-3)
    assert component.mu == pytest.approx(summary.p_peak / 0.90, rel=1e-3)

    # 2. Peak activation probability is 0.90 at peak
    assert summary.peak_p_on == pytest.approx(0.90)

    # 3. Factor dimensions
    assert len(factors.month_factors) == 12
    assert len(factors.weekday_factors) == 96
    assert len(factors.weekend_factors) == 96
    assert np.all(factors.month_factors >= 0)
    assert np.all(factors.weekday_factors >= 0)
    assert np.all(factors.weekend_factors >= 0)

    # 4. Critical point should match peak p(on) = 0.90
    cp = factors.find_critical_point()
    assert cp.p_on == pytest.approx(0.90, rel=1e-2)

    # 5. Fleet expectation as N -> inf matches P_peak
    assert (component.mu * cp.p_on) == pytest.approx(summary.p_peak, rel=1e-2)

    # Test with default max_p_on=1.0
    comp1, fac1, summ1 = fit_gm_from_year_profile(
        csv_path=csv_file,
        simultaneity=simultaneity,
    )
    assert summ1.peak_p_on == pytest.approx(1.0)
    assert comp1.mu_ratio == pytest.approx(simultaneity, rel=1e-3)
    # 99th percentile upper reach: mu + 2.32635 * sigma should equal P_max
    p99_single = comp1.mu + 2.32635 * comp1.sigma
    assert p99_single == pytest.approx(comp1.p_max, rel=1e-2)


def test_convert_w_profile_to_kw_csv(tmp_path: Path):
    from grid_gmm.profile_fitter import convert_w_profile_to_kw_csv

    # Create dummy source with Watts
    src_csv = tmp_path / "src_watts.csv"
    with open(src_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime_utc", "archetype_w", "admd_scaled_w_per_hp"])
        writer.writerow(["2000-01-01 00:00:00+00:00", "500.0", "750.0"])
        writer.writerow(["2000-01-01 01:00:00+00:00", "600.0", "900.0"])

    out_csv = tmp_path / "out_kw.csv"
    res_path = convert_w_profile_to_kw_csv(src_csv, out_csv, value_column="admd_scaled_w_per_hp", expand_to_15min=True)
    assert res_path.exists()

    with open(out_csv, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["timestamp", "consumption_kw"]
        rows = list(reader)
        # 2 hours expanded to 15-min -> 8 rows
        assert len(rows) == 8
        # 750 W -> 0.75 kW
        assert float(rows[0][1]) == pytest.approx(0.75)


def test_fit_gm_end_to_end_simulation(tmp_path: Path):
    csv_file = tmp_path / "test_profile.csv"
    _create_synthetic_csv(csv_file, num_days=14)

    component, factors, _ = fit_gm_from_year_profile(
        csv_path=csv_file,
        simultaneity=0.7,
        mu_ratio=0.5,
        sigma_ratio=0.2,
    )

    engine = MonteCarloEngine(component=component, factors=factors, seed=123)
    results = engine.run_critical_time_simulation(
        connection_counts=[1, 5, 20],
        num_runs=50,
        percentiles=[50.0, 95.0, 99.0],
    )

    assert results.component.mu_ratio == 0.5
    assert results.component.sigma_ratio == 0.2
    assert len(results.connections) == 3
    # Check percentile monotonicity
    assert results.p_per_conn[50.0][0] <= results.p_per_conn[95.0][0] <= results.p_per_conn[99.0][0]
