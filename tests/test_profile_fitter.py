"""Unit tests for the profile_fitter module."""

import csv
from pathlib import Path

import numpy as np
import pytest

from conftest import write_profile_csv
from grid_gmm.constants import Z_99
from grid_gmm.profile_fitter import (
    convert_w_profile_to_kw_csv,
    derive_component,
    fit_gm_from_year_profile,
    load_year_profile_csv,
)
from grid_gmm.simulation import MonteCarloEngine


def test_load_year_profile_csv(profile_csv: Path):
    profile = load_year_profile_csv(profile_csv)
    n = 14 * 96
    assert len(profile) == n
    assert profile.months.shape == profile.is_weekend.shape == profile.interval_idx.shape == (n,)
    assert np.all((profile.months >= 1) & (profile.months <= 12))
    assert np.all((profile.interval_idx >= 0) & (profile.interval_idx < 96))
    assert np.all(profile.power_kw >= 0.0)
    assert np.any(profile.is_weekend)
    assert profile.source_unit == "kW"
    assert profile.source_resolution_minutes == 15


def test_watt_profile_is_detected_and_converted(tmp_path: Path):
    kw = load_year_profile_csv(write_profile_csv(tmp_path / "kw.csv"))
    with pytest.warns(UserWarning, match="assuming values are in W"):
        w = load_year_profile_csv(write_profile_csv(tmp_path / "w.csv", scale=1000.0))
    assert w.source_unit == "W"
    np.testing.assert_allclose(w.power_kw, kw.power_kw, atol=1e-3)


def test_peak_at_threshold_stays_kw(tmp_path: Path):
    path = tmp_path / "p.csv"
    path.write_text("timestamp,consumption_kw\n2024-01-01 00:00,20.0\n2024-01-01 00:15,1.0\n")
    assert load_year_profile_csv(path).source_unit == "kW"


@pytest.mark.parametrize("step_minutes", [30, 60])
def test_coarse_profiles_are_expanded_to_15_minutes(tmp_path: Path, step_minutes: int):
    path = write_profile_csv(tmp_path / "coarse.csv", num_days=7, step_minutes=step_minutes)
    profile = load_year_profile_csv(path)
    assert profile.source_resolution_minutes == step_minutes
    assert len(profile) == 7 * 96
    np.testing.assert_array_equal(profile.interval_idx[:4], [0, 1, 2, 3])
    # Energy is preserved by the expansion
    rows = list(csv.reader(path.open()))[1:]
    expected_kwh = sum(float(r[1]) for r in rows) * step_minutes / 60
    assert profile.energy_kwh == pytest.approx(expected_kwh)


def test_unsupported_resolution_raises(tmp_path: Path):
    path = write_profile_csv(tmp_path / "p.csv", num_days=1, step_minutes=20)
    with pytest.raises(ValueError, match="Unsupported profile resolution"):
        load_year_profile_csv(path)


def test_headerless_and_named_value_column(tmp_path: Path):
    headerless = write_profile_csv(tmp_path / "nohdr.csv", num_days=1, header=None)
    assert len(load_year_profile_csv(headerless)) == 96

    multi = tmp_path / "multi.csv"
    multi.write_text("ts,a_kw,b_kw\n2024-01-01 00:00,1.0,2.0\n2024-01-01 00:15,1.5,4.0\n")
    assert load_year_profile_csv(multi).peak_kw == 1.5  # column 1 by default
    assert load_year_profile_csv(multi, value_column="b_kw").peak_kw == 4.0
    assert load_year_profile_csv(multi, value_column=2).peak_kw == 4.0
    with pytest.raises(ValueError, match="not found in CSV header"):
        load_year_profile_csv(multi, value_column="missing")


def test_negative_values_are_clipped_with_single_warning(tmp_path: Path):
    path = tmp_path / "neg.csv"
    path.write_text("timestamp,kw\n2024-01-01 00:00,-1\n2024-01-01 00:15,-2\n2024-01-01 00:30,3\n")
    with pytest.warns(UserWarning, match="2 negative power value"):
        profile = load_year_profile_csv(path)
    np.testing.assert_array_equal(profile.power_kw, [0.0, 0.0, 3.0])


def test_timezone_conversion(tmp_path: Path):
    path = tmp_path / "utc.csv"
    path.write_text("timestamp,kw\n2024-07-01 18:00:00+00:00,1\n2024-07-01 18:15:00+00:00,2\n")
    assert load_year_profile_csv(path).interval_idx.tolist() == [72, 73]  # wall clock as written
    berlin = load_year_profile_csv(path, timezone="Europe/Berlin")  # UTC+2 in summer
    assert berlin.interval_idx.tolist() == [80, 81]


def test_unparseable_row_reports_line_number(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("timestamp,kw\n2024-01-01 00:00,1\nnot-a-date,2\n")
    with pytest.raises(ValueError, match="row 3"):
        load_year_profile_csv(path)


def test_fit_gm_from_year_profile(profile_csv: Path):
    simultaneity = 0.35
    component, factors, summary = fit_gm_from_year_profile(profile_csv, simultaneity=simultaneity, max_p_on=0.90)

    assert summary.p_peak > 0
    assert component.p_max == pytest.approx(summary.p_peak / simultaneity)
    assert component.mu_ratio == pytest.approx(simultaneity / 0.90)
    assert component.mu == pytest.approx(summary.p_peak / 0.90)
    assert summary.peak_p_on == pytest.approx(0.90)

    assert np.all(factors.month_factors >= 0)
    cp = factors.find_critical_point()
    assert cp.p_on == pytest.approx(0.90)
    assert component.mu * cp.p_on == pytest.approx(summary.p_peak)

    comp1, _, summ1 = fit_gm_from_year_profile(profile_csv, simultaneity=simultaneity)
    assert summ1.peak_p_on == pytest.approx(1.0)
    assert comp1.mu_ratio == pytest.approx(simultaneity)
    assert comp1.mu + Z_99 * comp1.sigma == pytest.approx(comp1.p_max)


def test_fit_from_watt_profile_matches_kw_profile(tmp_path: Path):
    kw_comp, _, _ = fit_gm_from_year_profile(write_profile_csv(tmp_path / "kw.csv"), simultaneity=0.5)
    with pytest.warns(UserWarning):
        w_comp, _, w_summary = fit_gm_from_year_profile(
            write_profile_csv(tmp_path / "w.csv", scale=1000.0), simultaneity=0.5
        )
    assert w_summary.source_unit == "W"
    assert w_comp.p_max == pytest.approx(kw_comp.p_max, rel=1e-3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"simultaneity": 0.0}, "Simultaneity"),
        ({"simultaneity": 0.5, "max_p_on": 1.5}, "max_p_on"),
        ({"simultaneity": 0.9, "max_p_on": 0.5}, "simultaneity <= max_p_on"),
    ],
)
def test_derive_component_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        derive_component(p_peak=1.0, **kwargs)


def test_convert_w_profile_to_kw_csv(tmp_path: Path):
    src_csv = tmp_path / "src_watts.csv"
    src_csv.write_text(
        "datetime_utc,archetype_w,admd_scaled_w_per_hp\n"
        "2000-01-01 00:00:00+00:00,500.0,750.0\n"
        "2000-01-01 01:00:00+00:00,600.0,900.0\n"
    )
    out_csv = tmp_path / "out_kw.csv"
    assert convert_w_profile_to_kw_csv(src_csv, out_csv, value_column="admd_scaled_w_per_hp").exists()

    with open(out_csv) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["timestamp", "consumption_kw"]
    assert len(rows) == 1 + 8  # 2 hours -> 8 quarter hours
    assert float(rows[1][1]) == pytest.approx(0.75)
    assert rows[2][0] == "2000-01-01 00:15:00+00:00"  # timezone offset preserved

    with pytest.raises(ValueError, match="not found"):
        convert_w_profile_to_kw_csv(src_csv, out_csv, value_column="nope")


def test_fit_gm_end_to_end_simulation(profile_csv: Path):
    component, factors, _ = fit_gm_from_year_profile(profile_csv, simultaneity=0.7, mu_ratio=0.5, sigma_ratio=0.2)
    results = MonteCarloEngine(component, factors, seed=123).run_critical_time_simulation(
        connection_counts=[1, 5, 20], num_runs=50
    )
    assert results.component.mu_ratio == 0.5
    assert results.component.sigma_ratio == 0.2
    assert results.p_per_conn[50.0][0] <= results.p_per_conn[95.0][0] <= results.p_per_conn[99.0][0]
