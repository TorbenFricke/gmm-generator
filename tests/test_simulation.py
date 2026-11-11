"""Unit tests for the Monte Carlo engine, reporting and plotting."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from grid_gmm.example_data import create_example_component, create_example_profiles
from grid_gmm.factors import FactorProfiles
from grid_gmm.model import GaussianComponent
from grid_gmm.plotting import plot_percentiles_per_connection
from grid_gmm.reporting import format_summary_table, metadata_path_for, save_results_to_csv
from grid_gmm.simulation import MonteCarloEngine


def _run(seed=42, connections=(1, 2, 5, 10, 20), runs=100, percentiles=(50.0, 95.0, 99.0)):
    engine = MonteCarloEngine(create_example_component(), create_example_profiles(), seed=seed)
    return engine.run_critical_time_simulation(connections, num_runs=runs, percentiles=percentiles)


def test_monte_carlo_engine_execution():
    results = _run()
    assert results.connections == [1, 2, 5, 10, 20]
    assert results.num_runs == 100
    p50, p95, p99 = (results.p_per_conn[p] for p in (50.0, 95.0, 99.0))
    assert len(p50) == 5
    assert np.all(p50 <= p95)
    assert np.all(p95 <= p99)
    assert np.all(p99 <= results.component.p_max + 1e-6)
    for n in results.connections:
        assert results.raw_totals[n].shape == (100,)


def test_seed_reproducibility():
    a, b, c = _run(seed=1), _run(seed=1), _run(seed=2)
    for n in a.connections:
        np.testing.assert_array_equal(a.raw_totals[n], b.raw_totals[n])
    assert not np.array_equal(a.raw_totals[20], c.raw_totals[20])


def test_large_n_converges_to_truncated_mean_times_p_on():
    # Strongly asymmetric truncation: mu * p_on would overestimate the expectation by ~25 %.
    component = GaussianComponent(p_max=10.0, mu_ratio=0.9, sigma_ratio=0.3)
    factors = FactorProfiles([1.0] * 12, [0.5] * 96, [0.5] * 96)
    results = MonteCarloEngine(component, factors, seed=0).run_critical_time_simulation([5000], num_runs=50)

    assert results.expected_load_per_conn == pytest.approx(component.mean * 0.5)
    assert results.p_per_conn[50.0][0] == pytest.approx(results.expected_load_per_conn, rel=0.01)
    assert abs(results.p_per_conn[50.0][0] - component.mu * 0.5) > 0.5


def test_bincount_aggregation_matches_manual_sum():
    engine = MonteCarloEngine(create_example_component(), create_example_profiles(), seed=3)
    totals = engine.simulate_total_power(n_conn=10, p_on=0.5, num_runs=20)

    replay = np.random.default_rng(3)
    active = replay.binomial(10, 0.5, size=20)
    loads = create_example_component().sample_active_power(int(active.sum()), rng=replay)
    expected = [chunk.sum() for chunk in np.split(loads, np.cumsum(active)[:-1])]
    np.testing.assert_allclose(totals, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_runs": 0}, "num_runs"),
        ({"connection_counts": [0, 5]}, "connection counts"),
        ({"percentiles": [101.0]}, "percentiles"),
    ],
)
def test_simulation_argument_validation(kwargs, message):
    engine = MonteCarloEngine(create_example_component(), create_example_profiles(), seed=0)
    with pytest.raises(ValueError, match=message):
        engine.run_critical_time_simulation(**kwargs)


def test_csv_and_metadata_export(tmp_path: Path):
    results = _run()
    csv_file = tmp_path / "out.csv"
    meta_file = save_results_to_csv(results, csv_file)

    assert meta_file == metadata_path_for(csv_file) == tmp_path / "out.meta.json"
    with open(csv_file) as f:
        rows = list(csv.DictReader(f))  # plain CSV: no comment rows
    assert len(rows) == 5
    assert rows[0]["connections"] == "1"
    assert float(rows[-1]["p_per_conn_p99_kw"]) == pytest.approx(results.p_per_conn[99.0][-1], abs=1e-3)

    meta = json.loads(meta_file.read_text())
    assert meta["critical_point"]["month_name"] == "December"
    assert meta["critical_point"]["day_type"] == "weekday"
    assert meta["expected_load_per_conn_kw"] == pytest.approx(results.expected_load_per_conn)


def test_summary_table_supports_arbitrary_percentiles():
    table = format_summary_table(_run(percentiles=(10.0, 90.0)))
    assert "P_tot p10" in table and "kW/conn p90" in table
    assert "p99" not in table


def test_plot_generation(tmp_path: Path):
    results = _run()
    plot_file = tmp_path / "plot.png"
    fig = plot_percentiles_per_connection(results, output_path=plot_file, dpi=50)
    assert plot_file.stat().st_size > 0
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert any("P_max" in label for label in labels)
    assert any("Expected (N→∞)" in label for label in labels)
