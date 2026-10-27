"""Unit tests for Monte Carlo simulation engine and outputs."""

from pathlib import Path
import numpy as np
import pytest

from grid_gmm.example_data import create_example_component, create_example_profiles
from grid_gmm.simulation import MonteCarloEngine
from grid_gmm.cli import save_results_to_csv
from grid_gmm.plotting import plot_percentiles_per_connection


def test_monte_carlo_engine_execution(tmp_path: Path):
    component = create_example_component()
    profiles = create_example_profiles()
    connections = [1, 2, 5, 10, 20]
    runs = 100

    engine = MonteCarloEngine(component=component, factors=profiles, seed=42)
    results = engine.run_critical_time_simulation(
        connection_counts=connections,
        num_runs=runs,
        percentiles=[50.0, 95.0, 99.0],
    )

    assert results.connections == connections
    assert results.num_runs == runs
    assert results.critical_point is not None

    # Check percentile order: p50 <= p95 <= p99 for each connection count
    p50 = results.p_per_conn[50.0]
    p95 = results.p_per_conn[95.0]
    p99 = results.p_per_conn[99.0]

    assert len(p50) == len(connections)
    assert np.all(p50 <= p95)
    assert np.all(p95 <= p99)

    # Power per connection must not exceed p_max
    assert np.all(p99 <= component.p_max + 1e-6)

    # Test CSV export
    csv_file = tmp_path / "test_output.csv"
    save_results_to_csv(results, csv_file)
    assert csv_file.exists()
    content = csv_file.read_text(encoding="utf-8")
    assert "Critical Month" in content
    assert "connections,p_max_kw" in content

    # Test Plot generation
    plot_file = tmp_path / "test_plot.png"
    fig = plot_percentiles_per_connection(results, output_path=plot_file)
    assert plot_file.exists()
    assert plot_file.stat().st_size > 0

    # Verify horizontal lines are present
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert any("P_max" in l for l in labels)
    assert any("Expected (N→∞)" in l for l in labels)

