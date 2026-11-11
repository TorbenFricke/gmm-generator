"""Monte Carlo simulation engine for grid connection loads at the critical point in time."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from grid_gmm.factors import CriticalPoint, FactorProfiles
from grid_gmm.model import GaussianComponent

DEFAULT_CONNECTIONS = (1, 2, 5, 10, 15, 20, 30, 50, 100, 200)
DEFAULT_NUM_RUNS = 200
DEFAULT_PERCENTILES = (50.0, 95.0, 99.0)
DEFAULT_SEED = 42


@dataclass(frozen=True, eq=False)
class SimulationResults:
    """Aggregated results of a Monte Carlo simulation.

    ``p_per_conn`` and ``p_total`` map each percentile to an array aligned with ``connections``.
    ``raw_totals`` maps each connection count to the ``(num_runs,)`` array of simulated total power.
    """

    component: GaussianComponent
    critical_point: CriticalPoint
    connections: list[int]
    num_runs: int
    percentiles: list[float]
    p_per_conn: dict[float, np.ndarray]
    p_total: dict[float, np.ndarray]
    raw_totals: dict[int, np.ndarray]

    @property
    def expected_load_per_conn(self) -> float:
        """Expected load per connection (kW) as N -> inf: E[active power] * p(on)."""
        return self.component.expected_load(self.critical_point.p_on)


class MonteCarloEngine:
    """Runs Monte Carlo simulations of N simultaneously connected households."""

    def __init__(
        self,
        component: GaussianComponent,
        factors: FactorProfiles,
        seed: int | None = None,
    ) -> None:
        self.component = component
        self.factors = factors
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def simulate_total_power(self, n_conn: int, p_on: float, num_runs: int) -> np.ndarray:
        """Simulate the total power (kW) of ``n_conn`` connections for ``num_runs`` independent runs."""
        # Number of active connections per run ~ Binomial(n_conn, p_on)
        active_counts = self.rng.binomial(n_conn, p_on, size=num_runs)
        # Draw all active loads at once, then sum them per run
        loads = self.component.sample_active_power(size=int(active_counts.sum()), rng=self.rng)
        run_ids = np.repeat(np.arange(num_runs), active_counts)
        return np.bincount(run_ids, weights=loads, minlength=num_runs).astype(float)

    def run_critical_time_simulation(
        self,
        connection_counts: Sequence[int] = DEFAULT_CONNECTIONS,
        num_runs: int = DEFAULT_NUM_RUNS,
        percentiles: Sequence[float] = DEFAULT_PERCENTILES,
    ) -> SimulationResults:
        """Find the critical point in time and simulate each connection count at that point.

        Args:
            connection_counts: Connection numbers N to simulate (each >= 1).
            num_runs: Monte Carlo runs per connection count (>= 1).
            percentiles: Percentiles to evaluate, each in [0, 100].
        """
        connections = [int(n) for n in connection_counts]
        pct_list = [float(p) for p in percentiles]
        if num_runs < 1:
            raise ValueError(f"num_runs must be >= 1, got {num_runs}")
        if not connections or min(connections) < 1:
            raise ValueError(f"connection counts must be >= 1, got {connections}")
        if not pct_list or not all(0.0 <= p <= 100.0 for p in pct_list):
            raise ValueError(f"percentiles must be in [0, 100], got {pct_list}")

        critical_point = self.factors.find_critical_point()

        raw_totals = {n: self.simulate_total_power(n, critical_point.p_on, num_runs) for n in connections}
        p_total = {pct: np.array([np.percentile(raw_totals[n], pct) for n in connections]) for pct in pct_list}
        p_per_conn = {pct: np.array([np.percentile(raw_totals[n] / n, pct) for n in connections]) for pct in pct_list}

        return SimulationResults(
            component=self.component,
            critical_point=critical_point,
            connections=connections,
            num_runs=num_runs,
            percentiles=pct_list,
            p_per_conn=p_per_conn,
            p_total=p_total,
            raw_totals=raw_totals,
        )
