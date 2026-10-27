"""Monte Carlo simulation engine for evaluating grid connection loads at the critical point in time."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union
import numpy as np

from grid_gmm.factors import CriticalPoint, FactorProfiles
from grid_gmm.model import GaussianComponent


DEFAULT_CONNECTIONS = (1, 2, 5, 10, 15, 20, 30, 50, 100, 200)
DEFAULT_NUM_RUNS = 200
DEFAULT_PERCENTILES = (50.0, 95.0, 99.0)


@dataclass
class SimulationResults:
    """Stores the aggregated results from the Monte Carlo simulation."""

    component: GaussianComponent
    critical_point: CriticalPoint
    connections: List[int]
    num_runs: int
    percentiles: List[float]
    # Arrays aligned with connections
    p_per_conn: Dict[float, np.ndarray]  # percentile -> array of power per connection (kW/conn)
    p_total: Dict[float, np.ndarray]  # percentile -> array of total power (kW)
    # Full raw simulation data: connections -> (num_runs,) total power array
    raw_totals: Dict[int, np.ndarray]

    def to_csv_rows(self) -> List[List[Union[int, float]]]:
        """Convert results to rows suitable for CSV export."""
        rows = []
        for i, n in enumerate(self.connections):
            row = [n]
            # Add total power percentiles
            for pct in self.percentiles:
                row.append(float(self.p_total[pct][i]))
            # Add per-connection power percentiles
            for pct in self.percentiles:
                row.append(float(self.p_per_conn[pct][i]))
            row.append(self.component.p_max)
            rows.append(row)
        return rows


class MonteCarloEngine:
    """Engine for running Monte Carlo simulations of grid connections."""

    def __init__(
        self,
        component: GaussianComponent,
        factors: FactorProfiles,
        seed: Optional[int] = None,
    ):
        self.component = component
        self.factors = factors
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def run_critical_time_simulation(
        self,
        connection_counts: Sequence[int] = DEFAULT_CONNECTIONS,
        num_runs: int = DEFAULT_NUM_RUNS,
        percentiles: Sequence[float] = DEFAULT_PERCENTILES,
    ) -> SimulationResults:
        """Calculate the critical point in time and run Monte Carlo simulations for each connection count.

        Args:
            connection_counts: Sequence of connection numbers to simulate.
            num_runs: Number of Monte Carlo simulation runs per connection count (e.g. 200).
            percentiles: Percentiles to evaluate (e.g. [50.0, 95.0, 99.0]).

        Returns:
            SimulationResults dataclass with percentile values and metadata.
        """
        # Step 1: Pre-calculate the most critical point in time
        critical_point = self.factors.find_critical_point()
        p_on = critical_point.p_on

        pct_list = [float(p) for p in percentiles]
        p_per_conn: Dict[float, List[float]] = {p: [] for p in pct_list}
        p_total: Dict[float, List[float]] = {p: [] for p in pct_list}
        raw_totals: Dict[int, np.ndarray] = {}

        for n_conn in connection_counts:
            # Step 2: For this connection count, run num_runs simulations
            # Number of active households across num_runs: Binomial(n_conn, p_on)
            active_counts = self.rng.binomial(n_conn, p_on, size=num_runs)

            # Sample powers for each simulation run
            total_powers = np.zeros(num_runs, dtype=float)
            total_active_all_runs = int(np.sum(active_counts))

            if total_active_all_runs > 0:
                # Sample all active loads at once for vectorized efficiency
                all_sampled_loads = self.component.sample_active_power(
                    size=total_active_all_runs,
                    rng=self.rng,
                )

                # Assign loads to each simulation run
                offset = 0
                for r in range(num_runs):
                    k = active_counts[r]
                    if k > 0:
                        total_powers[r] = np.sum(all_sampled_loads[offset : offset + k])
                        offset += k

            raw_totals[n_conn] = total_powers
            power_per_conn = total_powers / n_conn

            # Step 3: Compute percentiles
            for pct in pct_list:
                tot_val = float(np.percentile(total_powers, pct))
                per_conn_val = float(np.percentile(power_per_conn, pct))
                p_total[pct].append(tot_val)
                p_per_conn[pct].append(per_conn_val)

        return SimulationResults(
            component=self.component,
            critical_point=critical_point,
            connections=list(connection_counts),
            num_runs=num_runs,
            percentiles=pct_list,
            p_per_conn={p: np.array(vals) for p, vals in p_per_conn.items()},
            p_total={p: np.array(vals) for p, vals in p_total.items()},
            raw_totals=raw_totals,
        )
