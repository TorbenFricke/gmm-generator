"""Command-line interface for running grid usage Gaussian simulation."""

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

from grid_gmm.example_data import (
    create_example_component,
    create_example_profiles,
    export_example_config,
    load_config_from_json,
)
from grid_gmm.model import GaussianComponent
from grid_gmm.plotting import plot_percentiles_per_connection
from grid_gmm.simulation import (
    DEFAULT_CONNECTIONS,
    DEFAULT_NUM_RUNS,
    DEFAULT_PERCENTILES,
    MonteCarloEngine,
    SimulationResults,
)


def save_results_to_csv(results: SimulationResults, filepath: Path) -> None:
    """Save simulation results to a CSV file with metadata header."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    cp = results.critical_point

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Metadata comments
        writer.writerow(["# Simulation Metadata"])
        writer.writerow(["# Critical Month", cp.month_name, f"(Month {cp.month})"])
        writer.writerow(["# Critical Day Type", cp.day_type])
        writer.writerow(["# Critical Interval", cp.time_str, f"(Index {cp.interval_idx})"])
        writer.writerow(["# Peak p(on)", f"{cp.p_on:.4f}"])
        writer.writerow(["# P_max (kW)", results.component.p_max])
        writer.writerow(["# mu (kW)", f"{results.component.mu:.4f}", f"({results.component.mu_ratio:.2%} of P_max)"])
        expected_inf = results.component.mu * cp.p_on
        writer.writerow(["# Expected Load (N->inf)", f"{expected_inf:.4f} kW", f"(mu * p(on) = {results.component.mu:.2f} * {cp.p_on:.4f})"])
        writer.writerow(["# Number of Runs per Connection Count", results.num_runs])
        writer.writerow([])

        # Table header
        header = ["connections", "p_max_kw"]
        for pct in results.percentiles:
            header.append(f"p_total_p{pct:g}_kw")
        for pct in results.percentiles:
            header.append(f"p_per_conn_p{pct:g}_kw")
        header.append("runs")
        writer.writerow(header)

        # Data rows
        for i, n in enumerate(results.connections):
            row = [n, results.component.p_max]
            for pct in results.percentiles:
                row.append(f"{results.p_total[pct][i]:.3f}")
            for pct in results.percentiles:
                row.append(f"{results.p_per_conn[pct][i]:.3f}")
            row.append(results.num_runs)
            writer.writerow(row)


def print_summary_table(results: SimulationResults) -> None:
    """Print an ASCII table of results to standard output."""
    cp = results.critical_point
    comp = results.component
    print("=" * 82)
    print("                 GRID USAGE GAUSSIAN LOAD SIMULATION REPORT               ")
    print("=" * 82)
    print(f"Component:          {comp.name}")
    print(f"Parameters:         P_max = {comp.p_max:.1f} kW | mu = {comp.mu:.2f} kW ({comp.mu_ratio:.1%}) | sigma = {comp.sigma:.2f} kW ({comp.sigma_ratio:.1%})")
    print("-" * 82)
    print("CRITICAL POINT IN TIME (Calculated ahead of Monte Carlo):")
    print(f"  Month:            {cp.month_name} (factor: {cp.month_factor:.2f})")
    print(f"  Day Type:         {cp.day_type.capitalize()}")
    print(f"  Interval / Time:  {cp.time_str} (factor: {cp.interval_factor:.2f})")
    expected_inf = comp.mu * cp.p_on
    print(f"  Peak p(on):       {cp.p_on:.2%} ({cp.p_on:.4f})")
    print(f"  Expected (N->inf):{expected_inf:.2f} kW / household (mu * p(on))")
    print(f"  Monte Carlo Runs: {results.num_runs} per connection count")
    print("=" * 82)
    print(f"{'Conns':>7} | {'P_tot p50':>10} | {'P_tot p95':>10} | {'P_tot p99':>10} | {'kW/conn p50':>11} | {'kW/conn p95':>11} | {'kW/conn p99':>11}")
    print("-" * 82)

    p50, p95, p99 = results.percentiles[0], results.percentiles[1], results.percentiles[2]
    for i, n in enumerate(results.connections):
        tot_50 = results.p_total[p50][i]
        tot_95 = results.p_total[p95][i]
        tot_99 = results.p_total[p99][i]
        pc_50 = results.p_per_conn[p50][i]
        pc_95 = results.p_per_conn[p95][i]
        pc_99 = results.p_per_conn[p99][i]
        print(f"{n:>7d} | {tot_50:>10.2f} | {tot_95:>10.2f} | {tot_99:>10.2f} | {pc_50:>11.2f} | {pc_95:>11.2f} | {pc_99:>11.2f}")
    print("=" * 82)


def main(args_list: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Simulate household grid usage with truncated Gaussian distribution at critical peak time."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to custom JSON configuration file")
    parser.add_argument(
        "--from-year-profile",
        type=str,
        default=None,
        help="Path to 2-column CSV (timestamp, consumption_kw) to generate GM parameters and factor profiles",
    )
    parser.add_argument(
        "--simultaneity",
        type=float,
        default=0.3,
        help="Target simultaneity factor s in (0, 1] when fitting from year profile (default: 0.3)",
    )
    parser.add_argument(
        "--max-p-on",
        type=float,
        default=1.0,
        help="Maximum activation probability p(on) at peak demand (default: 1.0)",
    )
    parser.add_argument("--save-example-config", type=str, default=None, help="Save default example config to JSON")
    parser.add_argument("--p-max", type=float, default=None, help="Override P_max in kW (default: 11.0)")
    parser.add_argument("--mu-ratio", type=float, default=None, help="Override mu ratio of P_max")
    parser.add_argument("--sigma-ratio", type=float, default=None, help="Override sigma ratio of P_max")
    parser.add_argument("--runs", type=int, default=DEFAULT_NUM_RUNS, help="Number of Monte Carlo simulations (default: 200)")
    parser.add_argument(
        "--connections",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONNECTIONS),
        help="List of connection counts to evaluate (default: 1 2 5 10 15 20 30 50 100 200)",
    )
    parser.add_argument("--csv", type=str, default="results/grid_usage_percentiles.csv", help="Path to save output CSV")
    parser.add_argument("--plot", type=str, default="results/grid_usage_percentiles.png", help="Path to save output plot PNG")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args(args_list)

    if args.save_example_config:
        export_example_config(args.save_example_config)
        print(f"Saved example configuration to {args.save_example_config}")
        return 0

    # Load, fit, or create component and factors
    if args.from_year_profile:
        from grid_gmm.profile_fitter import fit_gm_from_year_profile

        print(f"Fitting GM parameters from annual profile: {args.from_year_profile}")
        print(f"Target Simultaneity (Rusck g_inf): {args.simultaneity:.2f} | Max p(on): {args.max_p_on:.1%}")

        component, factors, summary = fit_gm_from_year_profile(
            csv_path=args.from_year_profile,
            simultaneity=args.simultaneity,
            max_p_on=args.max_p_on,
            mu_ratio=args.mu_ratio,
            sigma_ratio=args.sigma_ratio,
        )

        print(f"  Profile Diversified P_peak: {summary.p_peak:.2f} kW")
        print(f"  Derived Rusck P_max:        {summary.p_max:.2f} kW (P_peak / simultaneity)")
        print(f"  Active Load mu:             {component.mu:.2f} kW ({component.mu_ratio:.1%} of P_max)")
        print(f"  Active Load sigma:          {component.sigma:.2f} kW ({component.sigma_ratio:.1%} of P_max)")
        print(f"  Peak Activation p(on):      {summary.peak_p_on:.1%}")
        print(f"  Annual Energy:              {summary.annual_energy_kwh:.1f} kWh ({summary.total_records} intervals)")
        print()

    elif args.config:
        component, factors = load_config_from_json(args.config)
    else:
        component = create_example_component()
        factors = create_example_profiles()

    # Apply command-line overrides
    if args.p_max is not None:
        component.p_max = args.p_max
    if args.mu_ratio is not None:
        component.mu_ratio = args.mu_ratio
    if args.sigma_ratio is not None:
        component.sigma_ratio = args.sigma_ratio

    # Initialize engine
    engine = MonteCarloEngine(component=component, factors=factors, seed=args.seed)

    # Run simulation
    results = engine.run_critical_time_simulation(
        connection_counts=args.connections,
        num_runs=args.runs,
        percentiles=DEFAULT_PERCENTILES,
    )

    # Print summary table
    print_summary_table(results)

    # Save CSV
    csv_path = Path(args.csv)
    save_results_to_csv(results, csv_path)
    print(f"Results successfully saved to CSV: {csv_path.resolve()}")

    # Generate and save plot
    plot_path = Path(args.plot)
    plot_percentiles_per_connection(results, output_path=plot_path)
    print(f"Plot successfully saved to:        {plot_path.resolve()}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
