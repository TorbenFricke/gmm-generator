"""Command-line interface for the grid usage Gaussian simulation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from grid_gmm.config import load_config_from_json, save_config_to_json
from grid_gmm.example_data import create_example_component, create_example_profiles, export_example_config
from grid_gmm.factors import FactorProfiles
from grid_gmm.model import GaussianComponent
from grid_gmm.plotting import plot_percentiles_per_connection
from grid_gmm.profile_fitter import convert_w_profile_to_kw_csv, fit_gm_from_year_profile
from grid_gmm.reporting import format_summary_table, save_results_to_csv
from grid_gmm.simulation import (
    DEFAULT_CONNECTIONS,
    DEFAULT_NUM_RUNS,
    DEFAULT_PERCENTILES,
    DEFAULT_SEED,
    MonteCarloEngine,
)
from grid_gmm.xlsx_export import model_name_from_path, save_model_to_xlsx

DEFAULT_SIMULTANEITY = 0.3
DEFAULT_MAX_P_ON = 1.0
DEFAULT_CSV = "results/grid_usage_percentiles.csv"
DEFAULT_PLOT = "results/grid_usage_percentiles.png"

#: Options that only make sense together with --from-year-profile (dest names).
_FIT_ONLY_OPTIONS = ("simultaneity", "max_p_on", "timezone")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grid-gmm",
        description="Simulate household grid usage with a truncated Gaussian load model at the critical peak time.",
    )

    source = parser.add_argument_group("input source / standalone actions (mutually exclusive)")
    exclusive = source.add_mutually_exclusive_group()
    exclusive.add_argument("--config", type=Path, help="Load component and factors from a JSON configuration")
    exclusive.add_argument(
        "--from-year-profile",
        type=Path,
        metavar="CSV",
        help="Fit component and factors from a 'timestamp, power' CSV (kW, or W if the peak exceeds 20)",
    )
    exclusive.add_argument("--save-example-config", type=Path, metavar="JSON", help="Write the example config and exit")
    exclusive.add_argument(
        "--convert-w-profile",
        type=Path,
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        help="Convert a W profile CSV to a 15-minute 'timestamp, consumption_kw' CSV and exit",
    )

    fit = parser.add_argument_group("year-profile fitting (with --from-year-profile)")
    fit.add_argument(
        "--simultaneity",
        type=float,
        help=f"Target simultaneity factor s in (0, 1] (default: {DEFAULT_SIMULTANEITY})",
    )
    fit.add_argument(
        "--max-p-on",
        type=float,
        help=f"Activation probability p(on) at peak demand (default: {DEFAULT_MAX_P_ON})",
    )
    fit.add_argument(
        "--timezone",
        help="IANA timezone (e.g. Europe/Berlin) to convert timezone-aware timestamps to before fitting",
    )
    fit.add_argument(
        "--value-column",
        help="Name or index of the power column (also used by --convert-w-profile)",
    )

    model = parser.add_argument_group("model overrides")
    model.add_argument("--p-max", type=float, help="Override P_max in kW (not allowed with --from-year-profile)")
    model.add_argument("--mu-ratio", type=float, help="Override mu as ratio of P_max")
    model.add_argument("--sigma-ratio", type=float, help="Override sigma as ratio of P_max")
    model.add_argument("--name", help="Override the component name")

    sim = parser.add_argument_group("simulation")
    sim.add_argument(
        "--runs", type=int, default=DEFAULT_NUM_RUNS, help=f"Monte Carlo runs (default: {DEFAULT_NUM_RUNS})"
    )
    sim.add_argument(
        "--connections",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONNECTIONS),
        help="Connection counts to evaluate (default: %(default)s)",
    )
    sim.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=list(DEFAULT_PERCENTILES),
        help="Percentiles to report (default: %(default)s)",
    )
    sim.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default: {DEFAULT_SEED})")

    out = parser.add_argument_group("output")
    out.add_argument("--csv", type=Path, default=Path(DEFAULT_CSV), help="Results CSV (default: %(default)s)")
    out.add_argument("--plot", type=Path, default=Path(DEFAULT_PLOT), help="Results plot (default: %(default)s)")
    out.add_argument("--save-config", type=Path, metavar="JSON", help="Also save the used component and factors")
    out.add_argument(
        "--save-xlsx",
        type=Path,
        metavar="XLSX",
        help="Also export the model as a one-row xlsx (mu/sigma as ratios of P_max, named after the input file)",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.from_year_profile is None:
        given = [f"--{opt.replace('_', '-')}" for opt in _FIT_ONLY_OPTIONS if getattr(args, opt) is not None]
        if given:
            parser.error(f"{', '.join(given)} can only be used with --from-year-profile")
    elif args.p_max is not None:
        parser.error("--p-max cannot be combined with --from-year-profile: P_max is derived as P_peak / simultaneity")


def _fit_from_year_profile(args: argparse.Namespace) -> tuple[GaussianComponent, FactorProfiles]:
    simultaneity = DEFAULT_SIMULTANEITY if args.simultaneity is None else args.simultaneity
    max_p_on = DEFAULT_MAX_P_ON if args.max_p_on is None else args.max_p_on

    print(f"Fitting GM parameters from annual profile: {args.from_year_profile}")
    print(f"Target Simultaneity (Rusck g_inf): {simultaneity:.2f} | Max p(on): {max_p_on:.1%}")

    fit_kwargs = {"component_name": args.name} if args.name else {}
    component, factors, summary = fit_gm_from_year_profile(
        csv_path=args.from_year_profile,
        simultaneity=simultaneity,
        max_p_on=max_p_on,
        mu_ratio=args.mu_ratio,
        sigma_ratio=args.sigma_ratio,
        value_column=args.value_column,
        timezone=args.timezone,
        **fit_kwargs,
    )

    print(f"  Source:                     {summary.source_unit}, {summary.source_resolution_minutes}-minute data")
    print(f"  Profile Diversified P_peak: {summary.p_peak:.2f} kW")
    print(f"  Derived Rusck P_max:        {summary.p_max:.2f} kW (P_peak / simultaneity)")
    print(f"  Active Load mu:             {component.mu:.2f} kW ({component.mu_ratio:.1%} of P_max)")
    print(f"  Active Load sigma:          {component.sigma:.2f} kW ({component.sigma_ratio:.1%} of P_max)")
    print(f"  Peak Activation p(on):      {summary.peak_p_on:.1%}")
    print(f"  Annual Energy:              {summary.annual_energy_kwh:.1f} kWh ({summary.total_records} intervals)")
    print()
    return component, factors


def _load_model(args: argparse.Namespace) -> tuple[GaussianComponent, FactorProfiles]:
    if args.from_year_profile is not None:
        return _fit_from_year_profile(args)

    if args.config is not None:
        component, factors = load_config_from_json(args.config)
    else:
        component, factors = create_example_component(), create_example_profiles()

    overridden = component.with_overrides(
        p_max=args.p_max, mu_ratio=args.mu_ratio, sigma_ratio=args.sigma_ratio, name=args.name
    )
    return overridden, factors


def _run(args: argparse.Namespace) -> int:
    if args.save_example_config is not None:
        export_example_config(args.save_example_config)
        print(f"Saved example configuration to {args.save_example_config}")
        return 0

    if args.convert_w_profile is not None:
        src, dst = args.convert_w_profile
        convert_kwargs = {"value_column": args.value_column} if args.value_column is not None else {}
        convert_w_profile_to_kw_csv(src, dst, **convert_kwargs)
        print(f"Converted {src} -> {dst}")
        return 0

    component, factors = _load_model(args)
    if args.save_config is not None:
        save_config_to_json(component, factors, args.save_config)
        print(f"Configuration saved to:            {args.save_config.resolve()}")
    if args.save_xlsx is not None:
        source = args.from_year_profile or args.config
        name = model_name_from_path(source) if source is not None else component.name
        save_model_to_xlsx(component, factors, args.save_xlsx, name=name)
        print(f"Model exported to xlsx:            {args.save_xlsx.resolve()} (Name: {name})")

    engine = MonteCarloEngine(component=component, factors=factors, seed=args.seed)
    results = engine.run_critical_time_simulation(
        connection_counts=args.connections,
        num_runs=args.runs,
        percentiles=args.percentiles,
    )

    print(format_summary_table(results))

    meta_path = save_results_to_csv(results, args.csv)
    print(f"Results successfully saved to CSV: {args.csv.resolve()} (metadata: {meta_path.name})")

    plot_percentiles_per_connection(results, output_path=args.plot)
    print(f"Plot successfully saved to:        {args.plot.resolve()}")
    return 0


def main(args_list: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(args_list)
    _validate_args(parser, args)
    try:
        return _run(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
