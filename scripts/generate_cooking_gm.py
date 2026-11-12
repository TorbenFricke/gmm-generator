#!/usr/bin/env python3
"""Generate the Gaussian model (GM) for electric cooking from a typical daily profile.

The 24 hourly values below (kW, diversified average per household, hour starting at 00:00) are
assumed to repeat on every day of the year. The script

1. writes them as a one-year hourly profile to ``input_profiles/electric_cooking_hourly.csv``
   and plots the daily profile to ``docs/electric_cooking_daily_profile.png``,
2. derives the asymptotic simultaneity from the chosen single-household maximum,
   ``g_inf = P_peak / P_max`` (default ``P_max = 3 kW``),
3. keeps ``p(on)`` at the peak fixed (default 10 %) and calibrates ``mu`` and ``sigma``:

   - ``mu`` is solved so that the *truncated* mean matches the profile exactly,
     ``p(on) * E[P_active] = P_peak`` (i.e. ``g(N) -> g_inf``), and
   - ``sigma`` is chosen so that the simulated 99th-percentile simultaneity ``g(N)`` comes as
     close as possible to Rusck's ``g(N) = g_inf + (1 - g_inf) / sqrt(N)``,

4. runs the regular CLI with these parameters and saves config, xlsx export, results and plot to ``results/``.

Note: for small ``g_inf`` Rusck's curve needs more variability than any model of independent
households capped at ``P_max`` can provide (a single household's variance is at most
``g_inf * (1 - g_inf) * P_max^2``). The calibration then returns the closest feasible curve,
which lies below Rusck for intermediate N, and prints the comparison.

Usage:
    python scripts/generate_cooking_gm.py [--p-max 3.0] [--max-p-on 0.1]
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import truncnorm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from grid_gmm.cli import main as grid_gmm_main  # noqa: E402
from grid_gmm.simulation import DEFAULT_CONNECTIONS  # noqa: E402

#: Typical daily electric cooking load in kW for the hours 00:00-01:00 ... 23:00-24:00.
HOURLY_COOKING_KW = (
    0.001493597,  # 00:00
    0.001493597,
    0.001493597,
    0.001493597,
    0.002987193,
    0.014935966,  # 05:00
    0.059743863,
    0.059743863,
    0.029871932,
    0.014935966,
    0.007467983,  # 10:00
    0.029871932,
    0.029871932,
    0.014935966,
    0.007467983,
    0.007467983,  # 15:00
    0.149359659,
    0.149359659,  # 17:00 (peak)
    0.074679829,
    0.037339915,
    0.022403949,  # 20:00
    0.014935966,
    0.007467983,
    0.001493597,  # 23:00
)

#: Any non-leap year works, since every day has the same values.
PROFILE_YEAR = 2023

#: Upper bound of the sigma search (ratio of P_max). Beyond ~1 the truncated Gaussian is
#: practically uniform on [0, P_max], so larger values add no further variability.
SIGMA_RATIO_MAX = 1.0
CALIBRATION_RUNS = 20_000
CALIBRATION_SEED = 1
CALIBRATION_CONNECTIONS = np.array(DEFAULT_CONNECTIONS)

DEFAULT_PROFILE = REPO_ROOT / "input_profiles" / "electric_cooking_hourly.csv"
DEFAULT_RESULTS_STEM = REPO_ROOT / "results" / "electric_cooking_pmax_3kW"
DEFAULT_CONFIG = REPO_ROOT / "results" / "electric_cooking_config.json"
DEFAULT_XLSX = REPO_ROOT / "results" / "electric_cooking_gm.xlsx"
DEFAULT_PROFILE_PLOT = REPO_ROOT / "docs" / "electric_cooking_daily_profile.png"


def plot_daily_profile(path: Path, hourly_kw: tuple[float, ...] = HOURLY_COOKING_KW, dpi: int = 150) -> Path:
    """Plot the 24 hourly input values as a step curve (used in docs/electric_cooking_gm.md)."""
    from matplotlib.figure import Figure

    hours = np.arange(25)
    values = np.append(hourly_kw, hourly_kw[-1])  # repeat last value so the 23:00 step reaches 24:00
    fig = Figure(figsize=(8, 4), dpi=dpi)
    ax = fig.add_subplot()
    ax.step(hours, values, where="post", color="tab:orange", linewidth=2)
    ax.fill_between(hours, values, step="post", color="tab:orange", alpha=0.2)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, None)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], rotation=45)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average load per household (kW)")
    ax.set_title("Electric cooking: input daily profile (hourly, every day identical)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return path


def write_year_profile(path: Path, hourly_kw: tuple[float, ...] = HOURLY_COOKING_KW, year: int = PROFILE_YEAR) -> Path:
    """Repeat the daily hourly profile for every day of ``year`` and write it as ``timestamp, consumption_kw``."""
    if len(hourly_kw) != 24:
        raise ValueError(f"Expected 24 hourly values, got {len(hourly_kw)}")

    start = datetime(year, 1, 1)
    hours = (datetime(year + 1, 1, 1) - start) // timedelta(hours=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "consumption_kw"])
        for h in range(hours):
            ts = start + timedelta(hours=h)
            writer.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), f"{hourly_kw[ts.hour]:.9f}"])
    return path


def rusck(g_inf: float, n: np.ndarray) -> np.ndarray:
    return g_inf + (1.0 - g_inf) / np.sqrt(n)


def truncated_mean_ratio(mu: float, sigma: float) -> float:
    """Mean of N(mu, sigma) truncated to [0, 1] (all quantities as ratios of P_max)."""
    return float(truncnorm.mean(-mu / sigma, (1.0 - mu) / sigma, loc=mu, scale=sigma))


def mu_for_mean(target_mean: float, sigma: float) -> float:
    """Location mu whose truncated mean on [0, 1] equals ``target_mean`` (mean is increasing in mu)."""
    if truncated_mean_ratio(1.0, sigma) < target_mean:
        raise ValueError(f"No mu <= 1 reaches a mean of {target_mean:.3f} with sigma_ratio={sigma:.3f}")
    return brentq(lambda mu: truncated_mean_ratio(mu, sigma) - target_mean, -50.0, 1.0)


@dataclass(frozen=True)
class Calibration:
    mu_ratio: float
    sigma_ratio: float
    connections: np.ndarray
    g_model: np.ndarray
    g_rusck: np.ndarray

    @property
    def rms_rel_error(self) -> float:
        return float(np.sqrt(np.mean(((self.g_model - self.g_rusck) / self.g_rusck) ** 2)))


class _SimultaneityCurve:
    """99th-percentile simultaneity g(N) with common random numbers, so it is smooth in (mu, sigma)."""

    def __init__(self, p_on: float, connections: np.ndarray, runs: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.connections = connections
        self.runs = runs
        self.active = [rng.binomial(int(n), p_on, size=runs) for n in connections]
        self.uniforms = [rng.random(int(k.sum())) for k in self.active]

    def __call__(self, mu: float, sigma: float) -> np.ndarray:
        a, b = -mu / sigma, (1.0 - mu) / sigma
        values = []
        for n, active, u in zip(self.connections, self.active, self.uniforms):
            loads = truncnorm.ppf(u, a, b, loc=mu, scale=sigma)
            totals = np.bincount(np.repeat(np.arange(self.runs), active), weights=loads, minlength=self.runs)
            values.append(np.percentile(totals, 99) / n)
        return np.array(values)


def calibrate_mu_sigma(
    g_inf: float,
    p_on: float,
    connections: np.ndarray = CALIBRATION_CONNECTIONS,
    runs: int = CALIBRATION_RUNS,
    seed: int = CALIBRATION_SEED,
) -> Calibration:
    """Fix the mean via mu and choose sigma to minimize the RMS relative deviation from Rusck."""
    target_mean = g_inf / p_on
    if not (0.0 < target_mean < 1.0):
        raise ValueError(f"Need g_inf < p(on) for a feasible mean, got g_inf={g_inf:.3f}, p(on)={p_on:.3f}")

    curve = _SimultaneityCurve(p_on, connections, runs, seed)
    target = rusck(g_inf, connections)

    def objective(sigma: float) -> float:
        g = curve(mu_for_mean(target_mean, sigma), sigma)
        return float(np.mean(((g - target) / target) ** 2))

    sigma_lo = 0.01
    sigma = float(minimize_scalar(objective, bounds=(sigma_lo, SIGMA_RATIO_MAX), method="bounded").x)
    mu = mu_for_mean(target_mean, sigma)
    return Calibration(mu, sigma, connections, curve(mu, sigma), target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--p-max", type=float, default=3.0, help="Maximum power of one household in kW (default: 3)")
    parser.add_argument("--max-p-on", type=float, default=0.1, help="p(on) at the peak (default: 0.1)")
    parser.add_argument("--runs", type=int, default=1000, help="Monte Carlo runs of the final simulation")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE, help="Where to write the year profile")
    parser.add_argument(
        "--profile-plot", type=Path, default=DEFAULT_PROFILE_PLOT, help="Where to save the daily profile plot"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Where to save the fitted GM config")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX, help="Where to export the GM as one-row xlsx")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_STEM, help="Results path without extension")
    args = parser.parse_args(argv)
    results_stem = str(args.results)  # not with_suffix(): the stem may contain a dot

    profile = write_year_profile(args.profile)
    print(f"Wrote {len(HOURLY_COOKING_KW)}-hour daily profile for {PROFILE_YEAR} to {profile}")
    print(f"Plotted daily profile to {plot_daily_profile(args.profile_plot)}")

    p_peak = max(HOURLY_COOKING_KW)
    g_inf = p_peak / args.p_max
    print(f"P_peak = {p_peak:.4f} kW, P_max = {args.p_max:g} kW  ->  g_inf = P_peak / P_max = {g_inf:.4f}")

    cal = calibrate_mu_sigma(g_inf, args.max_p_on)
    print(f"Calibrated: mu_ratio = {cal.mu_ratio:.4f}, sigma_ratio = {cal.sigma_ratio:.4f}")
    if cal.sigma_ratio > 0.99 * SIGMA_RATIO_MAX:
        print("  (sigma at its upper bound: maximum variability reached, Rusck is not attainable exactly)")
    print(f"  {'N':>5} | {'Rusck':>6} | {'GM':>6}")
    for n, r, g in zip(cal.connections, cal.g_rusck, cal.g_model):
        print(f"  {n:>5} | {r:6.3f} | {g:6.3f}")
    print(f"  RMS relative deviation from Rusck: {cal.rms_rel_error:.1%}\n")

    return grid_gmm_main(
        [
            "--from-year-profile", str(profile),
            "--simultaneity", repr(g_inf),
            "--max-p-on", str(args.max_p_on),
            "--mu-ratio", repr(cal.mu_ratio),
            "--sigma-ratio", repr(cal.sigma_ratio),
            "--name", f"Electric Cooking (P_max={args.p_max:g} kW, p_on={args.max_p_on:.0%})",
            "--runs", str(args.runs),
            "--save-config", str(args.config),
            "--save-xlsx", str(args.xlsx),
            "--csv", results_stem + ".csv",
            "--plot", results_stem + ".png",
        ]
    )  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
