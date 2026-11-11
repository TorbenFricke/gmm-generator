"""Tabular reporting of simulation results: CSV + JSON metadata sidecar and console summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from grid_gmm.simulation import SimulationResults


def results_table(results: SimulationResults) -> tuple[list[str], list[list[int | float]]]:
    """Return (header, rows) of the per-connection-count results table."""
    header = ["connections", "p_max_kw"]
    header += [f"p_total_p{pct:g}_kw" for pct in results.percentiles]
    header += [f"p_per_conn_p{pct:g}_kw" for pct in results.percentiles]
    header.append("runs")

    rows: list[list[int | float]] = []
    for i, n in enumerate(results.connections):
        row: list[int | float] = [n, results.component.p_max]
        row += [float(results.p_total[pct][i]) for pct in results.percentiles]
        row += [float(results.p_per_conn[pct][i]) for pct in results.percentiles]
        row.append(results.num_runs)
        rows.append(row)
    return header, rows


def results_metadata(results: SimulationResults) -> dict[str, Any]:
    """Describe the simulation inputs and critical point as a JSON-serializable dict."""
    cp = results.critical_point
    comp = results.component
    return {
        "component": {
            "name": comp.name,
            "p_max_kw": comp.p_max,
            "mu_ratio": comp.mu_ratio,
            "sigma_ratio": comp.sigma_ratio,
            "mu_kw": comp.mu,
            "sigma_kw": comp.sigma,
            "mean_active_kw": comp.mean,
        },
        "critical_point": {
            "month": cp.month,
            "month_name": cp.month_name,
            "day_type": cp.day_type.value,
            "interval_idx": cp.interval_idx,
            "time": cp.time_str,
            "p_on": cp.p_on,
            "month_factor": cp.month_factor,
            "interval_factor": cp.interval_factor,
        },
        "expected_load_per_conn_kw": results.expected_load_per_conn,
        "num_runs": results.num_runs,
        "percentiles": results.percentiles,
        "connections": results.connections,
    }


def metadata_path_for(csv_path: Path) -> Path:
    """Path of the JSON metadata sidecar belonging to a results CSV (``x.csv`` -> ``x.meta.json``)."""
    return csv_path.with_suffix(".meta.json")


def save_results_to_csv(results: SimulationResults, filepath: str | Path) -> Path:
    """Write the results table as plain CSV plus a ``.meta.json`` sidecar; returns the sidecar path."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    header, rows = results_table(results)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([f"{v:.3f}" if isinstance(v, float) and col > 1 else v for col, v in enumerate(row)])

    meta_path = metadata_path_for(path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results_metadata(results), f, indent=2)
        f.write("\n")
    return meta_path


def format_summary_table(results: SimulationResults) -> str:
    """Render a human-readable report of the results for the console."""
    cp = results.critical_point
    comp = results.component
    pcts = results.percentiles

    columns = ["Conns"] + [f"P_tot p{p:g}" for p in pcts] + [f"kW/conn p{p:g}" for p in pcts]
    widths = [7] + [max(10, len(c)) for c in columns[1:]]
    width = sum(widths) + 3 * (len(widths) - 1)
    rule, double_rule = "-" * width, "=" * width

    lines = [
        double_rule,
        "GRID USAGE GAUSSIAN LOAD SIMULATION REPORT".center(width),
        double_rule,
        f"Component:          {comp.name}",
        f"Parameters:         P_max = {comp.p_max:.1f} kW | mu = {comp.mu:.2f} kW ({comp.mu_ratio:.1%})"
        f" | sigma = {comp.sigma:.2f} kW ({comp.sigma_ratio:.1%}) | E[active] = {comp.mean:.2f} kW",
        rule,
        "CRITICAL POINT IN TIME (Calculated ahead of Monte Carlo):",
        f"  Month:            {cp.month_name} (factor: {cp.month_factor:.2f})",
        f"  Day Type:         {cp.day_type.label}",
        f"  Interval / Time:  {cp.time_str} (factor: {cp.interval_factor:.2f})",
        f"  Peak p(on):       {cp.p_on:.2%} ({cp.p_on:.4f})",
        f"  Expected (N->inf):{results.expected_load_per_conn:.2f} kW / household (E[active] * p(on))",
        f"  Monte Carlo Runs: {results.num_runs} per connection count",
        double_rule,
        " | ".join(c.rjust(w) for c, w in zip(columns, widths)),
        rule,
    ]
    for i, n in enumerate(results.connections):
        values = [results.p_total[p][i] for p in pcts] + [results.p_per_conn[p][i] for p in pcts]
        cells = [f"{n:>{widths[0]}d}"] + [f"{v:>{w}.2f}" for v, w in zip(values, widths[1:])]
        lines.append(" | ".join(cells))
    lines.append(double_rule)
    return "\n".join(lines)
