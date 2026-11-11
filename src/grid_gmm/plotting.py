"""Plotting of simulation results.

Uses matplotlib's object-oriented API (``Figure``) instead of ``pyplot`` so that no global
figure state is created: figures are garbage-collected normally and no GUI backend is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

from grid_gmm.simulation import SimulationResults

PERCENTILE_STYLES: dict[float, dict[str, Any]] = {
    50.0: {"color": "#2ca02c", "marker": "o", "label": "50th Percentile (Median)"},
    95.0: {"color": "#ff7f0e", "marker": "s", "label": "95th Percentile"},
    99.0: {"color": "#d62728", "marker": "^", "label": "99th Percentile"},
}
_FALLBACK_STYLE = {"color": "#1f77b4", "marker": "d"}


def _style_for(pct: float) -> dict[str, Any]:
    return PERCENTILE_STYLES.get(pct, {**_FALLBACK_STYLE, "label": f"{pct:g}th Percentile"})


def plot_percentiles_per_connection(
    results: SimulationResults,
    output_path: str | Path | None = None,
    log_x: bool = True,
    dpi: int = 300,
) -> Figure:
    """Plot power per connection (y) against the number of connections (x).

    Draws one line per percentile, a dashed line at P_max and a dotted line at the
    expected load per connection for N -> inf (E[active power] * p(on)).

    Args:
        results: Output of :meth:`MonteCarloEngine.run_critical_time_simulation`.
        output_path: If given, the figure is saved there (format from the file extension).
        log_x: Use a logarithmic x-axis.
        dpi: Resolution of the saved image.
    """
    fig = Figure(figsize=(10, 6), dpi=dpi)
    ax = fig.subplots()

    connections = results.connections
    comp = results.component
    cp = results.critical_point

    for pct in results.percentiles:
        style = _style_for(pct)
        ax.plot(connections, results.p_per_conn[pct], linewidth=2, markersize=6, **style)

    ax.axhline(y=comp.p_max, color="#222222", linestyle="--", linewidth=1.8, label=f"P_max = {comp.p_max:.1f} kW")

    expected = results.expected_load_per_conn
    ax.axhline(
        y=expected,
        color="#6f42c1",
        linestyle=":",
        linewidth=2.2,
        label=f"Expected (N→∞): E[P_active] × p(on) = {expected:.2f} kW",
    )

    ax.set_xlabel("Number of Connected Households (N)", fontsize=12, labelpad=8)
    ax.set_ylabel("Power per Connection (kW / household)", fontsize=12, labelpad=8)

    if log_x:
        ax.set_xscale("log")
        ax.set_xticks(connections)
        ax.xaxis.set_major_formatter(ScalarFormatter())

    highest = max(results.p_per_conn[max(results.percentiles)])
    ax.set_ylim(bottom=0.0, top=max(comp.p_max * 1.08, highest * 1.1))

    ax.set_title(
        f"Simultaneous Grid Usage at Peak Time: {cp.month_name}, {cp.day_type.label} at {cp.time_str}\n"
        f"(P_max = {comp.p_max:.1f} kW, mu = {comp.mu:.2f} kW, "
        f"sigma = {comp.sigma:.2f} kW, peak p(on) = {cp.p_on:.1%}, Runs = {results.num_runs})",
        fontsize=11,
        pad=12,
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10, loc="upper right")
    fig.tight_layout()

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)

    return fig
