"""Plotting functions for grid load simulation results."""

import os
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt

from grid_gmm.simulation import SimulationResults


def plot_percentiles_per_connection(
    results: SimulationResults,
    output_path: Optional[Union[str, Path]] = None,
    log_x: bool = True,
    show_plot: bool = False,
) -> plt.Figure:
    """Plot the power per connection (y-axis) against the number of connections (x-axis).

    Includes:
    - A line for each percentile (50th, 95th, 99th).
    - A horizontal dashed line at P_max.

    Args:
        results: SimulationResults from MonteCarloEngine.
        output_path: Path to save the figure image (PNG, PDF, etc.).
        log_x: Whether to use logarithmic scale for the x-axis (recommended for 1 to 200).
        show_plot: Whether to call plt.show().

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    connections = results.connections
    p_max = results.component.p_max
    cp = results.critical_point

    # Styling for percentiles
    styles = {
        50.0: {"color": "#2ca02c", "marker": "o", "label": "50th Percentile (Median)", "linestyle": "-"},
        95.0: {"color": "#ff7f0e", "marker": "s", "label": "95th Percentile", "linestyle": "-"},
        99.0: {"color": "#d62728", "marker": "^", "label": "99th Percentile", "linestyle": "-"},
    }

    # Plot lines for each percentile
    for pct in results.percentiles:
        vals = results.p_per_conn[pct]
        style = styles.get(
            pct,
            {"color": "#1f77b4", "marker": "d", "label": f"{pct:g}th Percentile", "linestyle": "-"},
        )
        ax.plot(
            connections,
            vals,
            color=style["color"],
            marker=style["marker"],
            linewidth=2,
            markersize=6,
            label=style["label"],
        )

    # Horizontal line at P_max
    ax.axhline(
        y=p_max,
        color="#222222",
        linestyle="--",
        linewidth=1.8,
        label=f"P_max = {p_max:.1f} kW",
    )

    # Horizontal line at expected value for infinite connections: mu * p(on)
    expected_inf = results.component.mu * cp.p_on
    ax.axhline(
        y=expected_inf,
        color="#6f42c1",
        linestyle=":",
        linewidth=2.2,
        label=f"Expected (N→∞): mu × p(on) = {expected_inf:.2f} kW",
    )

    # Set axes labels and scales
    ax.set_xlabel("Number of Connected Households (N)", fontsize=12, labelpad=8)
    ax.set_ylabel("Power per Connection (kW / household)", fontsize=12, labelpad=8)

    if log_x:
        ax.set_xscale("log")
        ax.set_xticks(connections)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Set y-axis to start slightly below 0 or at 0, up to slightly above P_max
    y_max = max(p_max * 1.08, max(results.p_per_conn[max(results.percentiles)]) * 1.1)
    ax.set_ylim(bottom=0.0, top=y_max)

    # Title & Subtitle with Critical Point details
    ax.set_title(
        f"Simultaneous Grid Usage at Peak Time: {cp.month_name}, {cp.day_type.capitalize()} at {cp.time_str}\n"
        f"(P_max = {p_max:.1f} kW, mu = {results.component.mu:.2f} kW, "
        f"sigma = {results.component.sigma:.2f} kW, peak p(on) = {cp.p_on:.1%}, Runs = {results.num_runs})",
        fontsize=11,
        pad=12,
    )

    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10, loc="upper right")
    plt.tight_layout()

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=300)

    if show_plot:
        plt.show()

    return fig
