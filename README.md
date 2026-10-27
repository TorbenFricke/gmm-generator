# Grid Usage Gaussian Modeling

A Python simulation tool modeling household electricity grid demand using truncated Gaussian load distributions and time-dependent activation probabilities ($p(\text{on})$).

## Features

- **Truncated Gaussian Model**: Bounded strictly within $[0, P_{\text{max}}]$, with fixed $\mu$ and $\sigma$ ratios of $P_{\text{max}}$.
- **Time Profiles**: Evaluates 12 monthly factors and 96 15-minute interval factors (for both weekdays and weekends).
- **Critical Peak Pre-Calculation**: Determines the peak grid stress time step $(m^*, t^*, \text{day\_type}^*)$ ahead of the Monte Carlo simulation.
- **Monte Carlo Engine**: Runs 200 simulations for $N \in \{1, 2, 5, 10, 15, 20, 30, 50, 100, 200\}$ connections.
- **Percentiles Analysis**: Computes 50th, 95th, and 99th percentiles for total load and per-connection power.
- **Reporting & Visualization**: Exports results to CSV and generates a plot with percentile lines and a horizontal line at $P_{\text{max}}$.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Simulation

Run with default example profiles (11 kW connection, December weekday 19:15 peak):

```bash
python run_simulation.py
```

### CLI Options

```bash
python run_simulation.py \
  --p-max 11.0 \
  --mu-ratio 0.35 \
  --sigma-ratio 0.15 \
  --runs 200 \
  --connections 1 2 5 10 15 20 30 50 100 200 \
  --csv results/grid_usage_percentiles.csv \
  --plot results/grid_usage_percentiles.png \
  --seed 42
```

### Generating GM Parameters from a Year Profile

To derive parameters directly from a 2-column CSV (`timestamp, consumption_kw`) with a target simultaneity $s$ (fixing $\mu=0.5$ and $\sigma=0.2$):

```bash
python run_simulation.py \
  --from-year-profile path/to/year_profile.csv \
  --simultaneity 0.35 \
  --csv results/grid_usage_percentiles.csv \
  --plot results/grid_usage_percentiles.png
```

- Calculates $P_{\text{max}} = P_{\text{peak}} / s$
- Calculates peak activation probability $p(\text{on})_{\text{peak}} = s / \mu = 2s$
- Decomposes profile into 12 monthly factors and 96 weekday/weekend interval factors
- Runs critical point Monte Carlo simulation and outputs CSV & plot

Save default example dataset configuration to JSON:
```bash
python run_simulation.py --save-example-config config/example_input.json
```

Load custom JSON configuration:
```bash
python run_simulation.py --config config/example_input.json
```

## Running Tests

```bash
PYTHONPATH=src pytest -v
```
