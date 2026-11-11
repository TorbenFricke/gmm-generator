# Grid Usage Gaussian Modeling

A Python tool that simulates simultaneous household grid demand. Each connection is modelled by a
**single truncated Gaussian** active-load distribution combined with a time-dependent activation
probability $p(\text{on})$. (The package name `grid_gmm` is historical; Gaussian *mixtures* are not
implemented.)

## Model

- **Active load**: $P \sim \mathcal{N}(\mu, \sigma)$ truncated to $[0, P_{\text{max}}]$, with
  $\mu = \mu_{\text{ratio}} P_{\text{max}}$ and $\sigma = \sigma_{\text{ratio}} P_{\text{max}}$.
  $\mu$ is the *location* of the underlying normal distribution; because of the truncation the
  expected active power $E[P]$ (reported as `E[active]`) differs from $\mu$.
- **Time profiles**: $p(\text{on}) = f_{\text{month}}(m) \cdot f_{\text{interval}}(t, \text{day type})$,
  clipped to $[0, 1]$, from 12 month factors and 96 15-minute interval factors for weekdays and weekends.
- **Critical point**: the $(m^*, t^*, \text{day type}^*)$ with the highest $p(\text{on})$ is found
  before the Monte Carlo simulation.
- **Monte Carlo**: for every connection count $N$ the number of active connections is
  $\text{Binomial}(N, p(\text{on}))$ and each active load is drawn from the truncated Gaussian.
- **Outputs**: percentiles (default 50/95/99) of total load and load per connection, the
  $N \to \infty$ expectation $E[P] \cdot p(\text{on})$, a CSV, a JSON metadata sidecar and a plot.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`requirements.txt` lists the runtime dependencies; `requirements-lock.txt` pins the exact versions
the project was last tested with (`pip install -r requirements-lock.txt && pip install -e . --no-deps`).

## Running the Simulation

After installation the `grid-gmm` command is available. `python run_simulation.py` is an equivalent
entry point that works from a source checkout without installing.

Run with the built-in example (11 kW connection, December weekday 19:15 peak):

```bash
grid-gmm
```

### Options

```bash
grid-gmm \
  --p-max 11.0 \
  --mu-ratio 0.35 \
  --sigma-ratio 0.15 \
  --runs 200 \
  --connections 1 2 5 10 15 20 30 50 100 200 \
  --percentiles 50 95 99 \
  --csv results/grid_usage_percentiles.csv \
  --plot results/grid_usage_percentiles.png \
  --seed 42
```

Overrides are validated: an invalid value (for example `--mu-ratio 1.5`) is rejected. Run
`grid-gmm --help` for all options. `--config`, `--from-year-profile`, `--save-example-config` and
`--convert-w-profile` are mutually exclusive.

### Fitting the Model to a Year Profile

Fit from a CSV with a timestamp column and a power column:

```bash
grid-gmm \
  --from-year-profile input_profiles/hp_archetype_terrace_converted.csv \
  --simultaneity 0.7 \
  --max-p-on 1.0 \
  --save-config results/my_fitted_config.json
```

With target simultaneity $s$ and peak activation probability $p_{\text{peak}}$ (`--max-p-on`):

- $P_{\text{max}} = P_{\text{peak}} / s$
- $\mu_{\text{ratio}} = s / p_{\text{peak}}$ (so $\mu \cdot p_{\text{peak}} = P_{\text{peak}}$)
- $\sigma_{\text{ratio}} = (1 - \mu_{\text{ratio}}) / z_{0.99}$ (so $\mu + z_{0.99}\sigma = P_{\text{max}}$)
- $p(\text{on}, t) = p_{\text{peak}} \cdot P(t) / P_{\text{peak}}$, split into 12 month factors and
  96 weekday/weekend interval factors

`--mu-ratio` and `--sigma-ratio` override the derived values. `--p-max` cannot be used here,
because $P_{\text{max}}$ is derived from the profile.

Input handling:

- **Units**: values count as kW. If the profile peak is above **20**, the values are treated as
  **W** and converted to kW, with a warning.
- **Resolution**: 15-, 30- and 60-minute data are supported. Coarser data is expanded to 15-minute intervals.
- **Columns**: `--value-column` picks the power column by name or index. The default is
  `consumption_kw` if that column exists, otherwise column 1.
- **Timezones**: by default, the wall-clock time as written in the file is used. With
  `--timezone Europe/Berlin`, timezone-aware timestamps are converted to that zone first, so the
  weekday and time of day are local.

Convert a W profile to a 15-minute kW CSV (timezone offsets are kept):

```bash
grid-gmm --convert-w-profile input_profiles/hp_archetype_terrace.csv out.csv --value-column archetype_w
```

### Configuration Files

```bash
grid-gmm --save-example-config config/example_input.json
grid-gmm --config config/example_input.json
```

Config files have a `schema_version`. The `derived` section (`mu_kw`, `sigma_kw`, `mean_active_kw`)
is informational only and is ignored when loading.

### Output Files

- `<name>.csv`: a plain CSV with one row per connection count
- `<name>.meta.json`: the component parameters, critical point, expected load and number of runs
- `<name>.png`: the percentile plot

## Reproducing the Committed Results

```bash
scripts/reproduce.sh
```

The script regenerates the converted input profiles in `input_profiles/` and every artifact in
`results/` that comes from them. `results/fitted_year_percentiles.*` comes from a profile that is
not in the repository, so the script cannot recreate it.

## Development

```bash
scripts/check.sh   # ruff lint + format check, mypy, pytest
pytest -v          # tests only (no PYTHONPATH needed)
```

## Package Layout

| Module | Purpose |
| --- | --- |
| `constants.py` | Time resolution, month names, `DayType`, z-quantile, W/kW threshold |
| `model.py` | `GaussianComponent` (immutable, validated, truncated mean, sampling) |
| `factors.py` | `FactorProfiles`, `CriticalPoint` |
| `simulation.py` | `MonteCarloEngine`, `SimulationResults` |
| `profile_fitter.py` | Year-profile loading, component derivation, factor decomposition, W→kW conversion |
| `config.py` | JSON config (de)serialization |
| `example_data.py` | Built-in example component and profiles |
| `reporting.py` | CSV/JSON export and console summary |
| `plotting.py` | Percentile plot |
| `cli.py` | Command-line interface |
