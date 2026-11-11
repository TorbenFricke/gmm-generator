# Grid Usage Gaussian Modeling

A Python tool that simulates simultaneous household grid demand. Each connection is modelled by a
**single truncated Gaussian** active-load distribution combined with a time-dependent activation
probability $p(\text{on})$. (The package name `grid_gmm` is historical; Gaussian *mixtures* are not
implemented.)

## Model

- **Active load**: $P_{\text{active}} \sim \mathcal{N}(\mu, \sigma^2)$ truncated to $[0, P_{\max}]$, with
  $\mu = \mu_{\text{ratio}} P_{\max}$ and $\sigma = \sigma_{\text{ratio}} P_{\max}$.
  $\mu$ is the *location* of the underlying normal distribution; because of the truncation the
  expected active power $E[P_{\text{active}}]$ (reported as `E[active]`) differs from $\mu$.
- **Time profiles**: $p(\text{on}) = f_{\text{month}}(m) \cdot f_{\text{interval}}(t, \text{day type})$,
  clipped to $[0, 1]$, from 12 month factors and 96 15-minute interval factors for weekdays and weekends.
- **Critical point**: the $(m^*, t^*, \text{day type}^*)$ with the highest $p(\text{on})$ is found
  before the Monte Carlo simulation.
- **Monte Carlo**: for every connection count $N$ the number of active connections is
  $\text{Binomial}(N, p(\text{on}))$ and each active load is drawn from the truncated Gaussian.
- **Outputs**: percentiles (default 50/95/99) of total load and load per connection, the
  $N \to \infty$ expectation $E[P_{\text{active}}] \cdot p(\text{on})$, a CSV, a JSON metadata sidecar and a plot.

## Theory: Rusck's Simultaneity Model and the Gaussian Model

This section explains how the Gaussian model reproduces Rusck's simultaneity curve, how
$\mu$, $\sigma$ and $p$ shape it, and why the match only holds at the peak.

### Notation

All powers are expressed **in units of $P_{\max} = P_{\max}(1)$**, the maximum load of a single
connection, so $\mu$ and $\sigma$ are the `mu_ratio` and `sigma_ratio` of the code.

| Symbol | Meaning |
| --- | --- |
| $N$ | number of connections |
| $\mu,\ \sigma$ | location and scale of the active load (Gaussian truncated to $[0, 1]$) |
| $p(t)$ | activation probability $p(\text{on})$ at time step $t$; $p$ without argument is its value at the peak (`--max-p-on`) |
| $z$ | standard-normal quantile; $z_{0.99} \approx 2.326$ is the 99th percentile used throughout |
| $g(N)$ | simultaneity (Rusck): the maximum load of $N$ connections relative to the sum of their individual maxima, $g(N) = P_{\max}(N) / (N \cdot P_{\max}(1))$; evaluated at the peak, $g(N, t)$ at time step $t$ |
| $g_\infty$ | asymptotic simultaneity $\lim_{N \to \infty} g(N)$ at the peak (`--simultaneity`); $g_\infty(t)$ at time step $t$ |

The formulas below neglect truncation (they use $E[\text{load}] \approx \mu$ and
$\mathrm{Var} \approx \sigma^2$) and use the normal approximation for sums. The Monte Carlo
tables (40,000 runs per value) show the exact values.

### 1. Rusck's curve

$$
g(N) = g_\infty + \frac{1 - g_\infty}{\sqrt{N}}
$$

A single connection reaches $P_{\max}$ ($g(1) = 1$). Large groups approach $g_\infty$.

### 2. Simultaneity in the Gaussian model

At the peak, a connection is on with probability $p$ and then draws a load from the truncated Gaussian.
Otherwise its load is 0. Its mean is $p\mu$. Its variance has two parts:

$$
\mathrm{Var} = \underbrace{p \sigma^2}_{\text{spread while on}} + \underbrace{p(1-p) \mu^2}_{\text{on/off switching}} .
$$

The summed load of $N$ connections has no hard maximum below $N P_{\max}$, so the model takes the
99th percentile of the sum as the group maximum $P_{\max}(N)$. Dividing by $N P_{\max}(1) = N$ gives

$$
g(N) \approx p \mu + \frac{z_{0.99} \sqrt{p \sigma^2 + p(1-p) \mu^2}}{\sqrt{N}} .
$$

This has Rusck's shape: a constant plus a term that decays like $1/\sqrt{N}$. The two curves are
identical when

$$
p \mu = g_\infty \qquad\text{and}\qquad z_{0.99} \sqrt{p \sigma^2 + p(1-p) \mu^2} = 1 - g_\infty .
$$

### 3. Fitting at the peak

`--from-year-profile` sets

$$
\mu = \frac{g_\infty}{p}, \qquad \sigma = \frac{1 - \mu}{z_{0.99}}, \qquad P_{\max} = \frac{P_{\text{peak}}}{g_\infty} .
$$

The first condition, $p\mu = g_\infty$, always holds. Whether the second holds depends on $p$.

**$p = 1$: exact match.** No switching term remains, so $\mu = g_\infty$ and $z_{0.99} \sigma = 1 - g_\infty$.
Monte Carlo check for $g_\infty = 0.7$:

| $N$ | Rusck | Gaussian model |
| ---: | ---: | ---: |
| 1 | 1.000 | 0.966 |
| 2 | 0.912 | 0.892 |
| 5 | 0.834 | 0.824 |
| 10 | 0.795 | 0.788 |
| 50 | 0.742 | 0.737 |
| 200 | 0.721 | 0.717 |

The small gap for small $N$ comes from the truncation at $P_{\max}$.

**$p < 1$: only the asymptote matches.** The switching variance makes the $1/\sqrt{N}$ term
larger than $1 - g_\infty$. Even with $\sigma = 0$, the switching term alone exceeds $1 - g_\infty$ unless

$$
p \ge \frac{1}{1 + \left(\dfrac{1 - g_\infty}{z_{0.99} g_\infty}\right)^2} \qquad (\approx 0.967 \text{ for } g_\infty = 0.7) .
$$

Monte Carlo check for $g_\infty = 0.7$ and $p = 0.9$:

| $N$ | Rusck | Gaussian model |
| ---: | ---: | ---: |
| 1 | 1.000 | 0.971 |
| 10 | 0.795 | 0.831 |
| 50 | 0.742 | 0.771 |
| 200 | 0.721 | 0.736 |

### 4. Effect of $\mu$, $\sigma$ and $p$

| Parameter | Effect on $g_\infty = p \mu$ | Effect on the $1/\sqrt{N}$ term |
| --- | --- | --- |
| $\mu$ | sets the level | grows through the switching term $p(1-p)\mu^2$ |
| $\sigma$ | none | grows; sets how far small groups lie above the asymptote |
| $p$ | lowers it linearly | adds the switching variance, largest at $p = 1/2$ |

In short, $\mu$ and $p$ set the simultaneity of large groups. $\sigma$ and the on/off switching
set how much higher small groups go.

### 5. Away from the peak, the simultaneity is larger

The model uses the same $\mu$ and $\sigma$ at every time step. Only the activation probability
falls with the profile:

$$
p(t) = p \frac{P(t)}{P_{\text{peak}}} \le p .
$$

Fewer connections are on, but each one that is on still draws the full $\mu$. Fluctuations
relative to the mean therefore grow. To compare time steps, divide the simultaneity by that time
step's own asymptote $g_\infty(t) = p(t) \mu$:

$$
\frac{g(N, t)}{g_\infty(t)} \approx 1 + \frac{z_{0.99}}{\sqrt{N}} \sqrt{\frac{1 + \sigma^2/\mu^2}{p(t)} - 1} .
$$

This ratio decreases strictly as $p(t)$ grows. It is therefore smallest at the peak. For $p = 1$
it equals Rusck's $g(N)/g_\infty = 1 + \frac{1 - g_\infty}{g_\infty\sqrt{N}}$ there, and it is larger at every other
time step.

Monte Carlo check of $g(N,t)/g_\infty(t)$ with $g_\infty = 0.7$ and $p = 1$:

| $P(t)/P_{\text{peak}}$ | $N = 10$ | $N = 50$ | $N = 200$ |
| ---: | ---: | ---: | ---: |
| 1.00 (peak, = Rusck) | 1.13 | 1.06 | 1.03 |
| 0.50 | 1.75 | 1.34 | 1.17 |
| 0.25 | 2.44 | 1.61 | 1.30 |

**What this means:** the fitted model reproduces Rusck's simultaneity only at the critical time
step. At every other time step, a group of $N$ connections has a higher peak relative to its
mean demand than Rusck's curve predicts. Scaling the peak-time result down by
$P(t)/P_{\text{peak}}$ therefore underestimates off-peak group peaks. The simulation evaluates
only the critical time step, where the match holds by construction.

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

With target asymptotic simultaneity $g_\infty$ and peak activation probability $p$ (`--max-p-on`):

- $P_{\max} = P_{\text{peak}} / g_\infty$
- $\mu_{\text{ratio}} = g_\infty / p$ (so $\mu \cdot p = P_{\text{peak}}$)
- $\sigma_{\text{ratio}} = (1 - \mu_{\text{ratio}}) / z_{0.99}$ (so $\mu + z_{0.99}\sigma = P_{\max}$)
- $p(t) = p \cdot P(t) / P_{\text{peak}}$, split into 12 month factors and
  96 weekday/weekend interval factors

See [Theory](#theory-ruscks-simultaneity-model-and-the-gaussian-model) for why these choices
reproduce Rusck's curve exactly only when `--max-p-on 1.0`. `--mu-ratio` and `--sigma-ratio` override the derived values. `--p-max` cannot be used here,
because $P_{\max}$ is derived from the profile.

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
`results/` that comes from them. This includes the electric cooking GM from
`scripts/generate_cooking_gm.py`: a fixed daily hourly profile repeated over a year with
$P_{\max} = 3$ kW (so $g_\infty = P_{\text{peak}} / P_{\max} \approx 0.05$) and $p = 0.1$. Its $\mu$ fixes
the mean, and its $\sigma$ is calibrated against Rusck's curve. See [docs/electric_cooking_gm.md](docs/electric_cooking_gm.md) for how these values were derived. `results/fitted_year_percentiles.*` comes from a profile that is
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
