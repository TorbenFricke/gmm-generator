#!/usr/bin/env bash
# Regenerate every derived input profile and every artifact in results/ from the raw data.
#
# Usage:  scripts/reproduce.sh            (uses .venv/bin/python if present, else python3)
#         PYTHON=/path/to/python scripts/reproduce.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi

grid_gmm() {
    echo
    echo ">>> grid-gmm $*"
    "$PYTHON" run_simulation.py "$@"
}

RAW_PROFILE=input_profiles/hp_archetype_terrace.csv
HP_PROFILE=input_profiles/hp_archetype_terrace_converted.csv
SEED=42

# --- 1. Derived input profiles (W -> kW, hourly -> 15 min) -------------------------------
grid_gmm --convert-w-profile "$RAW_PROFILE" "$HP_PROFILE" --value-column admd_scaled_w_per_hp
grid_gmm --convert-w-profile "$RAW_PROFILE" input_profiles/hp_archetype_terrace_archetype_raw_converted.csv \
    --value-column archetype_w

# --- 2. Built-in example (11 kW household connection) ------------------------------------
grid_gmm --seed "$SEED" --runs 200 \
    --csv results/grid_usage_percentiles.csv \
    --plot results/grid_usage_percentiles.png

# --- 3. Heat pump terrace archetype, simultaneity s = 0.7 --------------------------------
hp_fit() {
    grid_gmm --from-year-profile "$HP_PROFILE" --simultaneity 0.7 --seed "$SEED" "$@"
}

# Rusck match with p(on)_peak = 100 %
hp_fit --max-p-on 1.0 --runs 200 \
    --name "Heat Pump Terrace Archetype (Rusck Match)" \
    --save-config results/hp_terrace_fitted_config.json \
    --csv results/hp_terrace_simultaneity_0.7.csv \
    --plot results/hp_terrace_simultaneity_0.7.png

hp_fit --max-p-on 1.0 --runs 1000 \
    --name "Heat Pump Terrace Archetype (Rusck Match)" \
    --csv results/hp_terrace_simultaneity_0.7_p_on_1.0.csv \
    --plot results/hp_terrace_simultaneity_0.7_p_on_1.0.png

# p(on)_peak = 90 %. The committed artifact used an explicit sigma_ratio of 0.0889, not the
# Rusck default (1 - mu_ratio) / z_0.99 = 0.0955; it is pinned here to stay reproducible.
hp_fit --max-p-on 0.9 --sigma-ratio 0.0889 --runs 200 \
    --name "Heat Pump Terrace Archetype (p_on=90%)" \
    --save-config results/hp_terrace_fitted_config_p_on_0.9.json \
    --csv results/hp_terrace_simultaneity_0.7_p_on_0.9.csv \
    --plot results/hp_terrace_simultaneity_0.7_p_on_0.9.png

# Binary on/off load: always P_max when on, p(on)_peak = s = 70 %
hp_fit --max-p-on 0.7 --mu-ratio 1.0 --sigma-ratio 0 --runs 1000 \
    --name "Heat Pump Terrace Archetype (binary on/off)" \
    --csv results/hp_terrace_simultaneity_0.7_binary.csv \
    --plot results/hp_terrace_simultaneity_0.7_binary.png

echo
echo "All artifacts regenerated."
