#!/usr/bin/env bash
# Example:
#   chmod +x run_03_lsbo_diagnostics.sh
#   SEEDS="0 1 2 3 4" ./run_03_lsbo_diagnostics.sh
set -euo pipefail

outdir="${OUTDIR:-toy_circle_data}"
runs_root="${RUNS_ROOT:-results/toy_runs}"
diagnostics_root="${DIAGNOSTICS_ROOT:-results/toy_diagnostics_main}"

read -r -a seeds <<< "${SEEDS:-0}"

n_steps="${N_STEPS:-150}"
n_init="${N_INIT:-15}"
xi="${XI:-0.12}"
eps_random="${EPS_RANDOM:-0.25}"
topk_ei="${TOPK_EI:-30}"
z_box="${Z_BOX:-3}"
grid_res="${GRID_RES:-140}"
n_data_scatter="${N_DATA_SCATTER:-3000}"
diagnostics_top_k="${DIAGNOSTICS_TOP_K:-10}"
diagnostics_background_res="${DIAGNOSTICS_BACKGROUND_RES:-60}"

for seed in "${seeds[@]}"; do
  plotroot="${runs_root}/lsbo/seed_${seed}"
  mkdir -p "$plotroot"

  echo "=== LSBO seed=${seed} ==="
  python -u 03_lsbo.py \
    --outdir "$outdir" \
    --plotroot "$plotroot" \
    --seed "$seed" \
    --n_steps "$n_steps" \
    --n_init "$n_init" \
    --xi "$xi" \
    --eps_random "$eps_random" \
    --topk_ei "$topk_ei" \
    --z_box "$z_box" \
    --grid_res "$grid_res" \
    --n_data_scatter "$n_data_scatter" \
    --diagnostics_root "$diagnostics_root" \
    --diagnostics_top_k "$diagnostics_top_k" \
    --diagnostics_background_res "$diagnostics_background_res" \
    > "${plotroot}/stdout.log" \
    2> "${plotroot}/stderr.log"

  echo "saved: ${plotroot}"
done

echo "Done. Aggregate later with:"
echo "python aggregate_toy_diagnostics.py --results_root ${diagnostics_root} --representative_iterations 10 50 100 --make_gifs"
