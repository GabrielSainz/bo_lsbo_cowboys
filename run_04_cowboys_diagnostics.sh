#!/usr/bin/env bash
# Example:
#   chmod +x run_04_cowboys_diagnostics.sh
#   SEEDS="2 3 4" ./run_04_cowboys_diagnostics.sh
set -euo pipefail

outdir="${OUTDIR:-toy_circle_data}"
runs_root="${RUNS_ROOT:-results/toy_runs}"
diagnostics_root="${DIAGNOSTICS_ROOT:-results/toy_diagnostics_main}"

read -r -a seeds <<< "${SEEDS:-0}"

n_steps="${N_STEPS:-150}"
n_init="${N_INIT:-15}"
acq="${ACQ:-pi}"
xi="${XI:-0.00}"
kappa="${KAPPA:-2.2}"
tau="${TAU:-1.0}"
pcn_beta="${PCN_BETA:-0.35}"
pcn_steps="${PCN_STEPS:-2500}"
pcn_burn="${PCN_BURN:-800}"
pcn_thin="${PCN_THIN:-10}"
z_box="${Z_BOX:-5}"
grid_res="${GRID_RES:-140}"
n_data_scatter="${N_DATA_SCATTER:-4000}"
diagnostics_top_k="${DIAGNOSTICS_TOP_K:-10}"
diagnostics_background_res="${DIAGNOSTICS_BACKGROUND_RES:-60}"

for seed in "${seeds[@]}"; do
  plotroot="${runs_root}/cowboys/acq_${acq}_tau${tau}_beta${pcn_beta}/seed_${seed}"
  mkdir -p "$plotroot"

  echo "=== COWBOYS seed=${seed} acq=${acq} tau=${tau} beta=${pcn_beta} ==="
  python -u 04_cowboys.py \
    --outdir "$outdir" \
    --plotroot "$plotroot" \
    --seed "$seed" \
    --n_steps "$n_steps" \
    --n_init "$n_init" \
    --acq "$acq" \
    --xi "$xi" \
    --kappa "$kappa" \
    --tau "$tau" \
    --pcn_beta "$pcn_beta" \
    --pcn_steps "$pcn_steps" \
    --pcn_burn "$pcn_burn" \
    --pcn_thin "$pcn_thin" \
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
