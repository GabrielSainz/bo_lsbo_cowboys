#!/usr/bin/env bash
# Example:
#   chmod +x run_04_cowboys_diagnostics.sh
#   SEEDS="1 2 3 4 5" M=200 N_CHAINS=4 ./run_04_cowboys_diagnostics.sh
set -euo pipefail

outdir="${OUTDIR:-toy_circle_data}"
runs_root="${RUNS_ROOT:-results/toy_runs}"
diagnostics_root="${DIAGNOSTICS_ROOT:-results/toy_diagnostics_main}"

read -r -a seeds <<< "${SEEDS:-1 2 3 4 5}"

n_steps="${N_STEPS:-150}"
n_init="${N_INIT:-15}"
candidate_budget="${M:-200}"
n_cand="${N_CAND:-$candidate_budget}"
acq="${ACQ:-pi}"
xi="${XI:-0.00}"
kappa="${KAPPA:-2.2}"
tau="${TAU:-1.0}"
n_chains="${N_CHAINS:-4}"
chain_init="${CHAIN_INIT:-mixed}"
pcn_beta="${PCN_BETA:-0.35}"
pcn_burn="${PCN_BURN:-0}"
pcn_thin="${PCN_THIN:-1}"
pcn_cand_per_chain="$(( (n_cand + n_chains - 1) / n_chains ))"
pcn_steps="${PCN_STEPS:-$(( pcn_burn + pcn_cand_per_chain * pcn_thin ))}"
z_box="${Z_BOX:-5}"
grid_res="${GRID_RES:-140}"
n_data_scatter="${N_DATA_SCATTER:-4000}"
diagnostics_top_k="${DIAGNOSTICS_TOP_K:-10}"
diagnostics_background_res="${DIAGNOSTICS_BACKGROUND_RES:-60}"

for seed in "${seeds[@]}"; do
  plotroot="${runs_root}/cowboys/acq_${acq}_tau${tau}_beta${pcn_beta}_m${n_cand}_chains${n_chains}/seed_${seed}"
  mkdir -p "$plotroot"

  echo "=== COWBOYS seed=${seed} acq=${acq} tau=${tau} beta=${pcn_beta} M=${n_cand} chains=${n_chains} ==="
  python -u 04_cowboys.py \
    --outdir "$outdir" \
    --plotroot "$plotroot" \
    --seed "$seed" \
    --n_steps "$n_steps" \
    --n_init "$n_init" \
    --n_cand "$n_cand" \
    --n_chains "$n_chains" \
    --chain_init "$chain_init" \
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
