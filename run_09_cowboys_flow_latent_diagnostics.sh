#!/usr/bin/env bash
# Example:
#   chmod +x run_09_cowboys_flow_latent_diagnostics.sh
#   SEEDS="0 1 2 3 4" N_STEPS=150 ./run_09_cowboys_flow_latent_diagnostics.sh
set -euo pipefail

outdir="${OUTDIR:-toy_circle_data}"
runs_root="${RUNS_ROOT:-results/toy_runs}"
diagnostics_root="${DIAGNOSTICS_ROOT:-results/toy_diagnostics_main}"

read -r -a seeds <<< "${SEEDS:-0}"

n_steps="${N_STEPS:-150}"
n_init="${N_INIT:-15}"
init_mode="${INIT_MODE:-mix}"
weight="${WEIGHT:-pi}"
select_acq="${SELECT_ACQ:-pi}"
plot_acq="${PLOT_ACQ:-pi}"
xi="${XI:-0.06}"
kappa="${KAPPA:-2.2}"
beta_tilt="${BETA_TILT:-1.0}"
tau_temp="${TAU_TEMP:-0.1}"
mh_steps="${MH_STEPS:-1000}"
mh_burn="${MH_BURN:-300}"
mh_thin="${MH_THIN:-5}"
lambda_local="${LAMBDA_LOCAL:-0.45}"
sigma_local="${SIGMA_LOCAL:-0.2}"
pool_prior="${POOL_PRIOR:-64}"
pool_replay="${POOL_REPLAY:-256}"
pool_local="${POOL_LOCAL:-256}"
pool_local_scale="${POOL_LOCAL_SCALE:-0.12}"
flow_train_steps="${FLOW_TRAIN_STEPS:-250}"
grid_res="${GRID_RES:-110}"
diagnostics_top_k="${DIAGNOSTICS_TOP_K:-10}"
diagnostics_z_box="${DIAGNOSTICS_Z_BOX:-4}"
diagnostics_background_res="${DIAGNOSTICS_BACKGROUND_RES:-60}"

for seed in "${seeds[@]}"; do
  tag="${weight}_bt${beta_tilt}_tau${tau_temp}_init${init_mode}_mh${mh_steps}"
  plotroot="${runs_root}/cowboys_flow_latent/${tag}/seed_${seed}"
  mkdir -p "$plotroot"

  echo "=== COWBOYS flow latent seed=${seed} ${tag} ==="
  python -u 09_cowboys_flow_latent.py \
    --outdir "$outdir" \
    --plotroot "$plotroot" \
    --seed "$seed" \
    --n_steps "$n_steps" \
    --n_init "$n_init" \
    --init_mode "$init_mode" \
    --weight "$weight" \
    --select_acq "$select_acq" \
    --plot_acq "$plot_acq" \
    --xi "$xi" \
    --kappa "$kappa" \
    --beta_tilt "$beta_tilt" \
    --tau_temp "$tau_temp" \
    --mh_steps "$mh_steps" \
    --mh_burn "$mh_burn" \
    --mh_thin "$mh_thin" \
    --lambda_local "$lambda_local" \
    --sigma_local "$sigma_local" \
    --pool_prior "$pool_prior" \
    --pool_replay "$pool_replay" \
    --pool_local "$pool_local" \
    --pool_local_scale "$pool_local_scale" \
    --flow_train_steps "$flow_train_steps" \
    --latent_xlim -4 4 \
    --latent_ylim -4 4 \
    --grid_res "$grid_res" \
    --diagnostics_root "$diagnostics_root" \
    --diagnostics_top_k "$diagnostics_top_k" \
    --diagnostics_z_box "$diagnostics_z_box" \
    --diagnostics_background_res "$diagnostics_background_res" \
    > "${plotroot}/stdout.log" \
    2> "${plotroot}/stderr.log"

  echo "saved: ${plotroot}"
done

echo "Done. Aggregate later with:"
echo "python aggregate_toy_diagnostics.py --results_root ${diagnostics_root} --representative_iterations 10 50 100 --make_gifs"
