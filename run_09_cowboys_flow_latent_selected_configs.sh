#!/usr/bin/env bash
# Run two selected COWBOYS flow-latent proposal configs across five seeds.
# Example:
#   chmod +x run_09_cowboys_flow_latent_selected_configs.sh
#   SEEDS="1 2 3 4 5" M=200 ./run_09_cowboys_flow_latent_selected_configs.sh
set -euo pipefail

outdir="${OUTDIR:-toy_circle_data}"
runs_root="${RUNS_ROOT:-results/toy_runs}"
diagnostics_root="${DIAGNOSTICS_ROOT:-results/toy_diagnostics_selected_flow}"

read -r -a seeds <<< "${SEEDS:-1 2 3 4 5}"

n_steps="${N_STEPS:-150}"
n_init="${N_INIT:-15}"
candidate_budget="${M:-200}"
init_mode="${INIT_MODE:-mix}"
weight="${WEIGHT:-pi}"
select_acq="${SELECT_ACQ:-pi}"
plot_acq="${PLOT_ACQ:-pi}"
xi="${XI:-0.06}"
kappa="${KAPPA:-2.2}"
mh_burn="${MH_BURN:-0}"
mh_thin="${MH_THIN:-1}"
mh_steps="${MH_STEPS:-$(( mh_burn + candidate_budget * mh_thin ))}"
pool_replay="${POOL_REPLAY:-256}"
flow_train_steps="${FLOW_TRAIN_STEPS:-250}"
grid_res="${GRID_RES:-110}"
diagnostics_top_k="${DIAGNOSTICS_TOP_K:-10}"
diagnostics_z_box="${DIAGNOSTICS_Z_BOX:-5}"
diagnostics_background_res="${DIAGNOSTICS_BACKGROUND_RES:-60}"

# label:lambda_local:sigma_local:pool_prior:pool_local:pool_local_scale:beta_tilt:tau_temp
configs=(
  "soft_global:0.20:0.20:128:256:0.12:0.5:0.20"
  "balanced_base:0.45:0.20:64:256:0.12:1.0:0.10"
)

for cfg in "${configs[@]}"; do
  IFS=: read -r label lambda_local sigma_local pool_prior pool_local pool_local_scale beta_tilt tau_temp <<< "$cfg"
  tag="${label}_${weight}_bt${beta_tilt}_tau${tau_temp}_lam${lambda_local}_sig${sigma_local}_m${candidate_budget}_mh${mh_steps}"

  for seed in "${seeds[@]}"; do
    plotroot="${runs_root}/cowboys_flow_latent_selected_configs/${tag}/seed_${seed}"
    mkdir -p "$plotroot"

    echo "=== COWBOYS flow selected config seed=${seed} ${tag} ==="
    if ! python -u 09_cowboys_flow_latent.py \
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
      --diagnostics_run_id "$tag" \
      --diagnostics_top_k "$diagnostics_top_k" \
      --diagnostics_z_box "$diagnostics_z_box" \
      --diagnostics_background_res "$diagnostics_background_res" \
      > "${plotroot}/stdout.log" \
      2> "${plotroot}/stderr.log"; then
      echo "ERROR: run failed for seed=${seed} ${tag}"
      echo "---- stdout tail (${plotroot}/stdout.log) ----"
      tail -n 80 "${plotroot}/stdout.log" || true
      echo "---- stderr tail (${plotroot}/stderr.log) ----"
      tail -n 120 "${plotroot}/stderr.log" || true
      exit 1
    fi

    echo "saved: ${plotroot}"
  done
done

echo "Done. Aggregate later with:"
echo "python aggregate_toy_diagnostics.py --results_root ${diagnostics_root} --representative_iterations 10 50 100 --make_gifs"
