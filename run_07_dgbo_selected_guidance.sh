#!/usr/bin/env bash
# Run two selected DGBO guidance configs across five seeds.
# Example:
#   chmod +x run_07_dgbo_selected_guidance.sh
#   SEEDS="1 2 3 4 5" M=200 ./run_07_dgbo_selected_guidance.sh
set -euo pipefail

outdir="${OUTDIR:-toy_circle_data}"
runs_root="${RUNS_ROOT:-results/toy_runs}"
diagnostics_root="${DIAGNOSTICS_ROOT:-results/toy_diagnostics_dgbo_selected}"

read -r -a seeds <<< "${SEEDS:-1 2 3 4 5}"

n_steps="${N_STEPS:-150}"
n_init="${N_INIT:-15}"
candidate_budget="${M:-200}"
n_cand="${N_CAND:-$candidate_budget}"
weight="${WEIGHT:-pi}"
select_acq="${SELECT_ACQ:-pi}"
xi="${XI:-0.06}"
guide_every="${GUIDE_EVERY:-1}"
candidate_batch_size="${CANDIDATE_BATCH_SIZE:-64}"
z_box="${Z_BOX:-5}"
grid_res="${GRID_RES:-140}"
plot_every="${PLOT_EVERY:-1}"
diagnostics_top_k="${DIAGNOSTICS_TOP_K:-10}"
diagnostics_background_res="${DIAGNOSTICS_BACKGROUND_RES:-60}"

# label:guidance_scale:tau_guidance:clip_guidance
configs=(
  "strong_default:2.0:20.0:30.0"
  "very_strong_high_tau:4.0:40.0:50.0"
)

for cfg in "${configs[@]}"; do
  IFS=: read -r label guidance_scale tau_guidance clip_guidance <<< "$cfg"
  tag="${label}_tau${tau_guidance}_gs${guidance_scale}_clip${clip_guidance}_cand${n_cand}"

  for seed in "${seeds[@]}"; do
    plotroot="${runs_root}/dgbo_selected_guidance/${tag}/seed_${seed}"
    mkdir -p "$plotroot"

    echo "=== DGBO selected guidance seed=${seed} ${tag} ==="
    if ! python -u 07_dgbo_latent_diffusion.py \
      --outdir "$outdir" \
      --plotroot "$plotroot" \
      --seed "$seed" \
      --n_steps "$n_steps" \
      --n_init "$n_init" \
      --weight "$weight" \
      --select_acq "$select_acq" \
      --xi "$xi" \
      --guide_every "$guide_every" \
      --n_cand "$n_cand" \
      --candidate_batch_size "$candidate_batch_size" \
      --tau_guidance "$tau_guidance" \
      --guidance_scale "$guidance_scale" \
      --clip_guidance "$clip_guidance" \
      --z_box "$z_box" \
      --grid_res "$grid_res" \
      --plot_every "$plot_every" \
      --diagnostics_root "$diagnostics_root" \
      --diagnostics_run_id "$tag" \
      --diagnostics_top_k "$diagnostics_top_k" \
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
echo
echo "Or plot this selected guidance comparison with:"
echo "python plot_dgbo_guidance_sweep.py --diagnostics_root ${diagnostics_root} --method_dir dgbo_latent_diffusion --runs_root ${runs_root} --run_group dgbo_selected_guidance --preset all"
