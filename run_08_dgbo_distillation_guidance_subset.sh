#!/usr/bin/env bash
# Sweep four DGBO-distillation guidance settings on one seed.
# Example:
#   chmod +x run_08_dgbo_distillation_guidance_subset.sh
#   SEED=1 M=200 ./run_08_dgbo_distillation_guidance_subset.sh
set -euo pipefail

outdir="${OUTDIR:-toy_circle_data}"
runs_root="${RUNS_ROOT:-results/toy_runs}"
diagnostics_root="${DIAGNOSTICS_ROOT:-results/toy_diagnostics_sweeps}"

seed="${SEED:-1}"
n_steps="${N_STEPS:-150}"
n_init="${N_INIT:-15}"
candidate_budget="${M:-200}"
n_cand="${N_CAND:-$candidate_budget}"
weight="${WEIGHT:-pi}"
select_acq="${SELECT_ACQ:-pi}"
xi="${XI:-0.06}"
guide_mode="${GUIDE_MODE:-distill}"
guide_every="${GUIDE_EVERY:-1}"
candidate_batch_size="${CANDIDATE_BATCH_SIZE:-64}"
distill_target_clip_low="${DISTILL_TARGET_CLIP_LOW:--30.0}"
distill_target_clip_high="${DISTILL_TARGET_CLIP_HIGH:-0.0}"
z_box="${Z_BOX:-5}"
grid_res="${GRID_RES:-140}"
plot_every="${PLOT_EVERY:-1}"
diagnostics_top_k="${DIAGNOSTICS_TOP_K:-10}"
diagnostics_background_res="${DIAGNOSTICS_BACKGROUND_RES:-60}"

# label:guide_mode:guidance_scale:tau_guidance:clip_guidance
configs=(
  "unguided:real:0.0:1.0:0.0"
  "soft_clipped:distill:0.5:10.0:5.0"
  "base_mid_clip:distill:1.0:20.0:30.0"
  "strong_default:distill:2.0:20.0:30.0"
)

for cfg in "${configs[@]}"; do
  IFS=: read -r label cfg_guide_mode guidance_scale tau_guidance clip_guidance <<< "$cfg"
  tag="${cfg_guide_mode}_${label}_tau${tau_guidance}_gs${guidance_scale}_clip${clip_guidance}_cand${n_cand}"
  plotroot="${runs_root}/dgbo_distillation_guidance_subset/${tag}/seed_${seed}"
  mkdir -p "$plotroot"

  echo "=== DGBO distillation guidance subset seed=${seed} ${tag} ==="
  if ! python -u 08_dgbo_latent_diffusion_distillation.py \
    --outdir "$outdir" \
    --plotroot "$plotroot" \
    --seed "$seed" \
    --n_steps "$n_steps" \
    --n_init "$n_init" \
    --weight "$weight" \
    --select_acq "$select_acq" \
    --xi "$xi" \
    --guide_mode "$cfg_guide_mode" \
    --guide_every "$guide_every" \
    --n_cand "$n_cand" \
    --candidate_batch_size "$candidate_batch_size" \
    --tau_guidance "$tau_guidance" \
    --guidance_scale "$guidance_scale" \
    --clip_guidance "$clip_guidance" \
    --distill_target_clip_low "$distill_target_clip_low" \
    --distill_target_clip_high "$distill_target_clip_high" \
    --z_box "$z_box" \
    --grid_res "$grid_res" \
    --plot_every "$plot_every" \
    --diagnostics_root "$diagnostics_root" \
    --diagnostics_run_id "$tag" \
    --diagnostics_top_k "$diagnostics_top_k" \
    --diagnostics_background_res "$diagnostics_background_res" \
    > "${plotroot}/stdout.log" \
    2> "${plotroot}/stderr.log"; then
    echo "ERROR: run failed for ${tag}"
    echo "---- stdout tail (${plotroot}/stdout.log) ----"
    tail -n 80 "${plotroot}/stdout.log" || true
    echo "---- stderr tail (${plotroot}/stderr.log) ----"
    tail -n 120 "${plotroot}/stderr.log" || true
    exit 1
  fi

  echo "saved: ${plotroot}"
done

echo "Done. Aggregate later with:"
echo "python aggregate_toy_diagnostics.py --results_root ${diagnostics_root} --representative_iterations 10 50 100 --make_gifs"
