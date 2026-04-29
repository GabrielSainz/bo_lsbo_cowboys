#!/usr/bin/env bash
# Example:
#   chmod +x run_08_dgbo_distillation_diagnostics.sh
#   SEEDS="0 1 2 3 4" N_STEPS=150 ./run_08_dgbo_distillation_diagnostics.sh
set -euo pipefail

outdir="${OUTDIR:-toy_circle_data}"
runs_root="${RUNS_ROOT:-results/toy_runs}"
diagnostics_root="${DIAGNOSTICS_ROOT:-results/toy_diagnostics_main}"

read -r -a seeds <<< "${SEEDS:-0}"

n_steps="${N_STEPS:-150}"
n_init="${N_INIT:-15}"
weight="${WEIGHT:-pi}"
select_acq="${SELECT_ACQ:-pi}"
xi="${XI:-0.06}"
guide_mode="${GUIDE_MODE:-distill}"
guide_every="${GUIDE_EVERY:-1}"
n_cand="${N_CAND:-1}"
tau_guidance="${TAU_GUIDANCE:-5.0}"
guidance_scale="${GUIDANCE_SCALE:-5.0}"
clip_guidance="${CLIP_GUIDANCE:-10.0}"
distill_target_clip_low="${DISTILL_TARGET_CLIP_LOW:--30.0}"
distill_target_clip_high="${DISTILL_TARGET_CLIP_HIGH:-0.0}"
z_box="${Z_BOX:-6}"
grid_res="${GRID_RES:-140}"
diagnostics_top_k="${DIAGNOSTICS_TOP_K:-10}"
diagnostics_background_res="${DIAGNOSTICS_BACKGROUND_RES:-60}"

for seed in "${seeds[@]}"; do
  tag="${guide_mode}_tau${tau_guidance}_gs${guidance_scale}_clip${clip_guidance}_cand${n_cand}"
  plotroot="${runs_root}/dgbo_latent_diffusion_distillation/${tag}/seed_${seed}"
  mkdir -p "$plotroot"

  echo "=== DGBO distillation seed=${seed} ${tag} ==="
  python -u 08_dgbo_latent_diffusion_distillation.py \
    --outdir "$outdir" \
    --plotroot "$plotroot" \
    --seed "$seed" \
    --n_steps "$n_steps" \
    --n_init "$n_init" \
    --weight "$weight" \
    --select_acq "$select_acq" \
    --xi "$xi" \
    --guide_mode "$guide_mode" \
    --guide_every "$guide_every" \
    --n_cand "$n_cand" \
    --tau_guidance "$tau_guidance" \
    --guidance_scale "$guidance_scale" \
    --clip_guidance "$clip_guidance" \
    --distill_target_clip_low "$distill_target_clip_low" \
    --distill_target_clip_high "$distill_target_clip_high" \
    --z_box "$z_box" \
    --grid_res "$grid_res" \
    --diagnostics_root "$diagnostics_root" \
    --diagnostics_top_k "$diagnostics_top_k" \
    --diagnostics_background_res "$diagnostics_background_res" \
    > "${plotroot}/stdout.log" \
    2> "${plotroot}/stderr.log"

  echo "saved: ${plotroot}"
done

echo "Done. Aggregate later with:"
echo "python aggregate_toy_diagnostics.py --results_root ${diagnostics_root} --representative_iterations 10 50 100 --make_gifs"
