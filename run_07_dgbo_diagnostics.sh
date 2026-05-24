#!/usr/bin/env bash
# Example:
#   chmod +x run_07_dgbo_diagnostics.sh
#   SEEDS="1 2 3 4 5" M=200 ./run_07_dgbo_diagnostics.sh
set -euo pipefail

outdir="${OUTDIR:-toy_circle_data}"
runs_root="${RUNS_ROOT:-results/toy_runs}"
diagnostics_root="${DIAGNOSTICS_ROOT:-results/toy_diagnostics_main}"

read -r -a seeds <<< "${SEEDS:-1 2 3 4 5}"

n_steps="${N_STEPS:-150}"
n_init="${N_INIT:-15}"
candidate_budget="${M:-200}"
weight="${WEIGHT:-pi}"
select_acq="${SELECT_ACQ:-pi}"
xi="${XI:-0.06}"
guide_every="${GUIDE_EVERY:-1}"
n_cand="${N_CAND:-$candidate_budget}"
candidate_batch_size="${CANDIDATE_BATCH_SIZE:-64}"
tau_guidance="${TAU_GUIDANCE:-20.0}"
guidance_scale="${GUIDANCE_SCALE:-2.0}"
clip_guidance="${CLIP_GUIDANCE:-30.0}"
z_box="${Z_BOX:-5}"
grid_res="${GRID_RES:-140}"
plot_every="${PLOT_EVERY:-0}"
diagnostics_top_k="${DIAGNOSTICS_TOP_K:-10}"
diagnostics_background_res="${DIAGNOSTICS_BACKGROUND_RES:-60}"

for seed in "${seeds[@]}"; do
  tag="tau${tau_guidance}_gs${guidance_scale}_clip${clip_guidance}_cand${n_cand}"
  plotroot="${runs_root}/dgbo_latent_diffusion/${tag}/seed_${seed}"
  mkdir -p "$plotroot"

  echo "=== DGBO seed=${seed} ${tag} ==="
  python -u 07_dgbo_latent_diffusion.py \
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
    --diagnostics_top_k "$diagnostics_top_k" \
    --diagnostics_background_res "$diagnostics_background_res" \
    > "${plotroot}/stdout.log" \
    2> "${plotroot}/stderr.log"

  echo "saved: ${plotroot}"
done

echo "Done. Aggregate later with:"
echo "python aggregate_toy_diagnostics.py --results_root ${diagnostics_root} --representative_iterations 10 50 100 --make_gifs"
