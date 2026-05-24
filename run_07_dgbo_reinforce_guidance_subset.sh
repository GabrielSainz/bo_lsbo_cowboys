#!/usr/bin/env bash
# Sweep four DGBO-REINFORCE guidance settings on one seed.
# Example:
#   chmod +x run_07_dgbo_reinforce_guidance_subset.sh
#   SEED=1 M=200 ./run_07_dgbo_reinforce_guidance_subset.sh
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
guide_every="${GUIDE_EVERY:-1}"
z_box="${Z_BOX:-5}"
grid_res="${GRID_RES:-140}"

reinforce_k="${REINFORCE_K:-512}"
reinforce_sigma="${REINFORCE_SIGMA:-0.25}"
reinforce_baseline="${REINFORCE_BASELINE:-median}"
reinforce_logw_clip_low="${REINFORCE_LOGW_CLIP_LOW:--12}"
reinforce_logw_clip_high="${REINFORCE_LOGW_CLIP_HIGH:-0}"
reinforce_adv_clip="${REINFORCE_ADV_CLIP:-5}"
diagnostics_top_k="${DIAGNOSTICS_TOP_K:-10}"
diagnostics_background_res="${DIAGNOSTICS_BACKGROUND_RES:-60}"

# label:guidance_scale:tau_guidance:clip_guidance
configs=(
  "unguided:0.0:1.0:0.0"
  "soft_clipped:0.5:10.0:5.0"
  "base_mid_clip:1.0:20.0:30.0"
  "strong_default:2.0:20.0:30.0"
)

for cfg in "${configs[@]}"; do
  IFS=: read -r label guidance_scale tau_guidance clip_guidance <<< "$cfg"
  tag="${label}_tau${tau_guidance}_gs${guidance_scale}_clip${clip_guidance}_cand${n_cand}"
  plotroot="${runs_root}/dgbo_reinforce_guidance_subset/${tag}/seed_${seed}"
  mkdir -p "$plotroot"

  echo "=== DGBO REINFORCE guidance subset seed=${seed} ${tag} ==="
  python -u 07_dgbo_latent_diffusion_reinforce.py \
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
    --tau_guidance "$tau_guidance" \
    --guidance_scale "$guidance_scale" \
    --clip_guidance "$clip_guidance" \
    --reinforce_k "$reinforce_k" \
    --reinforce_sigma "$reinforce_sigma" \
    --reinforce_baseline "$reinforce_baseline" \
    --reinforce_logw_clip_low "$reinforce_logw_clip_low" \
    --reinforce_logw_clip_high "$reinforce_logw_clip_high" \
    --reinforce_adv_clip "$reinforce_adv_clip" \
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
