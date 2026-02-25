#!/usr/bin/env bash
# cd "C:/Users/gabri/OneDrive - University of Copenhagen/Escritorio/university_of_copenhagen/ku/block_6/project_outside_the_course_scope/experiments/BO_lsbo_COWBOYS/"
# ./run_grid_dgbo.sh

set -euo pipefail

outdir="toy_circle_data"
mkdir -p "${outdir}"

# --- grid values ---
taus=(1.0 5.0 20.0 50.0 100.0)
guidance_scales=(0.5 1.0 2.0 5.0)
clips=(0.0 1.0 5.0 10.0 30.0)

# Optional: vary seed to check robustness quickly
seeds=(0)

# common args
n_steps=60
n_init=24
xi=0.06
weight="pi"
guide_every=1
n_cand=1

for seed in "${seeds[@]}"; do
  for tau in "${taus[@]}"; do
    for guidance_scale in "${guidance_scales[@]}"; do
      for clip_guidance in "${clips[@]}"; do

        tag="tau${tau}_gs${guidance_scale}_clip${clip_guidance}"
        plotroot="${outdir}/grid_dgbo_${tag}"
        mkdir -p "$plotroot"

        echo "=== RUN $tag ==="

        python -u 07_dgbo_latent_diffusion.py \
          --outdir "$outdir" \
          --seed "$seed" \
          --n_steps "$n_steps" \
          --n_init "$n_init" \
          --weight "$weight" \
          --xi "$xi" \
          --guide_every "$guide_every" \
          --n_cand "$n_cand" \
          --tau_guidance "$tau" \
          --guidance_scale "$guidance_scale" \
          --clip_guidance "$clip_guidance" \
          >  "${plotroot}/stdout.log" \
          2> "${plotroot}/stderr.log"

        echo "stdout: ${plotroot}/stdout.log"
        echo "stderr: ${plotroot}/stderr.log"
      done
    done
  done
done

echo "All runs done."
