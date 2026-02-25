#!/usr/bin/env bash
# cd C:/Users/gabri/OneDrive\ -\ University\ of\ Copenhagen/Escritorio/university_of_copenhagen/ku/block_6/project_outside_the_course_scope/experiments/BO_lsbo_COWBOYS/
# chmod +x run_dgbo.sh
# ./run_dgbo.sh
set -euo pipefail

tau="1.0"
guidance_scale="1.0"
clip_guidance="5.0"

outdir="toy_circle_data"
plotroot="${outdir}/03_dgbo_runs/${tau}_gs${guidance_scale}_clip${clip_guidance}"
mkdir -p "$plotroot"

python -u 07_dgbo_latent_diffusion.py \
  --outdir "$outdir" \
  --n_steps 60 \
  --n_init 24 \
  --weight pi \
  --xi 0.06 \
  --guide_every 1 \
  --n_cand 1 \
  --tau_guidance "$tau" \
  --guidance_scale "$guidance_scale" \
  --clip_guidance "$clip_guidance" \
  >  "${plotroot}/stdout.log" \
  2> "${plotroot}/stderr.log"

echo "Done."
echo "stdout: ${plotroot}/stdout.log"
echo "stderr: ${plotroot}/stderr.log"
