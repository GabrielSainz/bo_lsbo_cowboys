#!/usr/bin/env bash
# cd C:/Users/gabri/OneDrive\ -\ University\ of\ Copenhagen/Escritorio/university_of_copenhagen/ku/block_6/project_outside_the_course_scope/experiments/BO_lsbo_COWBOYS/
# chmod +x run_dgbo_reinforce.sh
# ./run_dgbo_reinforce.sh
set -euo pipefail

tau="20.0"
guidance_scale="2.0"
clip_guidance="10.0"

outdir="toy_circle_data"
plotroot="${outdir}/03_dgbo_reinforce/${tau}_gs${guidance_scale}_clip${clip_guidance}"
mkdir -p "$plotroot"

python -u 07_dgbo_latent_diffusion_reinforce.py \
  --outdir toy_circle_data \
  --plotroot "$plotroot" \
  --n_steps 60 \
  --n_init 24 \
  --n_cand 1 \
  --weight pi \
  --select_acq pi \
  --xi 0.06 \
  --guide_every 2 \
  --tau_guidance "$tau" \
  --guidance_scale "$guidance_scale" \
  --clip_guidance "$clip_guidance" \
  --reinforce_k 512 \
  --reinforce_sigma 0.25 \
  --reinforce_baseline median \
  --reinforce_logw_clip_low -12 \
  --reinforce_logw_clip_high 0 \
  --reinforce_adv_clip 5 \
  >  "${plotroot}/stdout.log" \
  2> "${plotroot}/stderr.log"

echo "Done."
echo "stdout: ${plotroot}/stdout.log"
echo "stderr: ${plotroot}/stderr.log"
