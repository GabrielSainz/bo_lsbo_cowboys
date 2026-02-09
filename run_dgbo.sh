#!/usr/bin/env bash
# cd C:/Users/gabri/OneDrive\ -\ University\ of\ Copenhagen/Escritorio/university_of_copenhagen/ku/block_6/project_outside_the_course_scope/experiments/BO_lsbo_COWBOYS/
# chmod +x run_dgbo.sh
# ./run_dgbo.sh
set -euo pipefail

tau="10.0"
outdir="toy_circle_data"
plotroot="${outdir}/plots_dgbo_tau${tau}"
mkdir -p "$plotroot"

python -u 07_dgbo_latent_diffusion.py \
  --outdir "$outdir" \
  --n_steps 60 \
  --n_init 24 \
  --weight pi \
  --xi 0.06 \
  --guide_every 1 \
  --n_cand 24 \
  --tau_guidance "$tau" \
  --clip_guidance 30 \
  >  "${plotroot}/stdout.log" \
  2> "${plotroot}/stderr.log"

echo "Done."
echo "stdout: ${plotroot}/stdout.log"
echo "stderr: ${plotroot}/stderr.log"
