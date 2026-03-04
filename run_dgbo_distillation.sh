#!/usr/bin/env bash
# cd C:/Users/gabri/OneDrive\ -\ University\ of\ Copenhagen/Escritorio/university_of_copenhagen/ku/block_6/project_outside_the_course_scope/experiments/BO_lsbo_COWBOYS/
# chmod +x run_dgbo_distillation.sh
# ./run_dgbo_distillation.sh
set -euo pipefail

tau="5.0" # 1
guidance_scale="2.0" # 1
clip_guidance="10.0" # 30
guide_mode="distill" # "real" or "distill"
distill_target_clip_low="-30.0"
distill_target_clip_high="0.0"

outdir="toy_circle_data"
plotroot="${outdir}/04_dgbo_runs_distillation/${guide_mode}_${tau}_gs${guidance_scale}_clip${clip_guidance}"
mkdir -p "$plotroot"

python -u 08_dgbo_latent_diffusion_distillation.py \
  --outdir "$outdir" \
  --plotroot "$plotroot" \
  --n_steps 20 \
  --n_init 24 \
  --weight "pi" \
  --select_acq "pi" \
  --xi 0.06 \
  --guide_every 1 \
  --n_cand 1 \
  --tau_guidance "$tau" \
  --guidance_scale "$guidance_scale" \
  --clip_guidance "$clip_guidance" \
  --guide_mode "$guide_mode" \
  --distill_target_clip_low "$distill_target_clip_low" \
  --distill_target_clip_high "$distill_target_clip_high" \
  >  "${plotroot}/stdout.log" \
  2> "${plotroot}/stderr.log"

echo "Done."
echo "stdout: ${plotroot}/stdout.log"
echo "stderr: ${plotroot}/stderr.log"
