#!/usr/bin/env bash
set -euo pipefail

PY="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/number_align_judge_single.py"
PROMPTS="/home/yancy/work/dm_backdoor_latent_space/prompts/cal_number_align-2026_1_11.txt"
IMG_GLOB="**/sliced/*.png"

# sd15
python "$PY" \
  --img_root /home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_TR_w_att_0_88_number_seed12345 \
  --prompt_file "$PROMPTS" \
  --img_glob "$IMG_GLOB" \
  --workers 6 --retries 3 --max_tokens 8 &

python "$PY" \
  --img_root /home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_TR_w_number_seed12345 \
  --prompt_file "$PROMPTS" \
  --img_glob "$IMG_GLOB" \
  --workers 6 --retries 3 --max_tokens 8 &

# sd21
python "$PY" \
  --img_root /home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_TR_w_att_0_88_number_seed12345 \
  --prompt_file "$PROMPTS" \
  --img_glob "$IMG_GLOB" \
  --workers 6 --retries 3 --max_tokens 8 &

python "$PY" \
  --img_root /home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_TR_w_number_seed12345 \
  --prompt_file "$PROMPTS" \
  --img_glob "$IMG_GLOB" \
  --workers 6 --retries 3 --max_tokens 8 &

wait
echo "[DONE] sd15+sd21 all number_align_judge_vote finished."
