#!/usr/bin/env bash
set -euo pipefail

PY="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/number_align_judge_single.py"
PROMPTS="/home/yancy/work/dm_backdoor_latent_space/prompts/cal_number_align-2026_1_11.txt"
IMG_GLOB="**/sliced/*.png"

# PRC: sd15 / sd21, w / w_att_0_85_number
D1="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_PRC_w_att_0_85_number_seed12345"
D2="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_PRC_w_number_seed12345"
D3="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_PRC_w_att_0_85_number_seed12345"
D4="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_PRC_w_number_seed12345"

LOGDIR="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/_logs"
mkdir -p "$LOGDIR"

run_one () {
  local img_root="$1"
  local tag="$2"
  python "$PY" \
    --img_root "$img_root" \
    --prompt_file "$PROMPTS" \
    --img_glob "$IMG_GLOB" \
    --workers 6 --retries 3 --max_tokens 8 \
  2>&1 | tee "$LOGDIR/${tag}.log"
}

# 4 jobs in parallel
run_one "$D1" "judge_sd15_PRC_w_att_0_85_number_seed12345" &
PID1=$!
run_one "$D2" "judge_sd15_PRC_w_number_seed12345" &
PID2=$!
run_one "$D3" "judge_sd21_PRC_w_att_0_85_number_seed12345" &
PID3=$!
run_one "$D4" "judge_sd21_PRC_w_number_seed12345" &
PID4=$!

echo "[INFO] Launched PIDs: $PID1 $PID2 $PID3 $PID4"
wait $PID1 $PID2 $PID3 $PID4
echo "[DONE] PRC sd15+sd21 all number_align_judge_single finished."
