#!/usr/bin/env bash
set -euo pipefail
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT INT TERM

PY=python
JUDGE=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/number_align_judge_single.py
PROMPTS=/home/yancy/work/dm_backdoor_latent_space/prompts/cal_number_align-2026_1_11.txt
IMG_GLOB="**/sliced/*.png"

# number 主题：PRC w_att 0.87
RD14=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_PRC_w_att_0_87_number_seed12345
RD15=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_PRC_w_att_0_87_number_seed12345
RD21=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_PRC_w_att_0_87_number_seed12345

run_one () {
  local tag="$1"
  local rd="$2"
  local log="${rd}/number_align_judge_single.log"

  test -d "${rd}" || { echo "[FATAL] missing run_dir: ${rd}" >&2; exit 2; }

  echo "[RUN] ${tag} -> ${rd}"
  ${PY} "${JUDGE}" \
    --img_root "${rd}" \
    --prompt_file "${PROMPTS}" \
    --img_glob "${IMG_GLOB}" \
    --workers 6 --retries 3 --max_tokens 8 \
    2>&1 | tee "${log}"
}

# 三模型并行（不占显卡）
run_one "sd14" "${RD14}" &
p14=$!
run_one "sd15" "${RD15}" &
p15=$!
run_one "sd21" "${RD21}" &
p21=$!

echo "[WAIT] pids: ${p14} ${p15} ${p21}"
wait "${p14}" "${p15}" "${p21}"

echo "[DONE] all number-align scoring finished."
echo "Logs:"
echo "  ${RD14}/number_align_judge_single.log"
echo "  ${RD15}/number_align_judge_single.log"
echo "  ${RD21}/number_align_judge_single.log"
