#!/usr/bin/env bash
set -euo pipefail
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT INT TERM

PY=python
PRC_PY=/home/yancy/work/dm_backdoor_latent_space/prc_detect_global_official_align-1_18_fixdim-meg.py

SD15=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers
SD21=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers

IMGROOT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number
FIXED_ARGS=(--steps 50 --guidance 7.5 --dtype fp32 --var 1.5 --fpr 1e-2 --inv_bs 2 --debug)

run_one () {
  local tag="$1"      # sd15/sd21
  local model_id="$2"
  local gpu="$3"

  local rd="${IMGROOT}/vis_${tag}_PRC_w_att_0_87_dongman_seed12345"
  local log="${rd}/detect_prc_fpr1e-2_invbs2.log"

  test -d "${rd}/sliced" || { echo "[FATAL] Missing sliced: ${rd}/sliced" >&2; exit 2; }

  echo "[RUN] ${tag} dongman on GPU${gpu} -> ${rd}"
  CUDA_VISIBLE_DEVICES=${gpu} ${PY} "${PRC_PY}" \
    --run_dir "${rd}" \
    --model_id "${model_id}" \
    "${FIXED_ARGS[@]}" \
    2>&1 | tee "${log}"
}

echo "============================================================"
echo "[STAGE] PRC detect | w_att_0_87 | dongman only | sd15+sd21"
echo "============================================================"

run_one "sd15" "${SD15}" 1 &
p15=$!
run_one "sd21" "${SD21}" 0 &
p21=$!

echo "[WAIT] pids: ${p15} ${p21}"
wait "${p15}" "${p21}"

echo
echo "[DONE] finished sd15+sd21 dongman."
echo "Logs:"
echo "  ${IMGROOT}/vis_sd15_PRC_w_att_0_87_dongman_seed12345/detect_prc_fpr1e-2_invbs2.log"
echo "  ${IMGROOT}/vis_sd21_PRC_w_att_0_87_dongman_seed12345/detect_prc_fpr1e-2_invbs2.log"
