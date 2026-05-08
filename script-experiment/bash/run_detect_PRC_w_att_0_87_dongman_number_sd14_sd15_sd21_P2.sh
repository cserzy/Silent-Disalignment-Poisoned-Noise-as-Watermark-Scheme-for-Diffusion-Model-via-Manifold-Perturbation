#!/usr/bin/env bash
set -euo pipefail
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT INT TERM

PY=python
PRC_PY=/home/yancy/work/dm_backdoor_latent_space/prc_detect_global_official_align-1_18_fixdim-meg.py

# ===== model checkpoints (你最新确认是 /home/yancy/work/...) =====
SD14=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers
SD15=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers
SD21=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers

# ===== run_dir root =====
IMGROOT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number

# ===== fixed args (match your command) =====
FIXED_ARGS=(--steps 50 --guidance 7.5 --dtype fp32 --var 1.5 --fpr 1e-2 --inv_bs 2 --debug)

run_pair_one_model () {
  local tag="$1"      # sd14/sd15/sd21
  local model_id="$2" # checkpoint dir

  # dongman on GPU1
  local rd_dongman="${IMGROOT}/vis_${tag}_PRC_w_att_0_87_dongman_seed12345"
  local log_dongman="${rd_dongman}/detect_prc_fpr1e-2_invbs2.log"
  test -d "${rd_dongman}/sliced" || { echo "[FATAL] Missing sliced: ${rd_dongman}/sliced" >&2; exit 2; }

  CUDA_VISIBLE_DEVICES=1 ${PY} "${PRC_PY}" \
    --run_dir "${rd_dongman}" \
    --model_id "${model_id}" \
    "${FIXED_ARGS[@]}" \
    2>&1 | tee "${log_dongman}" &
  pid1=$!

  # number on GPU0
  local rd_number="${IMGROOT}/vis_${tag}_PRC_w_att_0_87_number_seed12345"
  local log_number="${rd_number}/detect_prc_fpr1e-2_invbs2.log"
  test -d "${rd_number}/sliced" || { echo "[FATAL] Missing sliced: ${rd_number}/sliced" >&2; exit 2; }

  CUDA_VISIBLE_DEVICES=0 ${PY} "${PRC_PY}" \
    --run_dir "${rd_number}" \
    --model_id "${model_id}" \
    "${FIXED_ARGS[@]}" \
    2>&1 | tee "${log_number}" &
  pid0=$!

  echo "[RUN] ${tag} dongman(GPU1) pid=${pid1}  log=${log_dongman}"
  echo "[RUN] ${tag} number (GPU0) pid=${pid0}  log=${log_number}"

  # 每张卡最多 1 个：等这俩都完成，再跑下一模型
  wait "${pid1}" "${pid0}"
  echo "[DONE] ${tag} finished."
}

echo "============================================================"
echo "[STAGE] PRC detect | w_att_0_87 | dongman+number | perGPU=1"
echo "============================================================"

run_pair_one_model "sd14" "${SD14}"
run_pair_one_model "sd15" "${SD15}"
run_pair_one_model "sd21" "${SD21}"

echo
echo "[ALL DONE] PRC detect finished."
echo "Logs:"
echo "  ${IMGROOT}/vis_sd*_PRC_w_att_0_87_*_seed12345/detect_prc_fpr1e-2_invbs2.log"
