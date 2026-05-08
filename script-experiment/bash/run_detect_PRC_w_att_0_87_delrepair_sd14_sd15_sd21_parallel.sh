#!/usr/bin/env bash
set -euo pipefail
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT INT TERM

PY=python
PRC_PY=/home/yancy/work/dm_backdoor_latent_space/prc_detect_global_official_align-1_18_fixdim-meg.py

# checkpoints
SD14=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers
SD15=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers
SD21=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers

# run_dir root
IMGROOT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs
SEED=12345

# fixed args (match your sample)
FIXED_ARGS=(--steps 50 --guidance 7.5 --dtype fp32 --var 1.5 --fpr 1e-2 --inv_bs 2 --debug)

run_one () {
  local tag="$1"      # sd14/sd15/sd21
  local model_id="$2"
  local gpu="$3"

  local rd="${IMGROOT}/vis_${tag}_PRC_w_att_0_87_delrepair_seed${SEED}"
  local log="${rd}/detect_prc_fpr1e-2_invbs2.log"

  test -d "${rd}/sliced" || { echo "[FATAL] Missing sliced: ${rd}/sliced" >&2; exit 2; }

  echo "[RUN] ${tag} on GPU${gpu} -> ${rd}"
  CUDA_VISIBLE_DEVICES=${gpu} ${PY} "${PRC_PY}" \
    --run_dir "${rd}" \
    --model_id "${model_id}" \
    "${FIXED_ARGS[@]}" \
    2>&1 | tee "${log}"
}

echo "============================================================"
echo "[STAGE] PRC detect | w_att_0_87 delrepair | sd14/sd15/sd21 | perGPU=1"
echo "============================================================"

# Batch 1: sd14 (GPU0) + sd15 (GPU1) 并行
run_one "sd14" "${SD14}" 0 &
p14=$!
run_one "sd15" "${SD15}" 1 &
p15=$!
echo "[WAIT] batch1 pids: ${p14} ${p15}"
wait "${p14}" "${p15}"
echo "[DONE] batch1 finished."

# Batch 2: sd21 单跑（GPU0；你想放 GPU1 也行）
run_one "sd21" "${SD21}" 0
echo "[DONE] batch2 finished."

echo
echo "[ALL DONE] PRC delrepair detect finished."
echo "Logs:"
echo "  ${IMGROOT}/vis_sd14_PRC_w_att_0_87_delrepair_seed${SEED}/detect_prc_fpr1e-2_invbs2.log"
echo "  ${IMGROOT}/vis_sd15_PRC_w_att_0_87_delrepair_seed${SEED}/detect_prc_fpr1e-2_invbs2.log"
echo "  ${IMGROOT}/vis_sd21_PRC_w_att_0_87_delrepair_seed${SEED}/detect_prc_fpr1e-2_invbs2.log"
