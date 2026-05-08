#!/usr/bin/env bash
set -euo pipefail

trap 'kill $(jobs -p) 2>/dev/null || true' EXIT INT TERM

PY=python
GEN=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/gen_from_zT_bank_multi_models-1_19.py

# models
SD14=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers
SD15=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers
SD21=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers

# prompts
PROMPT_DONGMAN=/home/yancy/work/dm_backdoor_latent_space/prompts/cal_dongman_female_align-jinghua-2026-1_25.txt
PROMPT_NUMBER=/home/yancy/work/dm_backdoor_latent_space/prompts/cal_number_align-2026_1_11.txt

# zT bank (lam1=0.87)
ZT_DONGMAN=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_PRC_w_att_dongman_0_87.pt
ZT_NUMBER=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_PRC_w_att_number_0_87.pt

# out root
OUTROOT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number

# common gen args
COMMON_ARGS=(--steps 50 --cfg 7.5 --height 512 --width 512 --n_per_prompt 4 --start_latent 0 --dtype fp16 --seed 12345 --negative_prompt "")

run_pair () {
  local model_tag="$1"
  local model_id="$2"

  # GPU1: dongman  (按你示例)
  local out_dongman="${OUTROOT}/vis_${model_tag}_PRC_w_att_0_87_dongman_seed12345"
  local log_dongman="${out_dongman}/gen.log"
  mkdir -p "${out_dongman}"
  CUDA_VISIBLE_DEVICES=1 ${PY} "${GEN}" \
    --model_id "${model_id}" \
    --prompts "${PROMPT_DONGMAN}" \
    --zT_pt "${ZT_DONGMAN}" \
    --outdir "${out_dongman}" \
    "${COMMON_ARGS[@]}" \
    > "${log_dongman}" 2>&1 &

  pid1=$!

  # GPU0: number (按你示例)
  local out_number="${OUTROOT}/vis_${model_tag}_PRC_w_att_0_87_number_seed12345"
  local log_number="${out_number}/gen.log"
  mkdir -p "${out_number}"
  CUDA_VISIBLE_DEVICES=0 ${PY} "${GEN}" \
    --model_id "${model_id}" \
    --prompts "${PROMPT_NUMBER}" \
    --zT_pt "${ZT_NUMBER}" \
    --outdir "${out_number}" \
    "${COMMON_ARGS[@]}" \
    > "${log_number}" 2>&1 &

  pid0=$!

  echo "[RUN] ${model_tag}: dongman(GPU1) pid=${pid1}"
  echo "[RUN] ${model_tag}: number(GPU0)  pid=${pid0}"

  # 等这一批两张卡都跑完，再进入下一模型（确保每卡最多1个任务）
  wait "${pid1}" "${pid0}"
  echo "[DONE] ${model_tag} finished."
}

echo "============================================================"
echo "[STAGE] PRC w_att lam1=0.87 | dongman+number | perGPU=1"
echo "============================================================"

run_pair "sd14" "${SD14}"
run_pair "sd15" "${SD15}"
run_pair "sd21" "${SD21}"

echo
echo "[ALL DONE] outputs under: ${OUTROOT}"
echo "  - vis_sd14_PRC_w_att_0_87_{dongman,number}_seed12345"
echo "  - vis_sd15_PRC_w_att_0_87_{dongman,number}_seed12345"
echo "  - vis_sd21_PRC_w_att_0_87_{dongman,number}_seed12345"
