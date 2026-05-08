#!/usr/bin/env bash
set -euo pipefail
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT INT TERM

PY=python
GEN=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/gen_from_zT_bank_multi_models-1_19.py

# ===== checkpoints =====
SD14=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers
SD15=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers
SD21=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers

# ===== inputs =====
PROMPTS=/home/yancy/work/dm_backdoor_latent_space/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt
ZT_PT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_PRC_w_att_0_87_delrepair.pt

# ===== outputs =====
OUTROOT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs
SEED=12345

COMMON_ARGS=(--steps 50 --cfg 7.5 --height 512 --width 512 --n_per_prompt 4 --start_latent 0 --dtype fp16 --seed ${SEED} --negative_prompt "")

run_one () {
  local tag="$1"     # sd14/sd15/sd21
  local model_id="$2"
  local gpu="$3"

  local outdir="${OUTROOT}/vis_${tag}_PRC_w_att_0_87_delrepair_seed${SEED}"
  local log="${outdir}/gen.log"
  mkdir -p "${outdir}"

  echo "[RUN] ${tag} on GPU${gpu} -> ${outdir}"
  CUDA_VISIBLE_DEVICES=${gpu} ${PY} "${GEN}" \
    --model_id "${model_id}" \
    --prompts "${PROMPTS}" \
    --zT_pt "${ZT_PT}" \
    --outdir "${outdir}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${log}"
}

echo "============================================================"
echo "[STAGE] PRC w_att 0.87 delrepair | sd14/sd15/sd21 | perGPU=1"
echo "============================================================"

# Batch 1: sd14 (GPU0) + sd15 (GPU1) 并行
run_one "sd14" "${SD14}" 0 &
p14=$!
run_one "sd15" "${SD15}" 1 &
p15=$!
wait "${p14}" "${p15}"
echo "[DONE] batch(sd14+sd15) finished."

# Batch 2: sd21 (GPU0) 单跑（也可以改成 GPU1，看你空闲哪张）
run_one "sd21" "${SD21}" 0

echo
echo "[ALL DONE] outputs under: ${OUTROOT}"
echo "  - vis_sd14_PRC_w_att_0_87_delrepair_seed${SEED}"
echo "  - vis_sd15_PRC_w_att_0_87_delrepair_seed${SEED}"
echo "  - vis_sd21_PRC_w_att_0_87_delrepair_seed${SEED}"
