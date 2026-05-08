#!/usr/bin/env bash
set -euo pipefail

PY="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/gen_from_zT_bank_multi_models-1_19.py"
PROMPTS="/home/yancy/work/dm_backdoor_latent_space/prompts/cal_dongman_female_align-jinghua-2026-1_25.txt"

# models
SD15="/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers"
SD21="/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers"

# common args
STEPS=50
CFG=7.5
H=512
W=512
N_PER=4
START_LATENT=0
DTYPE="fp16"
SEED=12345
NEG=""

# zT banks (dongman)
# --- TR ---
TR_W_ATT="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment-number/generate_TR_w_att_0_88_dongman.pt"
TR_W="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment-number/generate_TR_w.pt"
# --- GS ---
GS_W_ATT="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment-number/generate_GS_w_att_dongman.pt"
GS_W="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_GS_w.pt"
# --- T2S ---
T2S_W_ATT="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment-number/generate_T2S_w_att_dongman.pt"
T2S_W="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_T2S_w.pt"
# --- PRC ---
PRC_W_ATT="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment-number/generate_PRC_w_att_0_85_dongman.pt"
PRC_W="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_PRC_w_0_85.pt"

# out root + logs
OUTROOT="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number"
LOGDIR="${OUTROOT}/_logs"
mkdir -p "${LOGDIR}"

run_one () {
  local gpu="$1"
  local model_id="$2"
  local zt_pt="$3"
  local outdir="$4"

  CUDA_VISIBLE_DEVICES="${gpu}" python "${PY}" \
    --model_id "${model_id}" \
    --prompts "${PROMPTS}" \
    --zT_pt "${zt_pt}" \
    --outdir "${outdir}" \
    --steps "${STEPS}" --cfg "${CFG}" --height "${H}" --width "${W}" \
    --n_per_prompt "${N_PER}" --start_latent "${START_LATENT}" \
    --dtype "${DTYPE}" --seed "${SEED}" \
    --negative_prompt "${NEG}"
}

queue_one_model () {
  local gpu="$1"
  local model_tag="$2"     # sd15 or sd21
  local model_id="$3"

  # 你要的：每张卡排队生成（顺序）
  run_one "${gpu}" "${model_id}" "${TR_W_ATT}"  "${OUTROOT}/vis_${model_tag}_TR_w_att_0_88_dongman_seed${SEED}"
  run_one "${gpu}" "${model_id}" "${TR_W}"      "${OUTROOT}/vis_${model_tag}_TR_w_dongman_seed${SEED}"

  run_one "${gpu}" "${model_id}" "${GS_W_ATT}"  "${OUTROOT}/vis_${model_tag}_GS_w_att_dongman_seed${SEED}"
  run_one "${gpu}" "${model_id}" "${GS_W}"      "${OUTROOT}/vis_${model_tag}_GS_w_dongman_seed${SEED}"

  run_one "${gpu}" "${model_id}" "${T2S_W_ATT}" "${OUTROOT}/vis_${model_tag}_T2S_w_att_dongman_seed${SEED}"
  run_one "${gpu}" "${model_id}" "${T2S_W}"     "${OUTROOT}/vis_${model_tag}_T2S_w_dongman_seed${SEED}"

  run_one "${gpu}" "${model_id}" "${PRC_W_ATT}" "${OUTROOT}/vis_${model_tag}_PRC_w_att_0_85_dongman_seed${SEED}"
  run_one "${gpu}" "${model_id}" "${PRC_W}"     "${OUTROOT}/vis_${model_tag}_PRC_w_dongman_seed${SEED}"
}

echo "[INFO] Launch queues:"
echo "  GPU0 -> SD1.5 (TR/GS/T2S/PRC, w_att+w, dongman)"
echo "  GPU1 -> SD2.1 (TR/GS/T2S/PRC, w_att+w, dongman)"
nvidia-smi || true

# GPU0: SD1.5 队列（后台）
(
  queue_one_model 0 "sd15" "${SD15}"
) 2>&1 | tee "${LOGDIR}/dongman_allWM_sd15_gpu0.log" &
PID0=$!

# GPU1: SD2.1 队列（后台）
(
  queue_one_model 1 "sd21" "${SD21}"
) 2>&1 | tee "${LOGDIR}/dongman_allWM_sd21_gpu1.log" &
PID1=$!

echo "[INFO] PIDs: sd15=${PID0}, sd21=${PID1}"
wait "${PID0}" "${PID1}"

echo "[DONE] All done: dongman all-watermarks for sd15+sd21."
nvidia-smi || true
