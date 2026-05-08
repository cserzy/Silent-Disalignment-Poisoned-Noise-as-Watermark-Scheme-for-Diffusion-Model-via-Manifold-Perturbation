#!/usr/bin/env bash
set -euo pipefail

# ========== Common args ==========
PY="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/gen_from_zT_bank_multi_models-1_19.py"
PROMPTS="/home/yancy/work/dm_backdoor_latent_space/prompts/cal_number_align-2026_1_11.txt"

ZT_W_ATT="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment-number/generate_TR_w_att_0_88_number.pt"
ZT_W="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment-number/generate_TR_w.pt"

STEPS=50
CFG=7.5
H=512
W=512
N_PER=4
START_LATENT=0
DTYPE="fp16"
SEED=12345
NEG=""

# ========== Model checkpoints ==========
SD15="/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers"
SD21="/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers"

# ========== Output dirs ==========
OUT_SD15_WATT="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_TR_w_att_0_88_number_seed12345"
OUT_SD15_W="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_TR_w_number_seed12345"

OUT_SD21_WATT="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_TR_w_att_0_88_number_seed12345"
OUT_SD21_W="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_TR_w_number_seed12345"

LOGDIR="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/_logs"
mkdir -p "${LOGDIR}"

echo "[INFO] Starting parallel generation on 2 GPUs..."
echo "[INFO] GPU0 -> SD1.5 (w_att then w)"
echo "[INFO] GPU1 -> SD2.1 (w_att then w)"
nvidia-smi || true

run_one () {
  local gpu="$1"
  local model_id="$2"
  local zt_pt="$3"
  local outdir="$4"
  local tag="$5"

  echo "[RUN] GPU=${gpu} tag=${tag}"
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

# GPU0: SD1.5 two runs (sequential), whole block in background
(
  run_one 0 "${SD15}" "${ZT_W_ATT}" "${OUT_SD15_WATT}" "sd15_TR_w_att_0_88_number"
  run_one 0 "${SD15}" "${ZT_W}"     "${OUT_SD15_W}"     "sd15_TR_w_number"
) 2>&1 | tee "${LOGDIR}/sd15_gpu0.log" &

PID0=$!

# GPU1: SD2.1 two runs (sequential), whole block in background
(
  run_one 1 "${SD21}" "${ZT_W_ATT}" "${OUT_SD21_WATT}" "sd21_TR_w_att_0_88_number"
  run_one 1 "${SD21}" "${ZT_W}"     "${OUT_SD21_W}"     "sd21_TR_w_number"
) 2>&1 | tee "${LOGDIR}/sd21_gpu1.log" &

PID1=$!

echo "[INFO] Launched PIDs: sd15=${PID0}, sd21=${PID1}"
echo "[INFO] Logs: ${LOGDIR}/sd15_gpu0.log  and  ${LOGDIR}/sd21_gpu1.log"

wait "${PID0}" "${PID1}"

echo "[DONE] All generations finished."
nvidia-smi || true
