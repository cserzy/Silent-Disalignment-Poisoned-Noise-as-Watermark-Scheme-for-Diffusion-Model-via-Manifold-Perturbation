#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
LATDIR=${EXP_ROOT}/latents_experiment
IMGROOT=${EXP_ROOT}/imgs

cd "${EXP_ROOT}" || exit 1

PROMPTS=${ROOT}/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt
MODEL_ID=${ROOT}/checkpoints/AltDiffusion-fp16

GEN=${EXP_ROOT}/script-experiment/gen_from_zT_bank_alt_diffusion-1_19.py
BLACK_DET=${EXP_ROOT}/script-experiment/detect/detect_black_ratio_alt.py
PRC_DET=${EXP_ROOT}/script-experiment/detect/prc_detect_alt_global_official_align.py

ZT_PT=${LATDIR}/generate_PRC_w_att_0_85_delssc.pt
WM_META=${LATDIR}/wm_meta

GPU=0
SEED=12345
GEN_STEPS=50
GEN_CFG=7.5
HEIGHT=512
WIDTH=512
DTYPE=fp16
N_PER_PROMPT=4
START_LATENT=0

INV_STEPS=50
PRC_FPR=1e-2
PRC_MASTER_KEY=prc_key_sd14_0124
PRC_MESSAGE_LENGTH=8
PRC_MAX_BP_ITER=5000

RUN_SAFEOFF=${IMGROOT}/vis_alt_ablate_prc_delssc_seed${SEED}
RUN_SAFEON=${IMGROOT}/vis_alt_ablate_safeon_prc_delssc_seed${SEED}

need_file () {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] missing file: $1" >&2
    exit 1
  fi
}

need_dir () {
  if [[ ! -d "$1" ]]; then
    echo "[ERROR] missing dir: $1" >&2
    exit 1
  fi
}

count_pngs () {
  local dir="$1"
  if [[ -d "${dir}/sliced" ]]; then
    find "${dir}/sliced" -maxdepth 1 -type f -name "*.png" | wc -l | tr -d ' '
  else
    echo 0
  fi
}

need_file "${GEN}"
need_file "${BLACK_DET}"
need_file "${PRC_DET}"
need_file "${ZT_PT}"
need_file "${PROMPTS}"
need_dir "${WM_META}"

echo "========== [PRC DELSSC ONLY][GPU${GPU}] =========="
echo "[INFO] zT_pt=${ZT_PT}"
echo "[INFO] wm_meta=${WM_META}"

echo "[STEP 1] regenerate safe-off full 200 images"
rm -rf "${RUN_SAFEOFF}"
mkdir -p "${RUN_SAFEOFF}"

CUDA_VISIBLE_DEVICES=${GPU} \
python "${GEN}" \
  --model_id "${MODEL_ID}" \
  --prompts "${PROMPTS}" \
  --zT_pt "${ZT_PT}" \
  --outdir "${RUN_SAFEOFF}" \
  --steps ${GEN_STEPS} \
  --cfg ${GEN_CFG} \
  --height ${HEIGHT} \
  --width ${WIDTH} \
  --device cuda \
  --dtype ${DTYPE} \
  --n_per_prompt ${N_PER_PROMPT} \
  --start_latent ${START_LATENT} \
  --seed ${SEED} \
  --disable_safety_checker \
  > "${RUN_SAFEOFF}/gen.runlog.txt" 2>&1

N_SAFE_OFF=$(count_pngs "${RUN_SAFEOFF}")
echo "[INFO] safe-off pngs=${N_SAFE_OFF}"
if [[ "${N_SAFE_OFF}" -lt 200 ]]; then
  echo "[ERROR] safe-off generation incomplete: ${N_SAFE_OFF}/200" >&2
  exit 1
fi

echo "[STEP 2] regenerate safe-on full 200 images"
rm -rf "${RUN_SAFEON}"
mkdir -p "${RUN_SAFEON}"

CUDA_VISIBLE_DEVICES=${GPU} \
python "${GEN}" \
  --model_id "${MODEL_ID}" \
  --prompts "${PROMPTS}" \
  --zT_pt "${ZT_PT}" \
  --outdir "${RUN_SAFEON}" \
  --steps ${GEN_STEPS} \
  --cfg ${GEN_CFG} \
  --height ${HEIGHT} \
  --width ${WIDTH} \
  --device cuda \
  --dtype ${DTYPE} \
  --n_per_prompt ${N_PER_PROMPT} \
  --start_latent ${START_LATENT} \
  --seed ${SEED} \
  > "${RUN_SAFEON}/gen.runlog.txt" 2>&1

N_SAFE_ON=$(count_pngs "${RUN_SAFEON}")
echo "[INFO] safe-on pngs=${N_SAFE_ON}"
if [[ "${N_SAFE_ON}" -lt 200 ]]; then
  echo "[ERROR] safe-on generation incomplete: ${N_SAFE_ON}/200" >&2
  exit 1
fi

echo "[STEP 3] black-rate detection on safe-on"
BLACK_OUT=${RUN_SAFEON}/black_detect
rm -rf "${BLACK_OUT}"
mkdir -p "${BLACK_OUT}"

python "${BLACK_DET}" \
  --run_dir "${RUN_SAFEON}" \
  --out_dir "${BLACK_OUT}" \
  > "${BLACK_OUT}/black_detect.runlog.txt" 2>&1

echo "[STEP 4] PRC watermark detection on safe-off"
DET_OUT=${RUN_SAFEOFF}/detect_prc_alt
rm -rf "${DET_OUT}"
mkdir -p "${DET_OUT}"

CUDA_VISIBLE_DEVICES=${GPU} \
python "${PRC_DET}" \
  --model_id "${MODEL_ID}" \
  --run_dir "${RUN_SAFEOFF}" \
  --meta_root "${WM_META}" \
  --dtype "${DTYPE}" \
  --inv_steps "${INV_STEPS}" \
  --inv_bs 1 \
  --fpr "${PRC_FPR}" \
  --master_key "${PRC_MASTER_KEY}" \
  --message_length "${PRC_MESSAGE_LENGTH}" \
  --max_bp_iter "${PRC_MAX_BP_ITER}" \
  --save_zt \
  --save_zt_dir "${DET_OUT}/latents_prc_alt" \
  --out_csv "${DET_OUT}/detect_results_prcGLOBAL_alt.csv" \
  > "${DET_OUT}/detect_prc_alt.runlog.txt" 2>&1

echo "[DONE] PRC delssc Alt pipeline finished."
echo "  safe-off:     ${RUN_SAFEOFF}"
echo "  safe-on:      ${RUN_SAFEON}"
echo "  black result: ${BLACK_OUT}"
echo "  prc detect:   ${DET_OUT}"