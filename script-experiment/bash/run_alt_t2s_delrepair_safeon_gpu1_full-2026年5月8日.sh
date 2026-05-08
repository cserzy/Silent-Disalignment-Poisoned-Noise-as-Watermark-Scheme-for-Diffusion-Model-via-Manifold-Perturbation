#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
LATDIR=${EXP_ROOT}/latents_experiment
IMGROOT=${EXP_ROOT}/imgs

MODEL_ID=${ROOT}/checkpoints/AltDiffusion-fp16
PROMPTS=${ROOT}/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt

GEN_PY=${EXP_ROOT}/script-experiment/gen_from_zT_bank_alt_diffusion-1_19.py
BLACK_DET=${EXP_ROOT}/script-experiment/detect/detect_black_ratio_alt.py
T2S_DET=${EXP_ROOT}/script-experiment/detect/detect_T2S_alt.py

ZT_PT=${LATDIR}/generate_T2S_w_att_delrepair.pt
META_PT=${LATDIR}/generate_T2S_w_att_delrepair_meta.pt

RUN_DIR=${IMGROOT}/vis_alt_ablate_safeon_t2s_delrepair_seed12345
BLACK_OUT=${RUN_DIR}/black_detect
DET_OUT=${IMGROOT}/vis_alt_ablate_t2s_delrepair_seed12345/detect_t2s_alt

GPU=1
SEED=12345
STEPS=50
CFG=7.5
HEIGHT=512
WIDTH=512
DTYPE=fp16
N_PER_PROMPT=4
START_LATENT=0
INV_STEPS=50

need_file () {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] missing file: $1" >&2
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

need_file "${GEN_PY}"
need_file "${BLACK_DET}"
need_file "${T2S_DET}"
need_file "${ZT_PT}"
need_file "${META_PT}"
need_file "${PROMPTS}"

echo "========== [T2S DELREPAIR SAFE-ON FULL][GPU${GPU}] =========="
echo "[INFO] run_dir=${RUN_DIR}"

EXISTING=$(count_pngs "${RUN_DIR}")
echo "[INFO] existing_pngs=${EXISTING}"

if [[ "${EXISTING}" -lt 200 ]]; then
  echo "[INFO] existing pngs < 200, remove old safe-on dir and regenerate full 200 imgs"
  rm -rf "${RUN_DIR}"
  mkdir -p "${RUN_DIR}"

  CUDA_VISIBLE_DEVICES=${GPU} \
  python "${GEN_PY}" \
    --model_id "${MODEL_ID}" \
    --prompts "${PROMPTS}" \
    --zT_pt "${ZT_PT}" \
    --outdir "${RUN_DIR}" \
    --steps ${STEPS} \
    --cfg ${CFG} \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --device cuda \
    --dtype ${DTYPE} \
    --n_per_prompt ${N_PER_PROMPT} \
    --start_latent ${START_LATENT} \
    --seed ${SEED} \
    > "${RUN_DIR}/gen.runlog.txt" 2>&1
else
  echo "[INFO] safe-on already complete, skip generation"
fi

FINAL_COUNT=$(count_pngs "${RUN_DIR}")
echo "[INFO] final_pngs=${FINAL_COUNT}"

if [[ "${FINAL_COUNT}" -lt 200 ]]; then
  echo "[ERROR] generation incomplete: final_pngs=${FINAL_COUNT}, expected 200" >&2
  exit 1
fi

echo "[STEP] black-ratio detect"
rm -rf "${BLACK_OUT}"
mkdir -p "${BLACK_OUT}"

python "${BLACK_DET}" \
  --run_dir "${RUN_DIR}" \
  --out_dir "${BLACK_OUT}" \
  > "${BLACK_OUT}/black_detect.runlog.txt" 2>&1

echo "[STEP] T2S detect on safe-off companion run"
SAFEOFF_RUN=${IMGROOT}/vis_alt_ablate_t2s_delrepair_seed12345
if [[ ! -d "${SAFEOFF_RUN}" ]]; then
  echo "[ERROR] safe-off run dir not found: ${SAFEOFF_RUN}" >&2
  echo "[ERROR] T2S decode is designed to run on safe-off images." >&2
  exit 1
fi

rm -rf "${DET_OUT}"
mkdir -p "${DET_OUT}"

CUDA_VISIBLE_DEVICES=${GPU} \
python "${T2S_DET}" \
  --model_id "${MODEL_ID}" \
  --run_dir "${SAFEOFF_RUN}" \
  --out_dir "${DET_OUT}" \
  --cluster_meta_pt "${META_PT}" \
  --dtype ${DTYPE} \
  --inv_steps ${INV_STEPS} \
  --save_zt \
  --compute_auc \
  > "${DET_OUT}/detect_t2s_alt.runlog.txt" 2>&1

echo "[DONE]"
echo "  safe-on imgs:   ${RUN_DIR}"
echo "  black detect:   ${BLACK_OUT}"
echo "  t2s detect:     ${DET_OUT}"
echo "========== [T2S DELREPAIR SAFE-ON FULL DONE] =========="