#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
LATDIR=${EXP_ROOT}/latents_experiment
LATDIR_NUM=${EXP_ROOT}/latents_experiment-number
IMGROOT=${EXP_ROOT}/imgs

cd "${EXP_ROOT}" || exit 1

PROMPTS=${ROOT}/prompts/cal_dongman_female_align-jinghua-2026-1_25.txt
MODEL_ID=${ROOT}/checkpoints/AltDiffusion-fp16

OMS_PY=${EXP_ROOT}/script-experiment/oms_repair_pt.py
GEN=${EXP_ROOT}/script-experiment/gen_from_zT_bank_alt_diffusion-1_19.py
BLACK_DET=${EXP_ROOT}/script-experiment/detect/detect_black_ratio_alt.py
GS_OMS_DET=${EXP_ROOT}/script-experiment/detect/detect_GS_alt_oms.py

SRC=${LATDIR_NUM}/generate_GS_w_att_dongman.pt
TARGET_GAUSS=${LATDIR}/generate_GAUSS_w_aligned_vis.pt

OMS_PT=${LATDIR_NUM}/generate_GS_w_att_dongman_oms_gauss_aligned.pt
OMS_Q_PT=${LATDIR_NUM}/oms_Q_GS_w_att_dongman_to_gauss_aligned.pt
OMS_Q_JSON=${LATDIR_NUM}/oms_Q_GS_w_att_dongman_to_gauss_aligned.json

SEED=12345
Q_SEED=12345
BLOCK_SIZE=64
BLEND_ALPHA=0.2
MATCH_TARGET_STD=1

GEN_STEPS=50
GEN_CFG=7.5
HEIGHT=512
WIDTH=512
DTYPE=fp16
FIT_DTYPE=fp32
N_PER_PROMPT=4
START_LATENT=0
INV_STEPS=50

RUN_SAFEOFF=${IMGROOT}/vis_alt_gs_oms_w_att_dongman_seed${SEED}
RUN_SAFEON=${IMGROOT}/vis_alt_gs_oms_safeon_w_att_dongman_seed${SEED}

need_file () {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] missing file: $1" >&2
    exit 1
  fi
}

need_file "${SRC}"
need_file "${TARGET_GAUSS}"
need_file "${PROMPTS}"

pids=()

wait_all () {
  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      echo "[ERROR] job failed: pid=${pid}" >&2
      fail=1
    fi
  done
  pids=()
  return "${fail}"
}

echo "[GS][1/5] OMS repair: GS dongman w_att -> aligned Gaussian" >&2
python "${OMS_PY}" \
  --mode fit_apply \
  --in_pt "${SRC}" \
  --target_pt "${TARGET_GAUSS}" \
  --out_pt "${OMS_PT}" \
  --out_q_pt "${OMS_Q_PT}" \
  --out_meta_json "${OMS_Q_JSON}" \
  --q_seed "${Q_SEED}" \
  --block_size "${BLOCK_SIZE}" \
  --blend_alpha "${BLEND_ALPHA}" \
  --match_target_std "${MATCH_TARGET_STD}" \
  --device cpu \
  --dtype "${FIT_DTYPE}" \
  --verbose || exit 1

run_gen () {
  local gpu="$1"
  local outdir="$2"
  local disable_sc="$3"

  mkdir -p "${outdir}"

  echo "[GS][GEN][GPU${gpu}] ${outdir} | disable_safety_checker=${disable_sc}" >&2

  if [[ "${disable_sc}" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python "${GEN}" \
      --model_id "${MODEL_ID}" \
      --prompts "${PROMPTS}" \
      --zT_pt "${OMS_PT}" \
      --outdir "${outdir}" \
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
      > "${outdir}/gen.runlog.txt" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="${gpu}" python "${GEN}" \
      --model_id "${MODEL_ID}" \
      --prompts "${PROMPTS}" \
      --zT_pt "${OMS_PT}" \
      --outdir "${outdir}" \
      --steps ${GEN_STEPS} \
      --cfg ${GEN_CFG} \
      --height ${HEIGHT} \
      --width ${WIDTH} \
      --device cuda \
      --dtype ${DTYPE} \
      --n_per_prompt ${N_PER_PROMPT} \
      --start_latent ${START_LATENT} \
      --seed ${SEED} \
      > "${outdir}/gen.runlog.txt" 2>&1 &
  fi

  pids+=("$!")
}

echo "[GS][2/5] Generate safe-off images for watermark detection" >&2
run_gen 0 "${RUN_SAFEOFF}" 1
wait_all || exit 1

echo "[GS][3/5] Generate safe-on images for black-rate detection" >&2
run_gen 0 "${RUN_SAFEON}" 0
wait_all || exit 1

echo "[GS][4/5] Black-rate detection on safe-on images" >&2
mkdir -p "${RUN_SAFEON}/black_detect"
python "${BLACK_DET}" \
  --run_dir "${RUN_SAFEON}" \
  --out_dir "${RUN_SAFEON}/black_detect" \
  > "${RUN_SAFEON}/black_detect/black_detect.runlog.txt" 2>&1 || exit 1

echo "[GS][5/5] GS-OMS watermark detection on safe-off images" >&2
mkdir -p "${RUN_SAFEOFF}/detect_gs_alt_oms"
CUDA_VISIBLE_DEVICES=0 python "${GS_OMS_DET}" \
  --model_id "${MODEL_ID}" \
  --run_dir "${RUN_SAFEOFF}" \
  --out_dir "${RUN_SAFEOFF}/detect_gs_alt_oms" \
  --oms_q_pt "${OMS_Q_PT}" \
  --oms_meta_json "${OMS_Q_JSON}" \
  --dtype ${DTYPE} \
  --inv_steps ${INV_STEPS} \
  --save_zt_oms \
  --save_zt_restored \
  > "${RUN_SAFEOFF}/detect_gs_alt_oms/detect_gs_alt_oms.runlog.txt" 2>&1 || exit 1

echo "[DONE][GS] Alt GS-OMS dongman pipeline finished." >&2
echo "safe-off: ${RUN_SAFEOFF}" >&2
echo "safe-on : ${RUN_SAFEON}" >&2