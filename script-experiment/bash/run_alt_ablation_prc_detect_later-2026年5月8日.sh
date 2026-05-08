#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
IMGROOT=${EXP_ROOT}/imgs

cd "${EXP_ROOT}" || exit 1

PRC_DET=${EXP_ROOT}/script-experiment/detect/prc_detect_alt_global_official_align.py
MODEL_ID=${ROOT}/checkpoints/AltDiffusion-fp16
WM_META=${EXP_ROOT}/latents_experiment/wm_meta

SEED=12345
DTYPE=fp16
INV_STEPS=50
FPR=1e-2
MASTER_KEY=prc_key_sd14_0124
MESSAGE_LENGTH=8
MAX_BP_ITER=5000

RUN_DELREPAIR=${IMGROOT}/vis_alt_ablate_prc_delrepair_seed${SEED}
RUN_DELSSC=${IMGROOT}/vis_alt_ablate_prc_delssc_seed${SEED}

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

wait_for_run () {
  local run_dir="$1"
  local tag="$2"
  local timeout_sec="${3:-21600}"   # default 6h
  local interval_sec="${4:-60}"

  echo "[WAIT][${tag}] waiting for ${run_dir}/sliced/manifest.csv ..." >&2

  local waited=0
  while true; do
    if [[ -f "${run_dir}/sliced/manifest.csv" ]]; then
      local n_imgs
      n_imgs=$(find "${run_dir}/sliced" -maxdepth 1 -type f -name "*.png" | wc -l | tr -d ' ')
      if [[ "${n_imgs}" -gt 0 ]]; then
        echo "[WAIT][${tag}] ready: ${n_imgs} png files found." >&2
        return 0
      fi
    fi

    if [[ "${waited}" -ge "${timeout_sec}" ]]; then
      echo "[ERROR][${tag}] timeout waiting for ${run_dir}/sliced/manifest.csv" >&2
      return 1
    fi

    sleep "${interval_sec}"
    waited=$((waited + interval_sec))
    echo "[WAIT][${tag}] waited ${waited}s ..." >&2
  done
}

run_prc_detect () {
  local tag="$1"
  local run_dir="$2"
  local out_dir="${run_dir}/detect_prc_alt"

  mkdir -p "${out_dir}"

  echo "[PRC-DETECT][${tag}] run_dir=${run_dir}" >&2

  CUDA_VISIBLE_DEVICES=0 \
    python "${PRC_DET}" \
      --model_id "${MODEL_ID}" \
      --run_dir "${run_dir}" \
      --meta_root "${WM_META}" \
      --dtype "${DTYPE}" \
      --inv_steps "${INV_STEPS}" \
      --inv_bs 1 \
      --fpr "${FPR}" \
      --master_key "${MASTER_KEY}" \
      --message_length "${MESSAGE_LENGTH}" \
      --max_bp_iter "${MAX_BP_ITER}" \
      --save_zt \
      --save_zt_dir "${out_dir}/latents_prc_alt" \
      --out_csv "${out_dir}/detect_results_prcGLOBAL_alt.csv" \
      > "${out_dir}/detect_prc_alt.runlog.txt" 2>&1

  echo "[DONE][${tag}] ${out_dir}" >&2
}

need_file "${PRC_DET}"
need_dir "${WM_META}"

echo "[INFO] PRC detector: ${PRC_DET}" >&2
echo "[INFO] wm_meta: ${WM_META}" >&2

wait_for_run "${RUN_DELREPAIR}" "prc_delrepair" 21600 60 || exit 1
run_prc_detect "prc_delrepair" "${RUN_DELREPAIR}" || exit 1

wait_for_run "${RUN_DELSSC}" "prc_delssc" 21600 60 || exit 1
run_prc_detect "prc_delssc" "${RUN_DELSSC}" || exit 1

echo "[DONE] PRC Alt ablation detection finished." >&2
echo "  ${RUN_DELREPAIR}/detect_prc_alt" >&2
echo "  ${RUN_DELSSC}/detect_prc_alt" >&2