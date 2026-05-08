#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
IMGROOT=${EXP_ROOT}/imgs
LATDIR=${EXP_ROOT}/latents_experiment

DET_PY=${EXP_ROOT}/script-experiment/detect/detect_TR_oms.py

SD14=${ROOT}/checkpoints/sd1-4-diffusers
SD15=${ROOT}/checkpoints/sd1-5-diffusers
SD21=${ROOT}/checkpoints/sd2-1-diffusers

OMS_WATT_Q=${LATDIR}/oms_Q_TR_w_att_to_gauss_aligned.pt
OMS_WATT_Q_JSON=${LATDIR}/oms_Q_TR_w_att_to_gauss_aligned.json

RUN_SD14=${IMGROOT}/vis_sd14_TR_w_att_oms_gauss_aligned_seed12345
RUN_SD15=${IMGROOT}/vis_sd15_TR_w_att_oms_gauss_aligned_seed12345
RUN_SD21=${IMGROOT}/vis_sd21_TR_w_att_oms_gauss_aligned_seed12345

pids=()

run_bg () {
  local gpu="$1"
  shift
  echo "[LAUNCH][GPU${gpu}] $*" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" "$@" &
  pids+=("$!")
}

wait_all_or_fail () {
  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "[ERROR] detect job failed: pid=$pid" >&2
      fail=1
    fi
  done
  pids=()
  return "$fail"
}

run_det () {
  local gpu="$1"
  local model_id="$2"
  local rundir="$3"

  local outdir="${rundir}/detect_tr_oms"
  local log="${outdir}/detect_tr_oms.runlog.txt"
  mkdir -p "${outdir}"

  run_bg "${gpu}" \
    python "${DET_PY}" \
      --model_id "${model_id}" \
      --run_dir "${rundir}" \
      --out_dir "${outdir}" \
      --oms_q_pt "${OMS_WATT_Q}" \
      --oms_meta_json "${OMS_WATT_Q_JSON}" \
      --inv_steps 50 \
      --dtype fp16 \
      --save_zt_oms \
      --save_zt_restored \
    > "${log}" 2>&1
}

echo
echo "============================================================"
echo "[TR DETECT ONLY] Start"
echo "============================================================"

run_det 0 "${SD14}" "${RUN_SD14}"
run_det 1 "${SD15}" "${RUN_SD15}"
run_det 1 "${SD21}" "${RUN_SD21}"

if ! wait_all_or_fail; then
  echo "[DONE] Some TR detect jobs failed. Check logs:" >&2
  echo "       ${IMGROOT}/vis_sd*_TR_w_att_oms_gauss_aligned_seed12345/detect_tr_oms/detect_tr_oms.runlog.txt" >&2
  exit 1
fi

echo
echo "============================================================"
echo "[DONE] All TR detect jobs finished OK."
echo "============================================================"