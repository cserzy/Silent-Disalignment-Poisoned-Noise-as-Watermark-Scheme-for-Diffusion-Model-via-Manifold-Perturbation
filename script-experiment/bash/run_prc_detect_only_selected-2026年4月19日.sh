#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
IMGROOT=${EXP_ROOT}/imgs
LATDIR=${EXP_ROOT}/latents_experiment

DET_PY=${EXP_ROOT}/script-experiment/detect/detect_PRC_oms.py

SD14=${ROOT}/checkpoints/sd1-4-diffusers
SD15=${ROOT}/checkpoints/sd1-5-diffusers
SD21=${ROOT}/checkpoints/sd2-1-diffusers

# -------------------------
# PRC repo path
# 主要用于 inverse_stable_diffusion 等外部模块
# 当前 prc.py / pseudogaussians.py 优先使用 detect/ 目录下的本地版本
# 如果真实路径不同，运行前 export PRC_REPO=/your/path
# -------------------------
PRC_REPO=${PRC_REPO:-${ROOT}/PRC-Watermark-main}

# -------------------------
# 指定要检测的 3 个目录
# -------------------------
RUN_SD14=${IMGROOT}/vis_sd14_PRC_w_att_oms_gauss_aligned_b64_a0p20_seed12345
RUN_SD15=${IMGROOT}/vis_sd15_PRC_w_att_oms_gauss_aligned_b128_a0p30_seed12345
RUN_SD21=${IMGROOT}/vis_sd21_PRC_w_att_oms_gauss_aligned_b64_a0p20_seed12345

# -------------------------
# 每个目录对应的 OMS q/json
# -------------------------
OMS_Q_SD14=${LATDIR}/oms_Q_PRC_w_att_to_gauss_aligned_b64_a0p20.pt
OMS_JSON_SD14=${LATDIR}/oms_Q_PRC_w_att_to_gauss_aligned_b64_a0p20.json

OMS_Q_SD15=${LATDIR}/oms_Q_PRC_w_att_to_gauss_aligned_b128_a0p30.pt
OMS_JSON_SD15=${LATDIR}/oms_Q_PRC_w_att_to_gauss_aligned_b128_a0p30.json

OMS_Q_SD21=${LATDIR}/oms_Q_PRC_w_att_to_gauss_aligned_b64_a0p20.pt
OMS_JSON_SD21=${LATDIR}/oms_Q_PRC_w_att_to_gauss_aligned_b64_a0p20.json

# -------------------------
# wm_meta 模板目录
# 按你之前的使用习惯：先拷到每个输出目录下再检测
# 如果真实模板目录不同，只改这一行
# -------------------------
WM_META_TEMPLATE_DIR=${EXP_ROOT}/imgs/vis_sd15_PRC_w_att_oms_gauss_aligned_seed12345/wm_meta

PRC_INV_STEPS=50
PRC_DTYPE=fp32
PRC_MAX_BP_ITER=5000
PRC_FPR=1e-2

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

copy_wm_meta() {
  local run_dir="$1"
  local dst="${run_dir}/wm_meta"

  if [ ! -d "${WM_META_TEMPLATE_DIR}" ]; then
    echo "[ERROR] wm_meta template dir not found: ${WM_META_TEMPLATE_DIR}" >&2
    return 1
  fi

  rm -rf "${dst}"
  mkdir -p "${run_dir}"
  cp -r "${WM_META_TEMPLATE_DIR}" "${dst}" || return 1
}

run_det () {
  local gpu="$1"
  local model_id="$2"
  local rundir="$3"
  local q_pt="$4"
  local q_json="$5"

  local outdir="${rundir}/detect_prc_oms"
  local log="${outdir}/detect_prc_oms.runlog.txt"

  mkdir -p "${outdir}"

  run_bg "${gpu}" \
    python "${DET_PY}" \
      --model_id "${model_id}" \
      --run_dir "${rundir}" \
      --out_dir "${outdir}" \
      --wm_meta_dir "${rundir}/wm_meta" \
      --oms_q_pt "${q_pt}" \
      --oms_meta_json "${q_json}" \
      --prc_repo "${PRC_REPO}" \
      --inv_steps "${PRC_INV_STEPS}" \
      --max_bp_iter "${PRC_MAX_BP_ITER}" \
      --fpr "${PRC_FPR}" \
      --dtype "${PRC_DTYPE}" \
      --save_zt_oms \
      --save_zt_restored \
      --guidance_scale 7.5 \
    > "${log}" 2>&1
}

echo
echo "============================================================"
echo "[PRC DETECT ONLY] Copy wm_meta into selected run dirs"
echo "============================================================"

copy_wm_meta "${RUN_SD14}" || exit 1
copy_wm_meta "${RUN_SD15}" || exit 1
copy_wm_meta "${RUN_SD21}" || exit 1

echo
echo "============================================================"
echo "[PRC DETECT ONLY] Start detection on selected 3 run dirs"
echo "============================================================"
echo "[INFO] max_bp_iter=${PRC_MAX_BP_ITER}, fpr=${PRC_FPR}"

run_det 0 "${SD14}" "${RUN_SD14}" "${OMS_Q_SD14}" "${OMS_JSON_SD14}"
run_det 0 "${SD15}" "${RUN_SD15}" "${OMS_Q_SD15}" "${OMS_JSON_SD15}"
run_det 0 "${SD21}" "${RUN_SD21}" "${OMS_Q_SD21}" "${OMS_JSON_SD21}"

if ! wait_all_or_fail; then
  echo "[DONE] Some PRC detect jobs failed. Check logs:" >&2
  echo "       ${RUN_SD14}/detect_prc_oms/detect_prc_oms.runlog.txt" >&2
  echo "       ${RUN_SD15}/detect_prc_oms/detect_prc_oms.runlog.txt" >&2
  echo "       ${RUN_SD21}/detect_prc_oms/detect_prc_oms.runlog.txt" >&2
  exit 1
fi

echo
echo "============================================================"
echo "[DONE] All selected PRC detect jobs finished OK."
echo "============================================================"