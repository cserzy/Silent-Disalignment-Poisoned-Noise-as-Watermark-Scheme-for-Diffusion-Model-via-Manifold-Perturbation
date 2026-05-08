#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
cd "${EXP_ROOT}" || exit 1

BLACK_DET=${EXP_ROOT}/script-experiment/detect/detect_black_ratio_alt.py
T2S_DET=${EXP_ROOT}/script-experiment/detect/detect_T2S_alt.py

MODEL_ID=${ROOT}/checkpoints/AltDiffusion-fp16

T2S_META_PT=${EXP_ROOT}/latents_experiment/generate_T2S_w_att_meta.pt
T2S_META_JSON=${EXP_ROOT}/latents_experiment/generate_T2S_w_att_meta.json

DTYPE=fp16
INV_STEPS=20

# 已生成好的目录
# safe-off: 用于水印检测
RUN_SAFEOFF_W=${EXP_ROOT}/imgs/vis_alt_t2s_w_seed12345
RUN_SAFEOFF_WATT=${EXP_ROOT}/imgs/vis_alt_t2s_w_att_seed12345

# safe-on: 用于黑图率检测
RUN_SAFEON_W=${EXP_ROOT}/imgs/vis_alt_t2s_safeon_w_seed12345
RUN_SAFEON_WATT=${EXP_ROOT}/imgs/vis_alt_t2s_safeon_w_att_seed12345

pids=()

run_black_job () {
  local run_dir="$1"
  local out_dir="${run_dir}/black_detect"

  mkdir -p "${out_dir}"

  echo "[LAUNCH][BLACK] ${run_dir}" >&2
  python "${BLACK_DET}" \
    --run_dir "${run_dir}" \
    --out_dir "${out_dir}" \
    > "${out_dir}/black_detect.runlog.txt" 2>&1 &

  pids+=("$!")
}

run_t2s_job () {
  local gpu="$1"
  local run_dir="$2"
  local out_dir="${run_dir}/detect_t2s_alt"

  mkdir -p "${out_dir}"

  echo "[LAUNCH][T2S][GPU${gpu}] ${run_dir}" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "${T2S_DET}" \
      --model_id "${MODEL_ID}" \
      --run_dir "${run_dir}" \
      --out_dir "${out_dir}" \
      --cluster_meta_pt "${T2S_META_PT}" \
      --dtype ${DTYPE} \
      --inv_steps ${INV_STEPS} \
      --save_zt \
    > "${out_dir}/detect_t2s_alt.runlog.txt" 2>&1 &

  pids+=("$!")
}

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

echo "[INFO] T2S meta:"
echo "  PT:   ${T2S_META_PT}" >&2
echo "  JSON: ${T2S_META_JSON}" >&2

echo "[INFO] Stage 1: black-rate detection on safe-on runs..." >&2
run_black_job "${RUN_SAFEON_W}"
run_black_job "${RUN_SAFEON_WATT}"
wait_all || exit 1

echo "[INFO] Stage 2: T2S watermark detection on safe-off runs..." >&2
run_t2s_job 0 "${RUN_SAFEOFF_W}"
run_t2s_job 1 "${RUN_SAFEOFF_WATT}"
wait_all || exit 1

echo "[DONE] T2S black-rate + watermark detection finished." >&2
echo "[DONE] Black-rate results:" >&2
echo "  ${RUN_SAFEON_W}/black_detect" >&2
echo "  ${RUN_SAFEON_WATT}/black_detect" >&2
echo "[DONE] T2S detect results:" >&2
echo "  ${RUN_SAFEOFF_W}/detect_t2s_alt" >&2
echo "  ${RUN_SAFEOFF_WATT}/detect_t2s_alt" >&2