#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
LATDIR_NUM=${EXP_ROOT}/latents_experiment-number
IMGROOT=${EXP_ROOT}/imgs

PROMPTS=${ROOT}/prompts/cal_dongman_female_align-jinghua-2026-1_25.txt

GEN_PY=${EXP_ROOT}/script-experiment/gen_from_zT_bank_multi_models-1_19.py
DET_PY=${EXP_ROOT}/script-experiment/detect/detect_T2S_oms.py
NSFW_PY=${ROOT}/nsfw_score_report_ring_wm_only_exposed_only-12.29.py
T2S_ROOT=${ROOT}/third_party/T2SMark

SD14=${ROOT}/checkpoints/sd1-4-diffusers
SD15=${ROOT}/checkpoints/sd1-5-diffusers
SD21=${ROOT}/checkpoints/sd2-1-diffusers

ZT_PT=${LATDIR_NUM}/generate_T2S_w_att_dongman_oms_gauss_aligned_b64_a0p20.pt

T2S_META_PT=${LATDIR_NUM}/generate_T2S_w_att_dongman_meta.pt
T2S_META_JSON=${LATDIR_NUM}/generate_T2S_w_att_dongman_meta.json

OMS_Q=${LATDIR_NUM}/oms_Q_T2S_w_att_dongman_to_gauss_aligned_b64_a0p20.pt
OMS_Q_JSON=${LATDIR_NUM}/oms_Q_T2S_w_att_dongman_to_gauss_aligned_b64_a0p20.json

RUN_SD14=${IMGROOT}/vis_sd14_T2S_w_att_dongman_oms_gauss_aligned_b64_a0p20_seed12345
RUN_SD15=${IMGROOT}/vis_sd15_T2S_w_att_dongman_oms_gauss_aligned_b64_a0p20_seed12345
RUN_SD21=${IMGROOT}/vis_sd21_T2S_w_att_dongman_oms_gauss_aligned_b64_a0p20_seed12345

STEPS=50
CFG=7.5
HEIGHT=512
WIDTH=512
N_PER_PROMPT=4
START_LATENT=0
GEN_DTYPE=fp16
SEED=12345
INV_STEPS=50

NSFW_THRESHOLD=0.6
NSFW_SWEEP="0.2,0.3,0.4,0.5,0.6,0.7,0.8"

export PYTHONPATH=${T2S_ROOT}:${PYTHONPATH:-}

run_cmd() {
  echo
  echo "============================================================"
  echo "[RUN] $*"
  echo "============================================================"
  "$@" || exit 1
}

check_file() {
  local f="$1"
  if [ ! -f "$f" ]; then
    echo "[ERROR] missing file: $f" >&2
    exit 1
  fi
}

check_file "${ZT_PT}"
check_file "${T2S_META_PT}"
check_file "${T2S_META_JSON}"
check_file "${OMS_Q}"
check_file "${OMS_Q_JSON}"

run_generate() {
  local gpu="$1"
  local model_id="$2"
  local run_dir="$3"
  local tag="$4"

  mkdir -p "$(dirname "${run_dir}")"

  echo
  echo "============================================================"
  echo "[GENERATE] ${tag}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${gpu}" run_cmd python "${GEN_PY}" \
    --model_id "${model_id}" \
    --prompts "${PROMPTS}" \
    --zT_pt "${ZT_PT}" \
    --outdir "${run_dir}" \
    --steps "${STEPS}" --cfg "${CFG}" --height "${HEIGHT}" --width "${WIDTH}" \
    --n_per_prompt "${N_PER_PROMPT}" --start_latent "${START_LATENT}" \
    --dtype "${GEN_DTYPE}" --seed "${SEED}" \
    --negative_prompt ""
}

run_nsfw() {
  local run_dir="$1"
  local tag="$2"

  local manifest=""
  if [ -f "${run_dir}/sliced/manifest.csv" ]; then
    manifest="${run_dir}/sliced/manifest.csv"
  elif [ -f "${run_dir}/manifest.csv" ]; then
    manifest="${run_dir}/manifest.csv"
  else
    echo "[ERROR] manifest not found for ${tag}: ${run_dir}" >&2
    exit 1
  fi

  local nsfw_dir="${run_dir}/nsfw_report"
  mkdir -p "${nsfw_dir}"

  echo
  echo "============================================================"
  echo "[NSFW] ${tag}"
  echo "============================================================"

  run_cmd python "${NSFW_PY}" \
    --manifests "${manifest}" \
    --out_dir "${nsfw_dir}" \
    --report_out "${nsfw_dir}/report.xlsx" \
    --threshold "${NSFW_THRESHOLD}" \
    --sweep "${NSFW_SWEEP}"
}

run_detect() {
  local gpu="$1"
  local model_id="$2"
  local run_dir="$3"
  local tag="$4"

  local det_dir="${run_dir}/detect_t2s_oms"
  mkdir -p "${det_dir}"

  echo
  echo "============================================================"
  echo "[DETECT] ${tag}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${gpu}" run_cmd python "${DET_PY}" \
    --model_id "${model_id}" \
    --run_dir "${run_dir}" \
    --out_dir "${det_dir}" \
    --cluster_meta_pt "${T2S_META_PT}" \
    --cluster_meta_json "${T2S_META_JSON}" \
    --oms_q_pt "${OMS_Q}" \
    --oms_meta_json "${OMS_Q_JSON}" \
    --t2s_root "${T2S_ROOT}" \
    --inv_steps "${INV_STEPS}" \
    --dtype "${GEN_DTYPE}" \
    --save_zt_oms \
    --save_zt_restored
}

echo
echo "============================================================"
echo "[PARTIAL PIPELINE] T2S dongman"
echo "sd14: detect only"
echo "sd15: generate + nsfw + detect"
echo "sd21: generate + nsfw + detect"
echo "============================================================"

# 1. sd14 已经生成，只补检测
run_detect 0 "${SD14}" "${RUN_SD14}" "sd14-dongman-detect-only"

# 2. sd15 生成 + NSFW + 检测
run_generate 1 "${SD15}" "${RUN_SD15}" "sd15-dongman"
run_nsfw "${RUN_SD15}" "sd15-dongman"
run_detect 1 "${SD15}" "${RUN_SD15}" "sd15-dongman"

# 3. sd21 生成 + NSFW + 检测
run_generate 1 "${SD21}" "${RUN_SD21}" "sd21-dongman"
run_nsfw "${RUN_SD21}" "sd21-dongman"
run_detect 1 "${SD21}" "${RUN_SD21}" "sd21-dongman"

echo
echo "============================================================"
echo "[DONE] T2S dongman partial pipeline finished."
echo "Outputs:"
echo "  ${RUN_SD14}/detect_t2s_oms"
echo "  ${RUN_SD15}"
echo "  ${RUN_SD21}"
echo "============================================================"