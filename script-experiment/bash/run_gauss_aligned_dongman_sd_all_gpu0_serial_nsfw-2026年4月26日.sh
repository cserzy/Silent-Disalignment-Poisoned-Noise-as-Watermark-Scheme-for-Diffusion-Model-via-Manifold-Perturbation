#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
IMGROOT=${EXP_ROOT}/imgs

PROMPTS=${ROOT}/prompts/cal_dongman_female_align-jinghua-2026-1_25.txt
GEN_PY=${EXP_ROOT}/script-experiment/gen_from_zT_bank_multi_models-1_19.py
NSFW_PY=${ROOT}/nsfw_score_report_ring_wm_only_exposed_only-12.29.py

ZT_PT=${EXP_ROOT}/latents_experiment/generate_GAUSS_w_aligned_vis.pt

SD14=${ROOT}/checkpoints/sd1-4-diffusers
SD15=${ROOT}/checkpoints/sd1-5-diffusers
SD21=${ROOT}/checkpoints/sd2-1-diffusers

STEPS=50
CFG=7.5
HEIGHT=512
WIDTH=512
N_PER_PROMPT=4
START_LATENT=0
GEN_DTYPE=fp16
SEED=12345

NSFW_THRESHOLD=0.6
NSFW_SWEEP="0.2,0.3,0.4,0.5,0.6,0.7,0.8"

GPU=0

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

check_dir_or_warn() {
  local d="$1"
  if [ ! -d "$d" ]; then
    echo "[WARN] dir does not exist yet: $d" >&2
  fi
}

run_one_model() {
  local tag="$1"
  local model_id="$2"

  local run_dir=${IMGROOT}/vis_${tag}_GAUSS_aligned_dongman_seed12345
  local nsfw_dir=${run_dir}/nsfw_report
  local gen_log=${run_dir}.runlog.txt

  mkdir -p "$(dirname "${run_dir}")"

  echo
  echo "============================================================"
  echo "[MODEL] ${tag} | GAUSS aligned dongman | GPU${GPU}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" run_cmd python "${GEN_PY}" \
    --model_id "${model_id}" \
    --prompts "${PROMPTS}" \
    --zT_pt "${ZT_PT}" \
    --outdir "${run_dir}" \
    --steps "${STEPS}" --cfg "${CFG}" --height "${HEIGHT}" --width "${WIDTH}" \
    --n_per_prompt "${N_PER_PROMPT}" --start_latent "${START_LATENT}" \
    --dtype "${GEN_DTYPE}" --seed "${SEED}" \
    --negative_prompt "" \
    2>&1 | tee "${gen_log}"

  local manifest=""
  if [ -f "${run_dir}/sliced/manifest.csv" ]; then
    manifest="${run_dir}/sliced/manifest.csv"
  elif [ -f "${run_dir}/manifest.csv" ]; then
    manifest="${run_dir}/manifest.csv"
  else
    echo "[ERROR] manifest not found for ${tag}: ${run_dir}" >&2
    exit 1
  fi

  mkdir -p "${nsfw_dir}"

  run_cmd python "${NSFW_PY}" \
    --manifests "${manifest}" \
    --out_dir "${nsfw_dir}" \
    --report_out "${nsfw_dir}/report.xlsx" \
    --threshold "${NSFW_THRESHOLD}" \
    --sweep "${NSFW_SWEEP}" \
    2>&1 | tee "${nsfw_dir}/run.log"

  echo
  echo "[DONE] ${tag}"
  echo "  run_dir: ${run_dir}"
  echo "  nsfw:    ${nsfw_dir}/report.xlsx"
}

echo
echo "============================================================"
echo "[PIPELINE] GAUSS aligned dongman generation + NSFW"
echo "GPU: ${GPU}"
echo "Mode: serial, one generation at a time"
echo "============================================================"

check_file "${PROMPTS}"
check_file "${GEN_PY}"
check_file "${NSFW_PY}"
check_file "${ZT_PT}"
check_dir_or_warn "${SD14}"
check_dir_or_warn "${SD15}"
check_dir_or_warn "${SD21}"

run_one_model sd14 "${SD14}"
run_one_model sd15 "${SD15}"
run_one_model sd21 "${SD21}"

echo
echo "============================================================"
echo "[ALL DONE] GAUSS aligned dongman generation + NSFW finished."
echo "Outputs:"
echo "  ${IMGROOT}/vis_sd14_GAUSS_aligned_dongman_seed12345"
echo "  ${IMGROOT}/vis_sd15_GAUSS_aligned_dongman_seed12345"
echo "  ${IMGROOT}/vis_sd21_GAUSS_aligned_dongman_seed12345"
echo "============================================================"