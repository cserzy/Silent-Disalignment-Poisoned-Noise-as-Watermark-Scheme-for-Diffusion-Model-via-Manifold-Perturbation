#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# PRC w_att -> aligned Gaussian OMS grid search on all models
# For each combo:
#   fit/apply OMS
#   generate on sd14/sd15/sd21
#   NSFW scoring
#   copy wm_meta into each run dir
#   PRC-OMS detect
# ============================================================

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19

PROMPTS=${ROOT}/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt

OMS_PY=${EXP_ROOT}/script-experiment/oms_repair_pt.py
GEN_PY=${EXP_ROOT}/script-experiment/gen_from_zT_bank_multi_models-1_19.py
DET_PY=${EXP_ROOT}/script-experiment/detect/detect_PRC_oms.py
NSFW_PY=${ROOT}/nsfw_score_report_ring_wm_only_exposed_only-12.29.py

LATDIR=${EXP_ROOT}/latents_experiment
IMGROOT=${EXP_ROOT}/imgs

SD14=${ROOT}/checkpoints/sd1-4-diffusers
SD15=${ROOT}/checkpoints/sd1-5-diffusers
SD21=${ROOT}/checkpoints/sd2-1-diffusers

# -------------------------
# PRC official repo path
# 如果真实路径不同，运行前 export PRC_REPO=/your/path
# -------------------------
PRC_REPO=${PRC_REPO:-${ROOT}/PRC-Watermark-main}

# -------------------------
# Input / target / meta
# -------------------------
SRC_WATT=${LATDIR}/generate_PRC_w_att_0_85_clip.pt
TARGET_GAUSS=${LATDIR}/generate_GAUSS_w_aligned_vis.pt

# 这里按你补充的要求：从一个已经存在的输出目录里拷贝 wm_meta
WM_META_TEMPLATE_DIR=${EXP_ROOT}/imgs/vis_sd15_PRC_w_att_oms_gauss_aligned_seed12345/wm_meta

# -------------------------
# Common params
# -------------------------
Q_SEED=12345
MATCH_TARGET_STD=1

STEPS=50
CFG=7.5
HEIGHT=512
WIDTH=512
N_PER_PROMPT=4
START_LATENT=0
GEN_DTYPE=fp16
FIT_DTYPE=fp32

NSFW_THRESHOLD=0.6
NSFW_SWEEP="0.2,0.3,0.4,0.5,0.6,0.7,0.8"
PRC_INV_STEPS=50
PRC_DTYPE=fp16

# -------------------------
# Grid
# 可按你需要改
# -------------------------
BLOCK_SIZES=(32 64 128)
ALPHAS=(0.10 0.20 0.30)

# -------------------------
# Helpers
# -------------------------
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
      echo "[ERROR] job failed: pid=$pid" >&2
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

process_one_nsfw() {
  local path="$1"

  echo "[NSFW] processing: $path" >&2

  if [ ! -d "$path" ]; then
    echo "[NSFW] skip, dir not found: $path" >&2
    return 0
  fi

  local manifest_file=""
  if [ -f "$path/sliced/manifest.csv" ]; then
    manifest_file="$path/sliced/manifest.csv"
  elif [ -f "$path/manifest.csv" ]; then
    manifest_file="$path/manifest.csv"
  else
    echo "[NSFW] skip, manifest not found: $path" >&2
    return 0
  fi

  local nsfw_report_dir="$path/nsfw_report"
  mkdir -p "$nsfw_report_dir"
  local log_file="$nsfw_report_dir/run.log"

  python "${NSFW_PY}" \
    --manifests "${manifest_file}" \
    --out_dir "${nsfw_report_dir}" \
    --report_out "${nsfw_report_dir}/report.xlsx" \
    --threshold "${NSFW_THRESHOLD}" \
    --sweep "${NSFW_SWEEP}" \
    >> "${log_file}" 2>&1

  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[NSFW] failed: $path (exit=$rc), log=${log_file}" >&2
    return $rc
  fi

  echo "[NSFW] done: $path" >&2
  return 0
}

run_one_combo() {
  local bs="$1"
  local alpha="$2"

  local alpha_tag
  alpha_tag=$(printf "%.2f" "${alpha}" | tr '.' 'p')

  local out_pt="${LATDIR}/generate_PRC_w_att_oms_gauss_aligned_b${bs}_a${alpha_tag}.pt"
  local out_q="${LATDIR}/oms_Q_PRC_w_att_to_gauss_aligned_b${bs}_a${alpha_tag}.pt"
  local out_json="${LATDIR}/oms_Q_PRC_w_att_to_gauss_aligned_b${bs}_a${alpha_tag}.json"

  local run_sd14="${IMGROOT}/vis_sd14_PRC_w_att_oms_gauss_aligned_b${bs}_a${alpha_tag}_seed12345"
  local run_sd15="${IMGROOT}/vis_sd15_PRC_w_att_oms_gauss_aligned_b${bs}_a${alpha_tag}_seed12345"
  local run_sd21="${IMGROOT}/vis_sd21_PRC_w_att_oms_gauss_aligned_b${bs}_a${alpha_tag}_seed12345"

  echo
  echo "============================================================"
  echo "[GRID] block_size=${bs} alpha=${alpha}"
  echo "============================================================"

  # ----------------------------------------------------------
  # 1) OMS fit/apply
  # ----------------------------------------------------------
  python "${OMS_PY}" \
    --mode fit_apply \
    --in_pt "${SRC_WATT}" \
    --target_pt "${TARGET_GAUSS}" \
    --out_pt "${out_pt}" \
    --out_q_pt "${out_q}" \
    --out_meta_json "${out_json}" \
    --q_seed "${Q_SEED}" \
    --block_size "${bs}" \
    --blend_alpha "${alpha}" \
    --match_target_std "${MATCH_TARGET_STD}" \
    --device cpu \
    --dtype "${FIT_DTYPE}" \
    --verbose || return 1

  # ----------------------------------------------------------
  # 2) Generation on 3 models
  # ----------------------------------------------------------
  pids=()

  run_bg 0 python "${GEN_PY}" \
    --model_id "${SD14}" \
    --prompts "${PROMPTS}" \
    --zT_pt "${out_pt}" \
    --outdir "${run_sd14}" \
    --steps "${STEPS}" --cfg "${CFG}" --height "${HEIGHT}" --width "${WIDTH}" \
    --n_per_prompt "${N_PER_PROMPT}" --start_latent "${START_LATENT}" \
    --dtype "${GEN_DTYPE}" --seed "${Q_SEED}" \
    --negative_prompt ""

  run_bg 1 python "${GEN_PY}" \
    --model_id "${SD15}" \
    --prompts "${PROMPTS}" \
    --zT_pt "${out_pt}" \
    --outdir "${run_sd15}" \
    --steps "${STEPS}" --cfg "${CFG}" --height "${HEIGHT}" --width "${WIDTH}" \
    --n_per_prompt "${N_PER_PROMPT}" --start_latent "${START_LATENT}" \
    --dtype "${GEN_DTYPE}" --seed "${Q_SEED}" \
    --negative_prompt ""

  run_bg 1 python "${GEN_PY}" \
    --model_id "${SD21}" \
    --prompts "${PROMPTS}" \
    --zT_pt "${out_pt}" \
    --outdir "${run_sd21}" \
    --steps "${STEPS}" --cfg "${CFG}" --height "${HEIGHT}" --width "${WIDTH}" \
    --n_per_prompt "${N_PER_PROMPT}" --start_latent "${START_LATENT}" \
    --dtype "${GEN_DTYPE}" --seed "${Q_SEED}" \
    --negative_prompt ""

  if ! wait_all_or_fail; then
    echo "[WARN] generation failed for combo b=${bs} a=${alpha}" >&2
    return 1
  fi

  # ----------------------------------------------------------
  # 3) NSFW scoring
  # ----------------------------------------------------------
  process_one_nsfw "${run_sd14}" || return 1
  process_one_nsfw "${run_sd15}" || return 1
  process_one_nsfw "${run_sd21}" || return 1

  # ----------------------------------------------------------
  # 4) Copy wm_meta into each run dir
  # ----------------------------------------------------------
  copy_wm_meta "${run_sd14}" || return 1
  copy_wm_meta "${run_sd15}" || return 1
  copy_wm_meta "${run_sd21}" || return 1

  # ----------------------------------------------------------
  # 5) PRC-OMS detect on 3 models
  # ----------------------------------------------------------
  pids=()

  run_bg 0 python "${DET_PY}" \
    --model_id "${SD14}" \
    --run_dir "${run_sd14}" \
    --out_dir "${run_sd14}/detect_prc_oms" \
    --wm_meta_dir "${run_sd14}/wm_meta" \
    --oms_q_pt "${out_q}" \
    --oms_meta_json "${out_json}" \
    --prc_repo "${PRC_REPO}" \
    --inv_steps "${PRC_INV_STEPS}" \
    --dtype "${PRC_DTYPE}" \
    --save_zt_oms \
    --save_zt_restored

  run_bg 1 python "${DET_PY}" \
    --model_id "${SD15}" \
    --run_dir "${run_sd15}" \
    --out_dir "${run_sd15}/detect_prc_oms" \
    --wm_meta_dir "${run_sd15}/wm_meta" \
    --oms_q_pt "${out_q}" \
    --oms_meta_json "${out_json}" \
    --prc_repo "${PRC_REPO}" \
    --inv_steps "${PRC_INV_STEPS}" \
    --dtype "${PRC_DTYPE}" \
    --save_zt_oms \
    --save_zt_restored

  run_bg 1 python "${DET_PY}" \
    --model_id "${SD21}" \
    --run_dir "${run_sd21}" \
    --out_dir "${run_sd21}/detect_prc_oms" \
    --wm_meta_dir "${run_sd21}/wm_meta" \
    --oms_q_pt "${out_q}" \
    --oms_meta_json "${out_json}" \
    --prc_repo "${PRC_REPO}" \
    --inv_steps "${PRC_INV_STEPS}" \
    --dtype "${PRC_DTYPE}" \
    --save_zt_oms \
    --save_zt_restored

  if ! wait_all_or_fail; then
    echo "[WARN] detect failed for combo b=${bs} a=${alpha}" >&2
    return 1
  fi

  echo "[DONE] combo finished: b=${bs}, a=${alpha}"
}

FAIL=0
for bs in "${BLOCK_SIZES[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    run_one_combo "${bs}" "${alpha}" || FAIL=1
  done
done

echo
echo "============================================================"
if [[ "${FAIL}" -eq 0 ]]; then
  echo "[ALL DONE] PRC grid search finished successfully."
else
  echo "[WARN] Some PRC grid jobs failed."
fi
echo "============================================================"

exit "${FAIL}"