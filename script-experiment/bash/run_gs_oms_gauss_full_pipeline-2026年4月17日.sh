#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# GS -> aligned Gaussian OMS full pipeline
#   Stage 1: fit/apply OMS for GS_w and GS_w_att (to aligned Gaussian target)
#   Stage 2: generate images on sd14/sd15/sd21 using repaired pt
#   Stage 3: NSFW scoring for all 6 run dirs
#   Stage 4: OMS-aware GS detection for all 6 run dirs
# ============================================================

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19

PROMPTS=${ROOT}/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt

OMS_PY=${EXP_ROOT}/script-experiment/oms_repair_pt.py
GEN_PY=${EXP_ROOT}/script-experiment/gen_from_zT_bank_multi_models-1_19.py
DET_PY=${EXP_ROOT}/script-experiment/detect/detect_GS_oms.py
NSFW_PY=${ROOT}/nsfw_score_report_ring_wm_only_exposed_only-12.29.py

LATDIR=${EXP_ROOT}/latents_experiment
IMGROOT=${EXP_ROOT}/imgs

SD14=${ROOT}/checkpoints/sd1-4-diffusers
SD15=${ROOT}/checkpoints/sd1-5-diffusers
SD21=${ROOT}/checkpoints/sd2-1-diffusers

# -------------------------
# Latent files
# -------------------------
SRC_W=${LATDIR}/generate_GS_w.pt
SRC_WATT=${LATDIR}/generate_GS_w_att.pt
TARGET_GAUSS=${LATDIR}/generate_GAUSS_w_aligned_vis.pt

OMS_W_PT=${LATDIR}/generate_GS_w_oms_gauss_aligned.pt
OMS_WATT_PT=${LATDIR}/generate_GS_w_att_oms_gauss_aligned.pt

OMS_W_Q=${LATDIR}/oms_Q_GS_w_to_gauss_aligned.pt
OMS_W_Q_JSON=${LATDIR}/oms_Q_GS_w_to_gauss_aligned.json

OMS_WATT_Q=${LATDIR}/oms_Q_GS_w_att_to_gauss_aligned.pt
OMS_WATT_Q_JSON=${LATDIR}/oms_Q_GS_w_att_to_gauss_aligned.json

# -------------------------
# Common params
# -------------------------
Q_SEED=12345
BLOCK_SIZE=64
BLEND_ALPHA=0.2
MATCH_TARGET_STD=1

STEPS=50
CFG=7.5
HEIGHT=512
WIDTH=512
N_PER_PROMPT=4
START_LATENT=0
GEN_DTYPE=fp16
FIT_DTYPE=fp32
INV_STEPS=50

NSFW_THRESHOLD=0.6
NSFW_SWEEP="0.2,0.3,0.4,0.5,0.6,0.7,0.8"
MAX_NSFW_JOBS=4

# -------------------------
# Output run dirs
# -------------------------
RUN_SD14_W=${IMGROOT}/vis_sd14_GS_w_oms_gauss_aligned_seed12345
RUN_SD15_W=${IMGROOT}/vis_sd15_GS_w_oms_gauss_aligned_seed12345
RUN_SD21_W=${IMGROOT}/vis_sd21_GS_w_oms_gauss_aligned_seed12345

RUN_SD14_WATT=${IMGROOT}/vis_sd14_GS_w_att_oms_gauss_aligned_seed12345
RUN_SD15_WATT=${IMGROOT}/vis_sd15_GS_w_att_oms_gauss_aligned_seed12345
RUN_SD21_WATT=${IMGROOT}/vis_sd21_GS_w_att_oms_gauss_aligned_seed12345

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

# ============================================================
# Stage 1: OMS fit/apply (CPU, sequential)
# ============================================================
echo
echo "============================================================"
echo "[STAGE 1] OMS fit/apply: GS_w -> aligned Gaussian"
echo "============================================================"

python "${OMS_PY}" \
  --mode fit_apply \
  --in_pt "${SRC_W}" \
  --target_pt "${TARGET_GAUSS}" \
  --out_pt "${OMS_W_PT}" \
  --out_q_pt "${OMS_W_Q}" \
  --out_meta_json "${OMS_W_Q_JSON}" \
  --q_seed "${Q_SEED}" \
  --block_size "${BLOCK_SIZE}" \
  --blend_alpha "${BLEND_ALPHA}" \
  --match_target_std "${MATCH_TARGET_STD}" \
  --device cpu \
  --dtype "${FIT_DTYPE}" \
  --verbose || exit 1

echo
echo "============================================================"
echo "[STAGE 1] OMS fit/apply: GS_w_att -> aligned Gaussian"
echo "============================================================"

python "${OMS_PY}" \
  --mode fit_apply \
  --in_pt "${SRC_WATT}" \
  --target_pt "${TARGET_GAUSS}" \
  --out_pt "${OMS_WATT_PT}" \
  --out_q_pt "${OMS_WATT_Q}" \
  --out_meta_json "${OMS_WATT_Q_JSON}" \
  --q_seed "${Q_SEED}" \
  --block_size "${BLOCK_SIZE}" \
  --blend_alpha "${BLEND_ALPHA}" \
  --match_target_std "${MATCH_TARGET_STD}" \
  --device cpu \
  --dtype "${FIT_DTYPE}" \
  --verbose || exit 1

# ============================================================
# Stage 2: generation (6 jobs)
# ============================================================
echo
echo "============================================================"
echo "[STAGE 2] Generate images for 6 runs"
echo "============================================================"

COMMON_GEN_ARGS=(
  --prompts "${PROMPTS}"
  --steps "${STEPS}" --cfg "${CFG}" --height "${HEIGHT}" --width "${WIDTH}"
  --n_per_prompt "${N_PER_PROMPT}" --start_latent "${START_LATENT}"
  --dtype "${GEN_DTYPE}" --seed "${Q_SEED}"
  --negative_prompt ""
)

run_gen () {
  local gpu="$1"
  local model_id="$2"
  local outdir="$3"
  local zt_pt="$4"

  local log="${outdir}.runlog.txt"
  mkdir -p "$(dirname "$outdir")"

  run_bg "${gpu}" \
    python "${GEN_PY}" \
      --model_id "${model_id}" \
      --zT_pt "${zt_pt}" \
      --outdir "${outdir}" \
      "${COMMON_GEN_ARGS[@]}" \
    > "${log}" 2>&1
}

# w
run_gen 0 "${SD14}" "${RUN_SD14_W}" "${OMS_W_PT}"
run_gen 0 "${SD15}" "${RUN_SD15_W}" "${OMS_W_PT}"
run_gen 1 "${SD21}" "${RUN_SD21_W}" "${OMS_W_PT}"

# w_att
run_gen 1 "${SD14}" "${RUN_SD14_WATT}" "${OMS_WATT_PT}"
run_gen 1 "${SD15}" "${RUN_SD15_WATT}" "${OMS_WATT_PT}"
run_gen 1 "${SD21}" "${RUN_SD21_WATT}" "${OMS_WATT_PT}"

if ! wait_all_or_fail; then
  echo "[DONE] Some generation jobs failed. Check logs:" >&2
  echo "       ${IMGROOT}/vis_sd*_GS_*_oms_gauss_aligned_seed12345.runlog.txt" >&2
  exit 1
fi

echo "[DONE] All generation jobs finished OK." >&2

# ============================================================
# Stage 3: NSFW scoring
# ============================================================
echo
echo "============================================================"
echo "[STAGE 3] NSFW scoring"
echo "============================================================"

paths=(
"${RUN_SD14_W}"
"${RUN_SD15_W}"
"${RUN_SD21_W}"
"${RUN_SD14_WATT}"
"${RUN_SD15_WATT}"
"${RUN_SD21_WATT}"
)

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

running=0
nsfw_failed=0
pids=()

for path in "${paths[@]}"; do
  process_one_nsfw "$path" &
  pids+=($!)
  running=$((running + 1))

  if [ "$running" -ge "$MAX_NSFW_JOBS" ]; then
    if ! wait -n; then
      nsfw_failed=1
    fi
    running=$((running - 1))
  fi
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    nsfw_failed=1
  fi
done

if [ "$nsfw_failed" -ne 0 ]; then
  echo "[WARN] NSFW stage finished with some failures. Check nsfw_report/run.log." >&2
else
  echo "[DONE] NSFW stage finished OK." >&2
fi

# ============================================================
# Stage 4: OMS-aware GS detection
# ============================================================
echo
echo "============================================================"
echo "[STAGE 4] OMS-aware GS detection"
echo "============================================================"

run_det () {
  local gpu="$1"
  local model_id="$2"
  local rundir="$3"
  local q_pt="$4"
  local q_json="$5"

  local outdir="${rundir}/detect_gs_oms"
  local log="${outdir}/detect_gs_oms.runlog.txt"

  mkdir -p "${outdir}"

  run_bg "${gpu}" \
    python "${DET_PY}" \
      --model_id "${model_id}" \
      --run_dir "${rundir}" \
      --out_dir "${outdir}" \
      --oms_q_pt "${q_pt}" \
      --oms_meta_json "${q_json}" \
      --inv_steps "${INV_STEPS}" \
      --dtype "${GEN_DTYPE}" \
      --save_zt_oms \
      --save_zt_restored \
    > "${log}" 2>&1
}

# w
run_det 0 "${SD14}" "${RUN_SD14_W}" "${OMS_W_Q}" "${OMS_W_Q_JSON}"
run_det 0 "${SD15}" "${RUN_SD15_W}" "${OMS_W_Q}" "${OMS_W_Q_JSON}"
run_det 1 "${SD21}" "${RUN_SD21_W}" "${OMS_W_Q}" "${OMS_W_Q_JSON}"

# w_att
run_det 1 "${SD14}" "${RUN_SD14_WATT}" "${OMS_WATT_Q}" "${OMS_WATT_Q_JSON}"
run_det 1 "${SD15}" "${RUN_SD15_WATT}" "${OMS_WATT_Q}" "${OMS_WATT_Q_JSON}"
run_det 1 "${SD21}" "${RUN_SD21_WATT}" "${OMS_WATT_Q}" "${OMS_WATT_Q_JSON}"

if ! wait_all_or_fail; then
  echo "[DONE] Some GS-OMS detect jobs failed. Check logs:" >&2
  echo "       ${IMGROOT}/vis_sd*_GS_*_oms_gauss_aligned_seed12345/detect_gs_oms/detect_gs_oms.runlog.txt" >&2
  exit 1
fi

echo "[DONE] All GS-OMS detect jobs finished OK." >&2

echo
echo "============================================================"
echo "[ALL DONE] GS -> aligned Gaussian OMS full pipeline finished."
echo "Outputs under:"
echo "  ${IMGROOT}/vis_sd*_GS_*_oms_gauss_aligned_seed12345/"
echo "============================================================"