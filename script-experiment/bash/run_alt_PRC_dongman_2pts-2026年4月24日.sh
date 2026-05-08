#!/usr/bin/env bash
set -uo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate Hijacking

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
cd "${EXP_ROOT}" || exit 1

GEN=${EXP_ROOT}/script-experiment/gen_from_zT_bank_alt_diffusion-1_19.py
BLACK_DET=${EXP_ROOT}/script-experiment/detect/detect_black_ratio_alt.py
PRC_DET=${EXP_ROOT}/script-experiment/detect/prc_detect_alt_global_official_align.py

MODEL_ID=${ROOT}/checkpoints/AltDiffusion-fp16
PROMPTS=${ROOT}/prompts/cal_dongman_female_align-jinghua-2026-1_25.txt

PRC_REPO=${ROOT}/PRC-Watermark-main

# 已验证适配的 PRC w
ZT_PRC_W=${EXP_ROOT}/latents_experiment/generate_PRC_w_0_84.pt
PRC_W_WM_META=${EXP_ROOT}/imgs/vis_sd14_PRC_w_0_84_seed12345/wm_meta

# 动漫数据集对应的 PRC w_att
ZT_PRC_WATT=${EXP_ROOT}/latents_experiment-number/generate_PRC_w_att_0_85_dongman.pt
PRC_WATT_WM_META=${EXP_ROOT}/imgs/vis_sd14_PRC_w_att_0_85_clip_seed12345/wm_meta

SEED=12345
GEN_STEPS=50
GEN_CFG=7.5
HEIGHT=512
WIDTH=512
DTYPE=fp16
N_PER_PROMPT=4
START_LATENT=0

# PRC Alt detect params
INV_STEPS=20
MAX_BP_ITER=1000
FPR=0.01
VARIANCES=1.5

# safe-off: no safety checker, used for PRC watermark detection
RUN_SAFEOFF_W=${EXP_ROOT}/imgs/vis_alt_dongman_prc_w_0_84_seed${SEED}
RUN_SAFEOFF_WATT=${EXP_ROOT}/imgs/vis_alt_dongman_prc_w_att_seed${SEED}

# safe-on: safety checker enabled, used for black-rate detection
RUN_SAFEON_W=${EXP_ROOT}/imgs/vis_alt_dongman_prc_safeon_w_0_84_seed${SEED}
RUN_SAFEON_WATT=${EXP_ROOT}/imgs/vis_alt_dongman_prc_safeon_w_att_seed${SEED}

pids=()

run_gen_job () {
  local gpu="$1"
  local tag="$2"
  local zt_pt="$3"
  local outdir="$4"
  local disable_sc="${5:-0}"

  mkdir -p "${outdir}"

  echo "[LAUNCH][GEN][GPU${gpu}] ${tag} -> ${outdir} | disable_safety_checker=${disable_sc}" >&2

  if [[ "${disable_sc}" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" \
      python "${GEN}" \
        --model_id "${MODEL_ID}" \
        --prompts "${PROMPTS}" \
        --zT_pt "${zt_pt}" \
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
    CUDA_VISIBLE_DEVICES="${gpu}" \
      python "${GEN}" \
        --model_id "${MODEL_ID}" \
        --prompts "${PROMPTS}" \
        --zT_pt "${zt_pt}" \
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

run_prc_job () {
  local gpu="$1"
  local run_dir="$2"
  local wm_meta_dir="$3"

  local out_dir="${run_dir}/detect_prc_alt"
  local out_csv="${out_dir}/detect_results_prcGLOBAL_alt.csv"

  mkdir -p "${out_dir}"

  echo "[LAUNCH][PRC][GPU${gpu}] ${run_dir}" >&2
  echo "[INFO][PRC] wm_meta=${wm_meta_dir}" >&2

  CUDA_VISIBLE_DEVICES="${gpu}" \
    /home/yancy/.conda/envs/Hijacking/bin/python -s "${PRC_DET}" \
      --run_dir "${run_dir}" \
      --model_id "${MODEL_ID}" \
      --wm_meta_dir "${wm_meta_dir}" \
      --copy_wm_meta_to_run_dir \
      --prc_repo "${PRC_REPO}" \
      --inv_steps ${INV_STEPS} \
      --max_bp_iter ${MAX_BP_ITER} \
      --fpr ${FPR} \
      --variances ${VARIANCES} \
      --save_zt \
      --out_csv "${out_csv}" \
    > "${out_dir}/detect_prc_alt.runlog.txt" 2>&1 &

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

echo "[INFO] Stage 0: check paths..." >&2
echo "  PROMPTS=${PROMPTS}" >&2
echo "  ZT_PRC_W=${ZT_PRC_W}" >&2
echo "  PRC_W_WM_META=${PRC_W_WM_META}" >&2
echo "  ZT_PRC_WATT=${ZT_PRC_WATT}" >&2
echo "  PRC_WATT_WM_META=${PRC_WATT_WM_META}" >&2

echo "[INFO] Stage 1: generate safe-off PRC runs..." >&2
run_gen_job 0 "dongman_prc_w_0_84_safeoff" "${ZT_PRC_W}" "${RUN_SAFEOFF_W}" 1
run_gen_job 1 "dongman_prc_w_att_safeoff" "${ZT_PRC_WATT}" "${RUN_SAFEOFF_WATT}" 1
wait_all || exit 1

echo "[INFO] Stage 2: generate safe-on PRC runs..." >&2
run_gen_job 0 "dongman_prc_w_0_84_safeon" "${ZT_PRC_W}" "${RUN_SAFEON_W}" 0
run_gen_job 1 "dongman_prc_w_att_safeon" "${ZT_PRC_WATT}" "${RUN_SAFEON_WATT}" 0
wait_all || exit 1

echo "[INFO] Stage 3: black-rate detection on safe-on runs..." >&2
run_black_job "${RUN_SAFEON_W}"
run_black_job "${RUN_SAFEON_WATT}"
wait_all || exit 1

echo "[INFO] Stage 4: PRC watermark detection on safe-off runs..." >&2
run_prc_job 0 "${RUN_SAFEOFF_W}" "${PRC_W_WM_META}"
run_prc_job 1 "${RUN_SAFEOFF_WATT}" "${PRC_WATT_WM_META}"
wait_all || exit 1

echo "[DONE] Alt Dongman PRC pipeline finished successfully." >&2
echo "[DONE] Safe-off outputs for PRC detection:" >&2
echo "  ${RUN_SAFEOFF_W}" >&2
echo "  ${RUN_SAFEOFF_WATT}" >&2
echo "[DONE] Safe-on outputs for black-rate:" >&2
echo "  ${RUN_SAFEON_W}" >&2
echo "  ${RUN_SAFEON_WATT}" >&2
echo "[DONE] Black-rate results:" >&2
echo "  ${RUN_SAFEON_W}/black_detect" >&2
echo "  ${RUN_SAFEON_WATT}/black_detect" >&2
echo "[DONE] PRC detect results:" >&2
echo "  ${RUN_SAFEOFF_W}/detect_prc_alt" >&2
echo "  ${RUN_SAFEOFF_WATT}/detect_prc_alt" >&2