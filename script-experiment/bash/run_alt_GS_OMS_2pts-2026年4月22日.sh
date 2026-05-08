#!/usr/bin/env bash
set -uo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate Hijacking

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
cd "${EXP_ROOT}" || exit 1

GEN=${EXP_ROOT}/script-experiment/gen_from_zT_bank_alt_diffusion-1_19.py
BLACK_DET=${EXP_ROOT}/script-experiment/detect/detect_black_ratio_alt.py
GS_OMS_DET=${EXP_ROOT}/script-experiment/detect/detect_GS_alt_oms.py

MODEL_ID=${ROOT}/checkpoints/AltDiffusion-fp16
PROMPTS=${ROOT}/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt

# =========================
# GS OMS latent / Q / meta
# =========================
ZT_GS_OMS_W=${EXP_ROOT}/latents_experiment/generate_GS_w_oms_gauss_aligned.pt
ZT_GS_OMS_WATT=${EXP_ROOT}/latents_experiment/generate_GS_w_att_oms_gauss_aligned.pt

OMS_Q_W_PT=${EXP_ROOT}/latents_experiment/oms_Q_GS_w_to_gauss_aligned.pt
OMS_Q_W_JSON=${EXP_ROOT}/latents_experiment/oms_Q_GS_w_to_gauss_aligned.json

OMS_Q_WATT_PT=${EXP_ROOT}/latents_experiment/oms_Q_GS_w_att_to_gauss_aligned.pt
OMS_Q_WATT_JSON=${EXP_ROOT}/latents_experiment/oms_Q_GS_w_att_to_gauss_aligned.json

# =========================
# Generation params
# =========================
SEED=12345
GEN_STEPS=50
GEN_CFG=7.5
HEIGHT=512
WIDTH=512
DTYPE=fp16
N_PER_PROMPT=4
START_LATENT=0

# =========================
# GS-OMS Alt detect params
# =========================
INV_STEPS=20

# =========================
# Output dirs
# =========================
# safe-off: used for watermark detection
RUN_SAFEOFF_W=${EXP_ROOT}/imgs/vis_alt_gs_oms_w_seed${SEED}
RUN_SAFEOFF_WATT=${EXP_ROOT}/imgs/vis_alt_gs_oms_w_att_seed${SEED}

# safe-on: used for black-rate detection
RUN_SAFEON_W=${EXP_ROOT}/imgs/vis_alt_gs_oms_safeon_w_seed${SEED}
RUN_SAFEON_WATT=${EXP_ROOT}/imgs/vis_alt_gs_oms_safeon_w_att_seed${SEED}

# =========================
# Basic file checks
# =========================
need_file () {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] missing file: $1" >&2
    exit 1
  fi
}

need_file "${ZT_GS_OMS_W}"
need_file "${ZT_GS_OMS_WATT}"
need_file "${OMS_Q_W_PT}"
need_file "${OMS_Q_W_JSON}"
need_file "${OMS_Q_WATT_PT}"
need_file "${OMS_Q_WATT_JSON}"
need_file "${PROMPTS}"

pids=()

run_gen_job () {
  local gpu="$1"
  local tag="$2"
  local zt_pt="$3"
  local outdir="$4"
  local disable_sc="$5"   # 1=disable safety checker, 0=enable safety checker

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

run_gs_oms_job () {
  local gpu="$1"
  local run_dir="$2"
  local out_dir="$3"
  local oms_q_pt="$4"
  local oms_meta_json="$5"

  mkdir -p "${out_dir}"

  echo "[LAUNCH][GS-OMS][GPU${gpu}] ${run_dir}" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "${GS_OMS_DET}" \
      --model_id "${MODEL_ID}" \
      --run_dir "${run_dir}" \
      --out_dir "${out_dir}" \
      --oms_q_pt "${oms_q_pt}" \
      --oms_meta_json "${oms_meta_json}" \
      --dtype ${DTYPE} \
      --inv_steps ${INV_STEPS} \
      --save_zt_oms \
      --save_zt_restored \
    > "${out_dir}/detect_gs_alt_oms.runlog.txt" 2>&1 &

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

echo "[INFO] Stage 1: generate safe-off GS-OMS runs (for watermark detection)..." >&2
run_gen_job 0 "gs_oms_w_safeoff" "${ZT_GS_OMS_W}" "${RUN_SAFEOFF_W}" 1
run_gen_job 1 "gs_oms_w_att_safeoff" "${ZT_GS_OMS_WATT}" "${RUN_SAFEOFF_WATT}" 1
wait_all || exit 1

echo "[INFO] Stage 2: generate safe-on GS-OMS runs (for black-rate / safety-trigger rate)..." >&2
run_gen_job 0 "gs_oms_w_safeon" "${ZT_GS_OMS_W}" "${RUN_SAFEON_W}" 0
run_gen_job 1 "gs_oms_w_att_safeon" "${ZT_GS_OMS_WATT}" "${RUN_SAFEON_WATT}" 0
wait_all || exit 1

echo "[INFO] Stage 3: black-rate detection on safe-on runs..." >&2
run_black_job "${RUN_SAFEON_W}"
run_black_job "${RUN_SAFEON_WATT}"
wait_all || exit 1

echo "[INFO] Stage 4: GS-OMS watermark detection on safe-off runs..." >&2
run_gs_oms_job 0 "${RUN_SAFEOFF_W}" "${RUN_SAFEOFF_W}/detect_gs_alt_oms" "${OMS_Q_W_PT}" "${OMS_Q_W_JSON}"
run_gs_oms_job 1 "${RUN_SAFEOFF_WATT}" "${RUN_SAFEOFF_WATT}/detect_gs_alt_oms" "${OMS_Q_WATT_PT}" "${OMS_Q_WATT_JSON}"
wait_all || exit 1

echo "[DONE] Alt GS-OMS full pipeline finished successfully." >&2
echo "[DONE] Safe-off outputs (for OMS watermark detection):" >&2
echo "  ${RUN_SAFEOFF_W}" >&2
echo "  ${RUN_SAFEOFF_WATT}" >&2
echo "[DONE] Safe-on outputs (for black-rate):" >&2
echo "  ${RUN_SAFEON_W}" >&2
echo "  ${RUN_SAFEON_WATT}" >&2
echo "[DONE] Black-rate results:" >&2
echo "  ${RUN_SAFEON_W}/black_detect" >&2
echo "  ${RUN_SAFEON_WATT}/black_detect" >&2
echo "[DONE] GS-OMS detect results:" >&2
echo "  ${RUN_SAFEOFF_W}/detect_gs_alt_oms" >&2
echo "  ${RUN_SAFEOFF_WATT}/detect_gs_alt_oms" >&2