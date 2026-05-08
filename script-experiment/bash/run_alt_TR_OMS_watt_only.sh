#!/usr/bin/env bash
set -uo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate Hijacking

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
cd "${EXP_ROOT}" || exit 1

GEN=${EXP_ROOT}/script-experiment/gen_from_zT_bank_alt_diffusion-1_19.py
BLACK_DET=${EXP_ROOT}/script-experiment/detect/detect_black_ratio_alt.py
TR_OMS_DET=${EXP_ROOT}/script-experiment/detect/detect_TR_alt_oms.py

MODEL_ID=${ROOT}/checkpoints/AltDiffusion-fp16
PROMPTS=${ROOT}/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt

# 只用这一组
ZT_TR_WATT_OMS=${EXP_ROOT}/latents_experiment/generate_TR_w_att_oms_gauss_aligned.pt
OMS_Q_PT=${EXP_ROOT}/latents_experiment/oms_Q_TR_w_att_to_gauss_aligned.pt
OMS_Q_JSON=${EXP_ROOT}/latents_experiment/oms_Q_TR_w_att_to_gauss_aligned.json

SEED=12345
GEN_STEPS=50
GEN_CFG=7.5
HEIGHT=512
WIDTH=512
DTYPE=fp16
N_PER_PROMPT=4
START_LATENT=0

# TR-OMS Alt detect params
TR_GUIDANCE=1.0
TR_STEPS=50
TR_INV_STEPS=50
TR_FP16=1

# 输出目录
RUN_SAFEOFF=${EXP_ROOT}/imgs/vis_alt_tr_oms_w_att_seed${SEED}
RUN_SAFEON=${EXP_ROOT}/imgs/vis_alt_tr_oms_safeon_w_att_seed${SEED}

need_file () {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] missing file: $1" >&2
    exit 1
  fi
}

need_file "${ZT_TR_WATT_OMS}"
need_file "${OMS_Q_PT}"
need_file "${OMS_Q_JSON}"
need_file "${PROMPTS}"

pids=()

run_gen_job () {
  local gpu="$1"
  local outdir="$2"
  local disable_sc="$3"   # 1=disable, 0=enable

  mkdir -p "${outdir}"

  echo "[LAUNCH][GEN][GPU${gpu}] -> ${outdir} | disable_safety_checker=${disable_sc}" >&2

  if [[ "${disable_sc}" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" \
      python "${GEN}" \
        --model_id "${MODEL_ID}" \
        --prompts "${PROMPTS}" \
        --zT_pt "${ZT_TR_WATT_OMS}" \
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
        --zT_pt "${ZT_TR_WATT_OMS}" \
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

run_tr_oms_job () {
  local gpu="$1"
  local run_dir="$2"
  local out_dir="${run_dir}/detect_tr_alt_oms"

  mkdir -p "${out_dir}"

  echo "[LAUNCH][TR-OMS][GPU${gpu}] ${run_dir}" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "${TR_OMS_DET}" \
      --img_dir "${run_dir}" \
      --model_id "${MODEL_ID}" \
      --mode img \
      --detect_prompt empty \
      --guidance_scale ${TR_GUIDANCE} \
      --steps ${TR_STEPS} \
      --inv_steps ${TR_INV_STEPS} \
      --fp16 ${TR_FP16} \
      --out_dir "${out_dir}" \
      --save_zt_oms \
      --save_zt_restored \
      --oms_q_pt "${OMS_Q_PT}" \
      --oms_meta_json "${OMS_Q_JSON}" \
    > "${out_dir}/detect_tr_alt_oms.runlog.txt" 2>&1 &

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

echo "[INFO] Stage 1: generate safe-off TR-OMS run (for watermark detection)..." >&2
run_gen_job 0 "${RUN_SAFEOFF}" 1
wait_all || exit 1

echo "[INFO] Stage 2: generate safe-on TR-OMS run (for black-rate / safety-trigger rate)..." >&2
run_gen_job 0 "${RUN_SAFEON}" 0
wait_all || exit 1

echo "[INFO] Stage 3: black-rate detection on safe-on run..." >&2
run_black_job "${RUN_SAFEON}"
wait_all || exit 1

echo "[INFO] Stage 4: TR-OMS watermark detection on safe-off run..." >&2
run_tr_oms_job 0 "${RUN_SAFEOFF}"
wait_all || exit 1

echo "[DONE] Alt TR-OMS pipeline finished successfully." >&2
echo "[DONE] Safe-off output (for TR-OMS detection):" >&2
echo "  ${RUN_SAFEOFF}" >&2
echo "[DONE] Safe-on output (for black-rate):" >&2
echo "  ${RUN_SAFEON}" >&2
echo "[DONE] Black-rate result:" >&2
echo "  ${RUN_SAFEON}/black_detect" >&2
echo "[DONE] TR-OMS detect result:" >&2
echo "  ${RUN_SAFEOFF}/detect_tr_alt_oms" >&2