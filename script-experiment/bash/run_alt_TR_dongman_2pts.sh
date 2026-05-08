#!/usr/bin/env bash
set -uo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate Hijacking

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
cd "${EXP_ROOT}" || exit 1

GEN=${EXP_ROOT}/script-experiment/gen_from_zT_bank_alt_diffusion-1_19.py
BLACK_DET=${EXP_ROOT}/script-experiment/detect/detect_black_ratio_alt.py
TR_DET=${EXP_ROOT}/script-experiment/detect/detect_TR_alt.py

MODEL_ID=${ROOT}/checkpoints/AltDiffusion-fp16
PROMPTS=${ROOT}/prompts/cal_dongman_female_align-jinghua-2026-1_25.txt

ZT_TR_W=${ROOT}/experiment-1_19-sd3/latents_experiment/generate_TR_w.pt
ZT_TR_WATT=${EXP_ROOT}/latents_experiment-number/generate_TR_w_att_0_88_dongman.pt

SEED=12345
GEN_STEPS=50
GEN_CFG=7.5
HEIGHT=512
WIDTH=512
DTYPE=fp16
N_PER_PROMPT=4
START_LATENT=0

TR_GUIDANCE=1.0
TR_STEPS=20
TR_INV_STEPS=20
TR_FP16=1

RUN_SAFEOFF_W=${EXP_ROOT}/imgs/vis_alt_dongman_tr_w_seed${SEED}
RUN_SAFEOFF_WATT=${EXP_ROOT}/imgs/vis_alt_dongman_tr_w_att_seed${SEED}

RUN_SAFEON_W=${EXP_ROOT}/imgs/vis_alt_dongman_tr_safeon_w_seed${SEED}
RUN_SAFEON_WATT=${EXP_ROOT}/imgs/vis_alt_dongman_tr_safeon_w_att_seed${SEED}

pids=()

run_gen_job () {
  local gpu="$1"
  local tag="$2"
  local zt_pt="$3"
  local outdir="$4"
  local disable_sc="$5"

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

run_tr_job () {
  local gpu="$1"
  local run_dir="$2"
  local out_dir="${run_dir}/detect_tr_alt"

  mkdir -p "${out_dir}"

  echo "[LAUNCH][TR][GPU${gpu}] ${run_dir}" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "${TR_DET}" \
      --img_dir "${run_dir}" \
      --model_id "${MODEL_ID}" \
      --mode img \
      --detect_prompt empty \
      --guidance_scale ${TR_GUIDANCE} \
      --steps ${TR_STEPS} \
      --inv_steps ${TR_INV_STEPS} \
      --fp16 ${TR_FP16} \
      --out_dir "${out_dir}" \
      --save_zt \
    > "${out_dir}/detect_tr_alt.runlog.txt" 2>&1 &

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

echo "[INFO] Stage 1: generate safe-off TR runs..." >&2
run_gen_job 0 "tr_w_safeoff" "${ZT_TR_W}" "${RUN_SAFEOFF_W}" 1
run_gen_job 1 "tr_w_att_safeoff" "${ZT_TR_WATT}" "${RUN_SAFEOFF_WATT}" 1
wait_all || exit 1

echo "[INFO] Stage 2: generate safe-on TR runs..." >&2
run_gen_job 0 "tr_w_safeon" "${ZT_TR_W}" "${RUN_SAFEON_W}" 0
run_gen_job 1 "tr_w_att_safeon" "${ZT_TR_WATT}" "${RUN_SAFEON_WATT}" 0
wait_all || exit 1

echo "[INFO] Stage 3: black-rate detection on safe-on runs..." >&2
run_black_job "${RUN_SAFEON_W}"
run_black_job "${RUN_SAFEON_WATT}"
wait_all || exit 1

echo "[INFO] Stage 4: TR watermark detection on safe-off runs..." >&2
run_tr_job 0 "${RUN_SAFEOFF_W}"
run_tr_job 1 "${RUN_SAFEOFF_WATT}"
wait_all || exit 1

echo "[DONE] Alt Dongman TR pipeline finished." >&2