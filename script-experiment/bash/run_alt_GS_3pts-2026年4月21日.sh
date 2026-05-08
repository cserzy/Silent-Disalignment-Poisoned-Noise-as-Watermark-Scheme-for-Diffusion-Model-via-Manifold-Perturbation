#!/usr/bin/env bash
set -uo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate Hijacking

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
cd "${EXP_ROOT}" || exit 1

GEN=${EXP_ROOT}/script-experiment/gen_from_zT_bank_alt_diffusion-1_19.py
BLACK_DET=${EXP_ROOT}/script-experiment/detect/detect_black_ratio_alt.py
WM_DET=${EXP_ROOT}/script-experiment/detect/detect_GS_alt.py

MODEL_ID=${ROOT}/checkpoints/AltDiffusion-fp16
PROMPTS=${ROOT}/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt

ZT_GAUSS=${EXP_ROOT}/latents_experiment/generate_GAUSS_w_aligned_vis.pt
ZT_GSW=${EXP_ROOT}/latents_experiment/generate_GS_w.pt
ZT_GSWATT=${EXP_ROOT}/latents_experiment/generate_GS_w_att.pt

SEED=12345
STEPS=50
CFG=7.5
HEIGHT=512
WIDTH=512
DTYPE=fp16
N_PER_PROMPT=4
START_LATENT=0
INV_STEPS=50

# -----------------------------
# 开安全审查：重新生成
# -----------------------------
RUN_SAFEON_GAUSS=${EXP_ROOT}/imgs/vis_alt_safeon_gauss_seed${SEED}
RUN_SAFEON_GSW=${EXP_ROOT}/imgs/vis_alt_safeon_gs_w_seed${SEED}
RUN_SAFEON_GSWATT=${EXP_ROOT}/imgs/vis_alt_safeon_gs_w_att_seed${SEED}

# 已生成好的关安全审查目录：用于水印检测
RUN_SAFEOFF_GAUSS=${EXP_ROOT}/imgs/vis_alt_gauss_seed${SEED}
RUN_SAFEOFF_GSW=${EXP_ROOT}/imgs/vis_alt_gs_w_seed${SEED}
RUN_SAFEOFF_GSWATT=${EXP_ROOT}/imgs/vis_alt_gs_w_att_seed${SEED}

pids=()

run_gen_job () {
  local gpu="$1"
  local tag="$2"
  local zt_pt="$3"
  local outdir="$4"

  mkdir -p "${outdir}"

  echo "[LAUNCH][GEN][GPU${gpu}] ${tag} -> ${outdir}" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "${GEN}" \
      --model_id "${MODEL_ID}" \
      --prompts "${PROMPTS}" \
      --zT_pt "${zt_pt}" \
      --outdir "${outdir}" \
      --steps ${STEPS} \
      --cfg ${CFG} \
      --height ${HEIGHT} \
      --width ${WIDTH} \
      --device cuda \
      --dtype ${DTYPE} \
      --n_per_prompt ${N_PER_PROMPT} \
      --start_latent ${START_LATENT} \
      --seed ${SEED} \
    > "${outdir}/gen.runlog.txt" 2>&1 &

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

run_wm_job () {
  local gpu="$1"
  local run_dir="$2"
  local out_dir="${run_dir}/detect_alt"

  mkdir -p "${out_dir}"

  echo "[LAUNCH][WM][GPU${gpu}] ${run_dir}" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "${WM_DET}" \
      --model_id "${MODEL_ID}" \
      --run_dir "${run_dir}" \
      --out_dir "${out_dir}" \
      --dtype ${DTYPE} \
      --inv_steps ${INV_STEPS} \
      --save_zt \
    > "${out_dir}/detect_alt.runlog.txt" 2>&1 &

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

echo "[INFO] Stage 1: Alt safe-on generation..." >&2
run_gen_job 0 "gauss"   "${ZT_GAUSS}"  "${RUN_SAFEON_GAUSS}"
run_gen_job 1 "gs_w"    "${ZT_GSW}"    "${RUN_SAFEON_GSW}"
wait_all || exit 1

run_gen_job 0 "gs_w_att" "${ZT_GSWATT}" "${RUN_SAFEON_GSWATT}"
wait_all || exit 1

echo "[INFO] Stage 2: black-ratio detection on safe-on runs..." >&2
run_black_job "${RUN_SAFEON_GAUSS}"
run_black_job "${RUN_SAFEON_GSW}"
run_black_job "${RUN_SAFEON_GSWATT}"
wait_all || exit 1

echo "[INFO] Stage 3: watermark detection on existing safe-off runs..." >&2
run_wm_job 0 "${RUN_SAFEOFF_GAUSS}"
run_wm_job 1 "${RUN_SAFEOFF_GSW}"
wait_all || exit 1

run_wm_job 0 "${RUN_SAFEOFF_GSWATT}"
wait_all || exit 1

echo "[DONE] Alt safe-on generation + black-rate detection + safe-off watermark detection finished." >&2
echo "[DONE] Safe-on outputs:" >&2
echo "  ${RUN_SAFEON_GAUSS}" >&2
echo "  ${RUN_SAFEON_GSW}" >&2
echo "  ${RUN_SAFEON_GSWATT}" >&2
echo "[DONE] Safe-off detect outputs:" >&2
echo "  ${RUN_SAFEOFF_GAUSS}/detect_alt" >&2
echo "  ${RUN_SAFEOFF_GSW}/detect_alt" >&2
echo "  ${RUN_SAFEOFF_GSWATT}/detect_alt" >&2