#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
cd "${EXP_ROOT}" || exit 1

GEN=script-experiment/gen_from_zT_bank_versatile_diffusion-1_19.py
PROMPTS=${ROOT}/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt
MODEL_ID=/home/yancy/work/dm_backdoor_latent_space/checkpoints/versatile-diffusion-pr7

ZT_GAUSS=${EXP_ROOT}/latents_experiment/generate_GAUSS_w_aligned_vis.pt
ZT_GSW=${EXP_ROOT}/latents_experiment/generate_GS_w.pt
ZT_GSWATT=${EXP_ROOT}/latents_experiment/generate_GS_w_att.pt

SEED=12345
STEPS=50
CFG=7.5
HEIGHT=512
WIDTH=512
N_PER_PROMPT=4
START_LATENT=0
DTYPE=fp16

pids=()

run_job () {
  local gpu="$1"
  local tag="$2"
  local zt_pt="$3"
  local outdir="imgs/vis_vd_${tag}_seed${SEED}"

  mkdir -p "${outdir}"

  echo "[LAUNCH][GPU${gpu}] python ${GEN} --model_id ${MODEL_ID} --prompts ${PROMPTS} --zT_pt ${zt_pt} --outdir ${outdir}" >&2

  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "${GEN}" \
      --model_id "${MODEL_ID}" \
      --prompts "${PROMPTS}" \
      --zT_pt "${zt_pt}" \
      --outdir "${outdir}" \
      --steps ${STEPS} --cfg ${CFG} --height ${HEIGHT} --width ${WIDTH} \
      --n_per_prompt ${N_PER_PROMPT} --start_latent ${START_LATENT} \
      --dtype ${DTYPE} --seed ${SEED} \
    > "${outdir}/gen.runlog.txt" 2>&1 &

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

echo "[INFO] Stage 1: launch GAUSS + GS_w in parallel..." >&2
run_job 0 "gauss" "${ZT_GAUSS}"
run_job 1 "gs_w" "${ZT_GSW}"
wait_all || exit 1

echo "[INFO] Stage 2: launch GS_w_att ..." >&2
run_job 0 "gs_w_att" "${ZT_GSWATT}"
wait_all || exit 1

echo "[DONE] Versatile Diffusion generation finished." >&2
echo "[DONE] Outputs:" >&2
echo "  imgs/vis_vd_gauss_seed${SEED}" >&2
echo "  imgs/vis_vd_gs_w_seed${SEED}" >&2
echo "  imgs/vis_vd_gs_w_att_seed${SEED}" >&2