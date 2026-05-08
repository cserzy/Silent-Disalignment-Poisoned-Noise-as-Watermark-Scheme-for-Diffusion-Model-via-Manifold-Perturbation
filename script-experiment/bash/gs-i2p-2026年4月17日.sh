#!/usr/bin/env bash
set -u
set -o pipefail

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19

GEN=${EXP_ROOT}/script-experiment/gen_from_zT_bank_multi_models-1_19.py
PROMPTS=${ROOT}/prompts/i2p_benchmark_sexual-50.txt
IMGROOT=${EXP_ROOT}/imgs_i2psexual50

SD14=${ROOT}/checkpoints/sd1-4-diffusers
SD15=${ROOT}/checkpoints/sd1-5-diffusers
SD21=${ROOT}/checkpoints/sd2-1-diffusers

ZT_GAUSS_NEW=${EXP_ROOT}/latents_experiment/generate_GAUSS_w_aligned_vis.pt

SEED=12345
STEPS=50
CFG=7.5
HEIGHT=512
WIDTH=512
N_PER_PROMPT=4
START_LATENT=0
DTYPE_GEN=fp16

mkdir -p "${IMGROOT}"

pids=()

run_gen_job () {
  local gpu="$1"
  local model_id="$2"
  local tag="$3"

  local run_dir="${IMGROOT}/vis_i2psexual50_${tag}_gauss_seed${SEED}"
  local log="${run_dir}/gen.runlog.txt"

  mkdir -p "${run_dir}"

  echo "[LAUNCH][GEN] GPU${gpu} ${tag} gauss" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "${GEN}" \
      --model_id "${model_id}" \
      --prompts "${PROMPTS}" \
      --zT_pt "${ZT_GAUSS_NEW}" \
      --outdir "${run_dir}" \
      --steps ${STEPS} --cfg ${CFG} --height ${HEIGHT} --width ${WIDTH} \
      --n_per_prompt ${N_PER_PROMPT} --start_latent ${START_LATENT} \
      --dtype ${DTYPE_GEN} --seed ${SEED} \
    > "${log}" 2>&1 &

  pids+=("$!")
}

wait_all_jobs () {
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

echo "[INFO] Regenerating baseline Gaussian runs with new pt..." >&2

# batch 1: sd14 + sd15
run_gen_job 0 "${SD14}" "sd14"
run_gen_job 1 "${SD15}" "sd15"
wait_all_jobs || exit 1

# batch 2: sd21
run_gen_job 0 "${SD21}" "sd21"
wait_all_jobs || exit 1

echo "[DONE] Baseline Gaussian regeneration finished." >&2
echo "[DONE] New pt used: ${ZT_GAUSS_NEW}" >&2