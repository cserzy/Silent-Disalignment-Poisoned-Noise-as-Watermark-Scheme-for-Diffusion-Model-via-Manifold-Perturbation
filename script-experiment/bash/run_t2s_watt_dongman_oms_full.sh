#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
LATDIR=${EXP_ROOT}/latents_experiment
LATDIR_NUM=${EXP_ROOT}/latents_experiment-number
IMGROOT=${EXP_ROOT}/imgs

PROMPTS=${ROOT}/prompts/cal_dongman_female_align-jinghua-2026-1_25.txt

OMS_PY=${EXP_ROOT}/script-experiment/oms_repair_pt.py
GEN_PY=${EXP_ROOT}/script-experiment/gen_from_zT_bank_multi_models-1_19.py
DET_PY=${EXP_ROOT}/script-experiment/detect/detect_T2S_oms.py
NSFW_PY=${ROOT}/nsfw_score_report_ring_wm_only_exposed_only-12.29.py
T2S_ROOT=${ROOT}/third_party/T2SMark

SD14=${ROOT}/checkpoints/sd1-4-diffusers
SD15=${ROOT}/checkpoints/sd1-5-diffusers
SD21=${ROOT}/checkpoints/sd2-1-diffusers

SRC=${LATDIR_NUM}/generate_T2S_w_att_dongman.pt
META_PT=${LATDIR_NUM}/generate_T2S_w_att_dongman_meta.pt
META_JSON=${LATDIR_NUM}/generate_T2S_w_att_dongman_meta.json
TARGET=${LATDIR}/generate_GAUSS_w_aligned_vis.pt

OUT_PT=${LATDIR_NUM}/generate_T2S_w_att_dongman_oms_gauss_aligned_b64_a0p20.pt
OUT_Q=${LATDIR_NUM}/oms_Q_T2S_w_att_dongman_to_gauss_aligned_b64_a0p20.pt
OUT_Q_JSON=${LATDIR_NUM}/oms_Q_T2S_w_att_dongman_to_gauss_aligned_b64_a0p20.json

Q_SEED=12345
BLOCK_SIZE=64
ALPHA=0.2
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

export PYTHONPATH=${T2S_ROOT}:${PYTHONPATH:-}

run_cmd() {
  echo
  echo "============================================================"
  echo "[RUN] $*"
  echo "============================================================"
  "$@" || exit 1
}

run_gen_nsfw_det() {
  local tag="$1"
  local model="$2"
  local gpu="$3"

  local run_dir=${IMGROOT}/vis_${tag}_T2S_w_att_dongman_oms_gauss_aligned_b64_a0p20_seed12345
  local nsfw_dir=${run_dir}/nsfw_report
  local det_dir=${run_dir}/detect_t2s_oms

  mkdir -p "${nsfw_dir}" "${det_dir}"

  echo
  echo "============================================================"
  echo "[MODEL] ${tag} | T2S dongman OMS"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${gpu}" run_cmd python "${GEN_PY}" \
    --model_id "${model}" \
    --prompts "${PROMPTS}" \
    --zT_pt "${OUT_PT}" \
    --outdir "${run_dir}" \
    --steps "${STEPS}" --cfg "${CFG}" --height "${HEIGHT}" --width "${WIDTH}" \
    --n_per_prompt "${N_PER_PROMPT}" --start_latent "${START_LATENT}" \
    --dtype "${GEN_DTYPE}" --seed "${Q_SEED}" \
    --negative_prompt ""

  run_cmd python "${NSFW_PY}" \
    --manifests "${run_dir}/sliced/manifest.csv" \
    --out_dir "${nsfw_dir}" \
    --report_out "${nsfw_dir}/report.xlsx" \
    --threshold "${NSFW_THRESHOLD}" \
    --sweep "${NSFW_SWEEP}"

  CUDA_VISIBLE_DEVICES="${gpu}" run_cmd python "${DET_PY}" \
    --model_id "${model}" \
    --run_dir "${run_dir}" \
    --out_dir "${det_dir}" \
    --cluster_meta_pt "${META_PT}" \
    --cluster_meta_json "${META_JSON}" \
    --oms_q_pt "${OUT_Q}" \
    --oms_meta_json "${OUT_Q_JSON}" \
    --t2s_root "${T2S_ROOT}" \
    --inv_steps "${INV_STEPS}" \
    --dtype "${GEN_DTYPE}" \
    --save_zt_oms \
    --save_zt_restored
}

echo
echo "============================================================"
echo "[STAGE 1] T2S dongman OMS fit/apply"
echo "============================================================"

run_cmd python "${OMS_PY}" \
  --mode fit_apply \
  --in_pt "${SRC}" \
  --target_pt "${TARGET}" \
  --out_pt "${OUT_PT}" \
  --out_q_pt "${OUT_Q}" \
  --out_meta_json "${OUT_Q_JSON}" \
  --q_seed "${Q_SEED}" \
  --block_size "${BLOCK_SIZE}" \
  --blend_alpha "${ALPHA}" \
  --match_target_std "${MATCH_TARGET_STD}" \
  --device cpu \
  --dtype "${FIT_DTYPE}" \
  --verbose

run_gen_nsfw_det sd14 "${SD14}" 0
run_gen_nsfw_det sd15 "${SD15}" 1
run_gen_nsfw_det sd21 "${SD21}" 1

echo
echo "============================================================"
echo "[DONE] T2S dongman OMS full pipeline finished."
echo "============================================================"