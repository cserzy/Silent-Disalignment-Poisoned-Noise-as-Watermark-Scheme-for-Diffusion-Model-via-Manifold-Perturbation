#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# GS w_att -> aligned Gaussian OMS grid search on SD2.1
# Search over block_size x blend_alpha
# For each combo:
#   fit/apply OMS -> generate -> NSFW -> GS-OMS detect
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

MODEL_SD21=${ROOT}/checkpoints/sd2-1-diffusers

SRC_WATT=${LATDIR}/generate_GS_w_att.pt
TARGET_GAUSS=${LATDIR}/generate_GAUSS_w_aligned_vis.pt

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
INV_STEPS=50

NSFW_THRESHOLD=0.6
NSFW_SWEEP="0.2,0.3,0.4,0.5,0.6,0.7,0.8"

# ----------------------------------------
# Grid
# 你后面可以自己删减/扩展
# ----------------------------------------
BLOCK_SIZES=(32 64 128)
ALPHAS=(0.10 0.20 0.30)

run_one_combo() {
  local bs="$1"
  local alpha="$2"

  local alpha_tag
  alpha_tag=$(printf "%.2f" "${alpha}" | tr '.' 'p')

  local out_pt="${LATDIR}/generate_GS_w_att_oms_gauss_aligned_b${bs}_a${alpha_tag}.pt"
  local out_q="${LATDIR}/oms_Q_GS_w_att_to_gauss_aligned_b${bs}_a${alpha_tag}.pt"
  local out_json="${LATDIR}/oms_Q_GS_w_att_to_gauss_aligned_b${bs}_a${alpha_tag}.json"

  local run_dir="${IMGROOT}/vis_sd21_GS_w_att_oms_gauss_aligned_b${bs}_a${alpha_tag}_seed12345"
  local detect_dir="${run_dir}/detect_gs_oms"
  local nsfw_dir="${run_dir}/nsfw_report"

  echo
  echo "============================================================"
  echo "[GRID] block_size=${bs} alpha=${alpha}"
  echo "============================================================"

  # 1) OMS fit/apply
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

  # 2) generate
  python "${GEN_PY}" \
    --model_id "${MODEL_SD21}" \
    --prompts "${PROMPTS}" \
    --zT_pt "${out_pt}" \
    --outdir "${run_dir}" \
    --steps "${STEPS}" --cfg "${CFG}" --height "${HEIGHT}" --width "${WIDTH}" \
    --n_per_prompt "${N_PER_PROMPT}" --start_latent "${START_LATENT}" \
    --dtype "${GEN_DTYPE}" --seed "${Q_SEED}" \
    --negative_prompt "" || return 1

  # 3) NSFW
  python "${NSFW_PY}" \
    --manifests "${run_dir}/sliced/manifest.csv" \
    --out_dir "${nsfw_dir}" \
    --report_out "${nsfw_dir}/report.xlsx" \
    --threshold "${NSFW_THRESHOLD}" \
    --sweep "${NSFW_SWEEP}" || return 1

  # 4) detect
  mkdir -p "${detect_dir}"
  python "${DET_PY}" \
    --model_id "${MODEL_SD21}" \
    --run_dir "${run_dir}" \
    --out_dir "${detect_dir}" \
    --oms_q_pt "${out_q}" \
    --oms_meta_json "${out_json}" \
    --inv_steps "${INV_STEPS}" \
    --dtype "${GEN_DTYPE}" \
    --save_zt_oms \
    --save_zt_restored || return 1

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
  echo "[ALL DONE] GS grid search finished successfully."
else
  echo "[WARN] Some GS grid jobs failed."
fi
echo "============================================================"

exit "${FAIL}"