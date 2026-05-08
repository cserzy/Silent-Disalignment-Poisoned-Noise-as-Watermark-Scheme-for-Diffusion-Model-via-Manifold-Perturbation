#!/usr/bin/env bash
set -u
set -o pipefail

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
PROMPTS=${ROOT}/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt

MODEL_SD14=${ROOT}/checkpoints/sd1-4-diffusers
MODEL_SD15=${ROOT}/checkpoints/sd1-5-diffusers
MODEL_SD21=${ROOT}/checkpoints/sd2-1-diffusers

OMS_Q_PT=${EXP_ROOT}/latents_experiment/oms_Q_GS.pt
OMS_META_JSON=${EXP_ROOT}/latents_experiment/oms_Q_GS.json

RUN_SD21=${EXP_ROOT}/imgs/vis_sd21_GS_w_att_oms_seed12345
RUN_SD14=${EXP_ROOT}/imgs/vis_sd14_GS_w_att_oms_seed12345
RUN_SD15=${EXP_ROOT}/imgs/vis_sd15_GS_w_att_oms_seed12345

DETECT_SCRIPT=${EXP_ROOT}/script-experiment/detect/detect_GS_oms.py
NSFW_SCRIPT=${ROOT}/nsfw_score_report_ring_wm_only_exposed_only-12.29.py

mkdir -p "${RUN_SD21}/detect_gs_oms" "${RUN_SD14}/detect_gs_oms" "${RUN_SD15}/detect_gs_oms"
mkdir -p "${RUN_SD14}/nsfw_report" "${RUN_SD15}/nsfw_report"

FAIL=0

run_cmd () {
  echo
  echo "============================================================"
  echo "[RUN] $*"
  echo "============================================================"
  "$@" || FAIL=1
}

# -----------------------------
# 1) SD21: OMS GS watermark detection
# -----------------------------
run_cmd python "${DETECT_SCRIPT}" \
  --model_id "${MODEL_SD21}" \
  --run_dir "${RUN_SD21}" \
  --out_dir "${RUN_SD21}/detect_gs_oms" \
  --oms_q_pt "${OMS_Q_PT}" \
  --oms_meta_json "${OMS_META_JSON}" \
  --inv_steps 50 \
  --dtype fp16 \
  --save_zt_oms \
  --save_zt_restored

# -----------------------------
# 2) SD14: NSFW
# -----------------------------
run_cmd python "${NSFW_SCRIPT}" \
  --manifests "${RUN_SD14}/sliced/manifest.csv" \
  --out_dir "${RUN_SD14}/nsfw_report" \
  --report_out "${RUN_SD14}/nsfw_report/report.xlsx" \
  --threshold 0.6 \
  --sweep 0.2,0.3,0.4,0.5,0.6,0.7,0.8

# -----------------------------
# 3) SD14: OMS GS watermark detection
# -----------------------------
run_cmd python "${DETECT_SCRIPT}" \
  --model_id "${MODEL_SD14}" \
  --run_dir "${RUN_SD14}" \
  --out_dir "${RUN_SD14}/detect_gs_oms" \
  --oms_q_pt "${OMS_Q_PT}" \
  --oms_meta_json "${OMS_META_JSON}" \
  --inv_steps 50 \
  --dtype fp16 \
  --save_zt_oms \
  --save_zt_restored

# -----------------------------
# 4) SD15: NSFW
# -----------------------------
run_cmd python "${NSFW_SCRIPT}" \
  --manifests "${RUN_SD15}/sliced/manifest.csv" \
  --out_dir "${RUN_SD15}/nsfw_report" \
  --report_out "${RUN_SD15}/nsfw_report/report.xlsx" \
  --threshold 0.6 \
  --sweep 0.2,0.3,0.4,0.5,0.6,0.7,0.8

# -----------------------------
# 5) SD15: OMS GS watermark detection
# -----------------------------
run_cmd python "${DETECT_SCRIPT}" \
  --model_id "${MODEL_SD15}" \
  --run_dir "${RUN_SD15}" \
  --out_dir "${RUN_SD15}/detect_gs_oms" \
  --oms_q_pt "${OMS_Q_PT}" \
  --oms_meta_json "${OMS_META_JSON}" \
  --inv_steps 50 \
  --dtype fp16 \
  --save_zt_oms \
  --save_zt_restored

echo
echo "============================================================"
if [[ "${FAIL}" -eq 0 ]]; then
  echo "[DONE] All jobs finished successfully."
else
  echo "[WARN] Some jobs failed. Please inspect logs above."
fi
echo "============================================================"
exit "${FAIL}"