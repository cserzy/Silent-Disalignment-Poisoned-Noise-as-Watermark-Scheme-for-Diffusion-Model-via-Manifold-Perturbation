#!/usr/bin/env bash
set -euo pipefail

# Ctrl+C / 退出时把后台任务一起干掉
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT INT TERM

PY=python
DET=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/detect/detect_TR.py

SD14=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers
SD15=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers
SD21=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers

BASE=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs

# ----------------------------
# GPU0: delssc (sd14/sd15/sd21)
# ----------------------------
CUDA_VISIBLE_DEVICES=0 $PY $DET \
  --img_dir  $BASE/vis_sd14_TR_w_att_0_88_delssc_seed12345/sliced \
  --model_id $SD14 \
  --out_xlsx $BASE/vis_sd14_TR_w_att_0_88_delssc_seed12345/sd14_TR_w_att_0_88_delssc_detect.xlsx \
  > $BASE/vis_sd14_TR_w_att_0_88_delssc_seed12345/detect_TR.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=0 $PY $DET \
  --img_dir  $BASE/vis_sd15_TR_w_att_0_88_delssc_seed12345/sliced \
  --model_id $SD15 \
  --out_xlsx $BASE/vis_sd15_TR_w_att_0_88_delssc_seed12345/sd15_TR_w_att_0_88_delssc_detect.xlsx \
  > $BASE/vis_sd15_TR_w_att_0_88_delssc_seed12345/detect_TR.log 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=0 $PY $DET \
  --img_dir  $BASE/vis_sd21_TR_w_att_0_88_delssc_seed12345/sliced \
  --model_id $SD21 \
  --out_xlsx $BASE/vis_sd21_TR_w_att_0_88_delssc_seed12345/sd21_TR_w_att_0_88_delssc_detect.xlsx \
  > $BASE/vis_sd21_TR_w_att_0_88_delssc_seed12345/detect_TR.log 2>&1 &
PID3=$!

# ----------------------------
# GPU1: delrepair (sd14/sd15/sd21)
# ----------------------------
CUDA_VISIBLE_DEVICES=1 $PY $DET \
  --img_dir  $BASE/vis_sd14_TR_w_att_0_88_delrepair_seed12345/sliced \
  --model_id $SD14 \
  --out_xlsx $BASE/vis_sd14_TR_w_att_0_88_delrepair_seed12345/sd14_TR_w_att_0_88_delrepair_detect.xlsx \
  > $BASE/vis_sd14_TR_w_att_0_88_delrepair_seed12345/detect_TR.log 2>&1 &
PID4=$!

CUDA_VISIBLE_DEVICES=1 $PY $DET \
  --img_dir  $BASE/vis_sd15_TR_w_att_0_88_delrepair_seed12345/sliced \
  --model_id $SD15 \
  --out_xlsx $BASE/vis_sd15_TR_w_att_0_88_delrepair_seed12345/sd15_TR_w_att_0_88_delrepair_detect.xlsx \
  > $BASE/vis_sd15_TR_w_att_0_88_delrepair_seed12345/detect_TR.log 2>&1 &
PID5=$!

CUDA_VISIBLE_DEVICES=1 $PY $DET \
  --img_dir  $BASE/vis_sd21_TR_w_att_0_88_delrepair_seed12345/sliced \
  --model_id $SD21 \
  --out_xlsx $BASE/vis_sd21_TR_w_att_0_88_delrepair_seed12345/sd21_TR_w_att_0_88_delrepair_detect.xlsx \
  > $BASE/vis_sd21_TR_w_att_0_88_delrepair_seed12345/detect_TR.log 2>&1 &
PID6=$!

echo "[LAUNCHED] $PID1 $PID2 $PID3 $PID4 $PID5 $PID6"
echo "[LOGS]"
echo "  $BASE/vis_sd14_TR_w_att_0_88_delssc_seed12345/detect_TR.log"
echo "  $BASE/vis_sd15_TR_w_att_0_88_delssc_seed12345/detect_TR.log"
echo "  $BASE/vis_sd21_TR_w_att_0_88_delssc_seed12345/detect_TR.log"
echo "  $BASE/vis_sd14_TR_w_att_0_88_delrepair_seed12345/detect_TR.log"
echo "  $BASE/vis_sd15_TR_w_att_0_88_delrepair_seed12345/detect_TR.log"
echo "  $BASE/vis_sd21_TR_w_att_0_88_delrepair_seed12345/detect_TR.log"

# 等待并汇总退出码：只要有一个失败，脚本整体失败
FAIL=0
for P in $PID1 $PID2 $PID3 $PID4 $PID5 $PID6; do
  if ! wait "$P"; then
    echo "[ERR] job failed: PID=$P"
    FAIL=1
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "[DONE] some jobs failed."
  exit 1
fi

echo "[DONE] all 6 jobs finished successfully."
