#!/usr/bin/env bash
set -euo pipefail

# ========= script & common paths =========
PY=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/generate_zT/generate_PRC_zT_w_att.py
SD14=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers
PROMPTS=/home/yancy/work/dm_backdoor_latent_space/prompts/cal_dongman_female_align-jinghua-2026-1_25.txt

OUTDIR=/home/yancy/work/dm_backdoor_latent_space/logs/tmp_prc_export_zT16_lamsweep
EXPORT_DIR=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment
WM_META_SUBDIR=wm_meta_prc

LOGROOT=/home/yancy/work/dm_backdoor_latent_space/logs/lam_sweep_PRC_queue
mkdir -p "$OUTDIR" "$EXPORT_DIR" "$LOGROOT"

# ========= fixed args (对齐你已跑通那条命令) =========
COMMON_ARGS=(
  --model_id "$SD14"
  --prompts "$PROMPTS"
  --outdir "$OUTDIR"
  --negative_prompt ""
  --steps 50 --cfg 7.5 --height 512 --width 512
  --rows 4 --cols 4 --gen_bs 4
  --seed 12345
  --ssc_N_cal 12 --ssc_energy_ratio 0.900 --ssc_mini_steps 6
  --ssc_d_sens_max 256 --ssc_d_wm 256
  --reuse_ssc 1
  --prc_message_length 8 --prc_error_prob 0.01
  --master_key "prc_key_yx_0504"
  --save_zT 0
  --export_zT16_only 1
  --export_latents_dir "$EXPORT_DIR"
  --wm_meta_subdir "$WM_META_SUBDIR"
)

lam_tag() { echo "$1" | sed 's/\./_/g'; }

run_one() {
  local gpu="$1"
  local lam="$2"
  local tag; tag="$(lam_tag "$lam")"
  local outname="generate_PRC_w_att_${tag}.pt"

  echo "[LAUNCH][GPU${gpu}] lam1=${lam} -> ${EXPORT_DIR}/${outname}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "$PY" \
    --lam1 "$lam" \
    --export_latents_name "$outname" \
    "${COMMON_ARGS[@]}"
  echo "[OK][GPU${gpu}] lam1=${lam} done"
}

echo "[INFO] GPU0 queue: 0.84 -> 0.87"
echo "[INFO] GPU1 queue: 0.86"
nvidia-smi || true

# GPU0：串行 0.84 -> 0.87（整体后台并 tee）
(
  run_one 0 0.88
) 2>&1 | tee "$LOGROOT/prc_gpu0_0.88.log" &

PID0=$!

# GPU1：串行 0.86（整体后台并 tee）
(
  run_one 1 0.89
) 2>&1 | tee "$LOGROOT/prc_gpu1_0.89.log" &

PID1=$!

echo "[INFO] Launched: GPU0 PID=$PID0, GPU1 PID=$PID1"
wait "$PID0" "$PID1"

echo "[DONE] All PRC export_zT16 jobs finished."
echo "[DONE] Exported: $EXPORT_DIR/generate_PRC_w_att_0_88.pt  $EXPORT_DIR/generate_PRC_w_att_0_89.pt  $EXPORT_DIR/generate_PRC_w_att_0_87.pt"
echo "[DONE] Logs: $LOGROOT/*.log"
nvidia-smi || true
