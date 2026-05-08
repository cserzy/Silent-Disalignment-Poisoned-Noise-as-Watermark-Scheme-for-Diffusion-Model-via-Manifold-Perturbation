#!/usr/bin/env bash
set -euo pipefail

PY=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/gen_from_zT_bank_multi_models-1_19.py
PROMPTS=/home/yancy/work/dm_backdoor_latent_space/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt
ZDIR=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment
IMGROOT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs

SD14=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers
SD15=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers
SD21=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers

COMMON_ARGS=(
  --prompts "$PROMPTS"
  --steps 50 --cfg 7.5 --height 512 --width 512
  --n_per_prompt 4 --start_latent 0
  --dtype fp16 --seed 12345
  --negative_prompt ""
)

mkdir -p "$IMGROOT"

echo "[INFO] Launch 6 jobs (GPU0: 4 jobs, GPU1: 2 jobs) ..."

# -------------------------
# GPU0: SD1.4 w / w_att
# -------------------------
CUDA_VISIBLE_DEVICES=0 python "$PY" \
  --model_id "$SD14" \
  --zT_pt "$ZDIR/generate_TR_w.pt" \
  --outdir "$IMGROOT/vis_sd14_TR_w_seed12345" \
  "${COMMON_ARGS[@]}" \
  > "$IMGROOT/vis_sd14_TR_w_seed12345.runlog.txt" 2>&1 &

CUDA_VISIBLE_DEVICES=0 python "$PY" \
  --model_id "$SD14" \
  --zT_pt "$ZDIR/generate_TR_w_att.pt" \
  --outdir "$IMGROOT/vis_sd14_TR_w_att_seed12345" \
  "${COMMON_ARGS[@]}" \
  > "$IMGROOT/vis_sd14_TR_w_att_seed12345.runlog.txt" 2>&1 &

# -------------------------
# GPU0: SD1.5 w / w_att
# -------------------------
CUDA_VISIBLE_DEVICES=0 python "$PY" \
  --model_id "$SD15" \
  --zT_pt "$ZDIR/generate_TR_w.pt" \
  --outdir "$IMGROOT/vis_sd15_TR_w_seed12345" \
  "${COMMON_ARGS[@]}" \
  > "$IMGROOT/vis_sd15_TR_w_seed12345.runlog.txt" 2>&1 &

CUDA_VISIBLE_DEVICES=0 python "$PY" \
  --model_id "$SD15" \
  --zT_pt "$ZDIR/generate_TR_w_att.pt" \
  --outdir "$IMGROOT/vis_sd15_TR_w_att_seed12345" \
  "${COMMON_ARGS[@]}" \
  > "$IMGROOT/vis_sd15_TR_w_att_seed12345.runlog.txt" 2>&1 &

# -------------------------
# GPU1: SD2.1 w / w_att
# -------------------------
CUDA_VISIBLE_DEVICES=1 python "$PY" \
  --model_id "$SD21" \
  --zT_pt "$ZDIR/generate_TR_w.pt" \
  --outdir "$IMGROOT/vis_sd21_TR_w_seed12345" \
  "${COMMON_ARGS[@]}" \
  > "$IMGROOT/vis_sd21_TR_w_seed12345.runlog.txt" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python "$PY" \
  --model_id "$SD21" \
  --zT_pt "$ZDIR/generate_TR_w_att.pt" \
  --outdir "$IMGROOT/vis_sd21_TR_w_att_seed12345" \
  "${COMMON_ARGS[@]}" \
  > "$IMGROOT/vis_sd21_TR_w_att_seed12345.runlog.txt" 2>&1 &

# wait for all background jobs
wait

echo "[DONE] All jobs finished."
echo "[DONE] Logs are under: $IMGROOT/*.runlog.txt"
