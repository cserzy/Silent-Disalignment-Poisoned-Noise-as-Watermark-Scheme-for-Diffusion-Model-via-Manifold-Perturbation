#!/usr/bin/env bash
set -euo pipefail

PY="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/generate_zT/generate_TR_zT_w_att.py"
MODEL="/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers"
PROMPTS="/home/yancy/work/dm_backdoor_latent_space/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt"
OUTDIR="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment"

# fixed args (strictly aligned with your template)
H=512
W=512
SEED=12345
SSC_N_CAL=12
SSC_MINI=6
TR_W_SEED=12345
TR_W_PATTERN=ring
TR_W_MASK_SHAPE=circle
TR_W_RADIUS=9
TR_W_CHANNEL=-1
TR_W_INJECTION=complex

LOGDIR="${OUTDIR}/_logs_lam1_sweep"
mkdir -p "$OUTDIR" "$LOGDIR"

run_one () {
  local gpu="$1"
  local lam="$2"

  local lam_tag="${lam//./_}"  # 0.86 -> 0_86
  local out_pt="${OUTDIR}/generate_TR_w_att_${lam_tag}.pt"

  echo "[RUN] GPU${gpu} lam1=${lam} -> $(basename "$out_pt")" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" python "$PY" \
    --model_id "$MODEL" \
    --prompts  "$PROMPTS" \
    --outdir   "$OUTDIR" \
    --out_pt   "$out_pt" \
    --height "$H" --width "$W" \
    --lam1 "$lam" --seed "$SEED" \
    --ssc_N_cal "$SSC_N_CAL" --ssc_mini_steps "$SSC_MINI" \
    --tr_w_seed "$TR_W_SEED" \
    --tr_w_pattern "$TR_W_PATTERN" \
    --tr_w_mask_shape "$TR_W_MASK_SHAPE" \
    --tr_w_radius "$TR_W_RADIUS" \
    --tr_w_channel "$TR_W_CHANNEL" \
    --tr_w_injection "$TR_W_INJECTION"
}

echo "[INFO] GPU0 queue: lam1 0.86 -> 0.89"
echo "[INFO] GPU1 queue: lam1 0.87 -> 0.90"
nvidia-smi || true

# GPU0: two jobs sequentially (whole queue runs in background)
(
  run_one 0 0.86
  run_one 0 0.89
) 2>&1 | tee "${LOGDIR}/tr_w_att_lam1_gpu0.log" &

PID0=$!

# GPU1: two jobs sequentially (whole queue runs in background)
(
  run_one 1 0.87
  run_one 1 0.90
) 2>&1 | tee "${LOGDIR}/tr_w_att_lam1_gpu1.log" &

PID1=$!

echo "[INFO] Launched queues: GPU0 PID=$PID0, GPU1 PID=$PID1"
wait "$PID0" "$PID1"

echo "[DONE] TR w_att lam1 sweep finished."
echo "[DONE] PTs: ${OUTDIR}/generate_TR_w_att_0_*.pt"
echo "[DONE] Logs: ${LOGDIR}/tr_w_att_lam1_gpu*.log"
nvidia-smi || true
