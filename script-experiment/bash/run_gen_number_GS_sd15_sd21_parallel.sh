#!/usr/bin/env bash
set -euo pipefail

PY="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/gen_from_zT_bank_multi_models-1_19.py"
PROMPTS="/home/yancy/work/dm_backdoor_latent_space/prompts/cal_number_align-2026_1_11.txt"

# zT banks (按你给的示例原样复用)
ZT_W_ATT="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment-number/generate_GS_w_att_number.pt"
ZT_W="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_GS_w.pt"

# models
SD15="/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers"
SD21="/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers"

# common gen args
STEPS=50
CFG=7.5
H=512
W=512
N_PER=4
START_LATENT=0
DTYPE="fp16"
SEED=12345
NEG=""

# out dirs
OUT_SD15_WATT="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_GS_w_att_number_seed12345"
OUT_SD15_W="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_GS_w_number_seed12345"

OUT_SD21_WATT="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_GS_w_att_number_seed12345"
OUT_SD21_W="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_GS_w_number_seed12345"

LOGDIR="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/_logs"
mkdir -p "$LOGDIR"

run_one () {
  local gpu="$1"
  local model_id="$2"
  local zt_pt="$3"
  local outdir="$4"

  CUDA_VISIBLE_DEVICES="${gpu}" python "$PY" \
    --model_id "$model_id" \
    --prompts "$PROMPTS" \
    --zT_pt "$zt_pt" \
    --outdir "$outdir" \
    --steps "$STEPS" --cfg "$CFG" --height "$H" --width "$W" \
    --n_per_prompt "$N_PER" --start_latent "$START_LATENT" \
    --dtype "$DTYPE" --seed "$SEED" \
    --negative_prompt "$NEG"
}

echo "[INFO] GPU0 -> SD1.5 (GS w_att_number -> w_number)"
echo "[INFO] GPU1 -> SD2.1 (GS w_att_number -> w_number)"
nvidia-smi || true

# GPU0: sd1.5 两条顺序跑（同卡不并行，省显存/省事）
(
  run_one 0 "$SD15" "$ZT_W_ATT" "$OUT_SD15_WATT"
  run_one 0 "$SD15" "$ZT_W"     "$OUT_SD15_W"
) 2>&1 | tee "$LOGDIR/gs_number_sd15_gpu0.log" &

PID0=$!

# GPU1: sd2.1 两条顺序跑
(
  run_one 1 "$SD21" "$ZT_W_ATT" "$OUT_SD21_WATT"
  run_one 1 "$SD21" "$ZT_W"     "$OUT_SD21_W"
) 2>&1 | tee "$LOGDIR/gs_number_sd21_gpu1.log" &

PID1=$!

echo "[INFO] Launched: sd15 PID=$PID0, sd21 PID=$PID1"
wait "$PID0" "$PID1"

echo "[DONE] GS number-align generation for sd15+sd21 finished."
nvidia-smi || true
