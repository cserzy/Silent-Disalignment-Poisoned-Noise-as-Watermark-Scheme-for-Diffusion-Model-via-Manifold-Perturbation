#!/usr/bin/env bash
set -euo pipefail

PY=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/gen_from_zT_bank_multi_models-1_19.py
PROMPTS=/home/yancy/work/dm_backdoor_latent_space/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt
LATDIR=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment
IMGROOT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs

SD14=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers
SD15=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers
SD21=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers

ZT_W=${LATDIR}/generate_T2S_w.pt
ZT_WATT=${LATDIR}/generate_T2S_w_att.pt

COMMON_ARGS=(
  --prompts "$PROMPTS"
  --steps 50 --cfg 7.5 --height 512 --width 512
  --n_per_prompt 4 --start_latent 0
  --dtype fp16 --seed 12345
  --negative_prompt ""
)

mkdir -p "$IMGROOT"

run_job () {
  local gpu="$1"
  local model_id="$2"
  local tag="$3"       # sd14 / sd15 / sd21
  local variant="$4"   # w / w_att
  local zt_pt="$5"

  local outdir="${IMGROOT}/vis_${tag}_T2S_${variant}_clip_seed12345"
  local log="${outdir}.runlog.txt"

  echo "[LAUNCH] GPU${gpu}  ${tag}  T2S_${variant}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "$PY" \
      --model_id "$model_id" \
      --zT_pt "$zt_pt" \
      --outdir "$outdir" \
      "${COMMON_ARGS[@]}" \
    > "$log" 2>&1 &

  echo $!  # print pid
}

echo "[INFO] Launching 6 jobs at once (3 per GPU)..."

# GPU0: all "w"

#PID1=$(run_job 0 "$SD14" "sd14" "w"     "$ZT_W")
#PID2=$(run_job 0 "$SD15" "sd15" "w"     "$ZT_W")
#PID3=$(run_job 0 "$SD21" "sd21" "w"     "$ZT_W")


# GPU1: all "w_att"
PID4=$(run_job 0 "$SD14" "sd14" "w_att" "$ZT_WATT")
PID5=$(run_job 1 "$SD15" "sd15" "w_att" "$ZT_WATT")
PID6=$(run_job 1 "$SD21" "sd21" "w_att" "$ZT_WATT")


echo "[DONE] All 6 jobs finished OK."
echo "[DONE] Logs: ${IMGROOT}/vis_*T2S*_seed12345.runlog.txt"
