#!/usr/bin/env bash
set -euo pipefail

PRC_PY=/home/yancy/work/dm_backdoor_latent_space/prc_detect_global_official_align-1_18_fixdim-meg.py
IMGROOT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs

SD14=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers
SD15=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers
SD21=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers

# fixed args (keep exactly as your template)
FIXED_ARGS=(--steps 50 --guidance 7.5 --dtype fp32 --var 1.5 --fpr 1e-2 --inv_bs 1 --debug)

pids=()

run_job () {
  local gpu="$1"
  local model_id="$2"
  local run_dir="$3"

  local log="${run_dir}/prc_detect.runlog.txt"
  mkdir -p "$run_dir"

  echo "[LAUNCH] GPU${gpu}  run_dir=$(basename "$run_dir")" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "$PRC_PY" \
      --run_dir  "$run_dir" \
      --model_id "$model_id" \
      "${FIXED_ARGS[@]}" \
    > "$log" 2>&1 &

  # IMPORTANT: collect PID in current shell (no command substitution)
  pids+=("$!")
}

echo "[INFO] Launching 6 PRC detect jobs at once (3 per GPU)..." >&2

# -------------------------
# GPU0: SD14(w) + SD15(w) + SD21(w)
# -------------------------
#run_job 0 "$SD14" "${IMGROOT}/vis_sd14_PRC_w_0_85_seed12345-copy"
#run_job 0 "$SD15" "${IMGROOT}/vis_sd15_PRC_w_0_85_seed12345"
#run_job 0 "$SD21" "${IMGROOT}/vis_sd21_PRC_w_0_84_seed12345"

# -------------------------
# GPU1: SD14(w_att) + SD15(w_att) + SD21(w_att)
# -------------------------
run_job 0 "$SD14" "${IMGROOT}/vis_sd14_PRC_w_att_0_85_clip_new_method_seed12345"
run_job 1 "$SD15" "${IMGROOT}/vis_sd15_PRC_w_att_0_85_clip_new_method_seed12345"
run_job 1 "$SD21" "${IMGROOT}/vis_sd21_PRC_w_att_0_85_clip_new_method_seed12345"

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "[ERROR] job failed: pid=$pid" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[DONE] Some PRC detect jobs failed. Check logs:" >&2
  echo "       ${IMGROOT}/vis_sd*_PRC_*_0_85_seed12345/prc_detect.runlog.txt" >&2
  exit 1
fi

echo "[DONE] All 6 PRC detect jobs finished OK." >&2
echo "[DONE] Logs: ${IMGROOT}/vis_sd*_PRC_*_0_85_seed12345/prc_detect.runlog.txt" >&2
