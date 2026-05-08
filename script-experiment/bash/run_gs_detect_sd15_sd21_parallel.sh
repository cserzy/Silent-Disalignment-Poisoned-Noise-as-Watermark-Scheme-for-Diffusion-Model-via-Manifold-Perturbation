#!/usr/bin/env bash
set -uo pipefail

DET=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/detect/detect_GS.py
IMGROOT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs

SD14=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers
SD15=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers
SD21=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers

pids=()

run_job () {
  local gpu="$1"
  local model_id="$2"
  local tag="$3"        # sd15 / sd21
  local variant="$4"    # w / w_att

  local run_dir="${IMGROOT}/vis_${tag}_GS_${variant}_seed12345"
  local out_dir="${run_dir}/detect"
  local log="${out_dir}/detect_gs.runlog.txt"

  mkdir -p "$out_dir"

  echo "[LAUNCH] GPU${gpu} ${tag} GS_${variant}" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "$DET" \
      --model_id "$model_id" \
      --run_dir  "$run_dir" \
      --out_dir  "$out_dir" \
      --save_zt \
    > "$log" 2>&1 &

  pids+=("$!")
}

echo "[INFO] Launching 4 GS detect jobs (2 per GPU)..." >&2

# GPU0: SD1.5 (w, w_att)
run_job 0 "$SD14" "sd14" "w"
run_job 0 "$SD15" "sd15" "w"
run_job 0 "$SD15" "sd15" "w_att"

# GPU1: SD2.1 (w, w_att)
run_job 0 "$SD14" "sd14" "w_att"
run_job 1 "$SD21" "sd21" "w"
run_job 1 "$SD21" "sd21" "w_att"

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "[ERROR] job failed: pid=$pid" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[DONE] Some GS detect jobs failed. Check logs under:" >&2
  echo "       ${IMGROOT}/vis_sd*_GS_*_seed12345/detect/detect_gs.runlog.txt" >&2
  exit 1
fi

echo "[DONE] All GS detect jobs finished OK." >&2
echo "[DONE] Outputs under: ${IMGROOT}/vis_sd*_GS_*_seed12345/detect/" >&2
