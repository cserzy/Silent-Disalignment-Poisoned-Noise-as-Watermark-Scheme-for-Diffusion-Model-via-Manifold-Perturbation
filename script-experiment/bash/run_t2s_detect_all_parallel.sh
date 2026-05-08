#!/usr/bin/env bash
set -uo pipefail

DET=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/detect/detect_T2S.py
LATDIR=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment
IMGROOT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs

SD14=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers
SD15=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers
SD21=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers

ZT_W=${LATDIR}/generate_T2S_w.pt
ZT_WATT=${LATDIR}/generate_T2S_w_att.pt

# fixed args (keep unchanged)
FIXED_ARGS=(--num_inversion_steps 10 --compute_auc --print_each)

pids=()

run_job () {
  local gpu="$1"
  local model_id="$2"
  local tag="$3"        # sd14 / sd15 / sd21
  local variant="$4"    # w / w_att
  local zt_pt="$5"

  local rundir="${IMGROOT}/vis_${tag}_T2S_${variant}_clip_seed12345"
  local images_glob="${rundir}/sliced/*.png"
  local out_json="${rundir}/t2s_detect_aligned.json"
  local out_csv="${rundir}/t2s_detect_aligned.csv"
  local log="${rundir}/t2s_detect_aligned.runlog.txt"

  echo "[LAUNCH] GPU${gpu} ${tag} T2S_${variant}" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "$DET" \
      --model_id "$model_id" \
      --images_glob "$images_glob" \
      --zT16_pt "$zt_pt" \
      --out_json "$out_json" \
      --out_csv  "$out_csv" \
      "${FIXED_ARGS[@]}" \
    > "$log" 2>&1 &

  pids+=("$!")   # collect PID in *current* shell
}

echo "[INFO] Launching 6 detect jobs at once (3 per GPU)..." >&2

# GPU0: all "w"
#run_job 0 "$SD14" "sd14" "w"     "$ZT_W"
#run_job 0 "$SD15" "sd15" "w"     "$ZT_W"
#run_job 0 "$SD21" "sd21" "w"     "$ZT_W"

# GPU1: all "w_att"
run_job 0 "$SD14" "sd14" "w_att" "$ZT_WATT"
run_job 1 "$SD15" "sd15" "w_att" "$ZT_WATT"
run_job 1 "$SD21" "sd21" "w_att" "$ZT_WATT"

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "[ERROR] job failed: pid=$pid" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[DONE] Some jobs failed. Check logs:" >&2
  echo "       ${IMGROOT}/vis_sd*_T2S_*_seed12345/t2s_detect_aligned.runlog.txt" >&2
  exit 1
fi

echo "[DONE] All 6 detect jobs finished OK." >&2
