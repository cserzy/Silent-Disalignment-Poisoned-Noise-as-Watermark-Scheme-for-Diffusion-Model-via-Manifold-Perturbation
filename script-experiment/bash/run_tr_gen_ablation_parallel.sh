#!/usr/bin/env bash
set -uo pipefail

PY=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/gen_from_zT_bank_multi_models-1_19.py
PROMPTS=/home/yancy/work/dm_backdoor_latent_space/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt
LATDIR=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment
IMGROOT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs

SD15=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers
SD21=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers

ZT_DELSSC=${LATDIR}/generate_TR_w_att_0_88_delssc.pt
ZT_DELREPAIR=${LATDIR}/generate_TR_w_att_0_88_delrepair.pt

COMMON_ARGS=(
  --prompts "$PROMPTS"
  --steps 50 --cfg 7.5 --height 512 --width 512
  --n_per_prompt 4 --start_latent 0
  --dtype fp16 --seed 12345
  --negative_prompt ""
)

pids=()

run_job () {
  local gpu="$1"
  local model_id="$2"
  local tag="$3"      # sd15 / sd21
  local abla="$4"     # delssc / delrepair
  local zt_pt="$5"

  local outdir="${IMGROOT}/vis_${tag}_TR_w_att_0_88_${abla}_seed12345"
  local log="${outdir}.runlog.txt"

  echo "[LAUNCH] GPU${gpu} ${tag} TR_w_att_0_88_${abla}" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "$PY" \
      --model_id "$model_id" \
      --zT_pt "$zt_pt" \
      --outdir "$outdir" \
      "${COMMON_ARGS[@]}" \
    > "$log" 2>&1 &

  pids+=("$!")   # pid of last background job
}

echo "[INFO] Launching 4 TR ablation generation jobs (sd15/sd21 x delssc/delrepair)..." >&2

# GPU0: delssc (2 jobs)
run_job 0 "$SD15" "sd15" "delssc"    "$ZT_DELSSC"
run_job 0 "$SD21" "sd21" "delssc"    "$ZT_DELSSC"

# GPU1: delrepair (2 jobs)
run_job 1 "$SD15" "sd15" "delrepair" "$ZT_DELREPAIR"
run_job 1 "$SD21" "sd21" "delrepair" "$ZT_DELREPAIR"

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "[ERROR] job failed: pid=$pid" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[DONE] Some jobs failed. Check logs:" >&2
  echo "       ${IMGROOT}/vis_sd15_TR_w_att_0_88_*_seed12345.runlog.txt" >&2
  echo "       ${IMGROOT}/vis_sd21_TR_w_att_0_88_*_seed12345.runlog.txt" >&2
  exit 1
fi

echo "[DONE] All 4 TR ablation gen jobs finished OK." >&2
echo "[DONE] Outputs under: ${IMGROOT}/vis_sd{15,21}_TR_w_att_0_88_*_seed12345/sliced/*.png" >&2
