#!/usr/bin/env bash
set -uo pipefail

PY=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/gen_from_zT_bank_multi_models-COCO-1_19.py
PROMPTS=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/prompts/coco_val2017_captions_1000.txt
LATDIR=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment
IMGROOT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs

SD14=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers
SD15=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers
SD21=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers

# ---- your 4 zT (all are w_att) ----
ZT_PRC=${LATDIR}/generate_PRC_w_att_0_85_clip.pt
ZT_GS=${LATDIR}/generate_GS_w_att.pt
ZT_T2S=${LATDIR}/generate_T2S_w_att.pt
ZT_TR=${LATDIR}/generate_TR_w_att_0_88.pt

SEED=12345

COMMON_ARGS=(
  --prompts "$PROMPTS"
  --steps 50 --cfg 7.5 --height 512 --width 512
  --n_per_prompt 1 --start_latent 0
  --dtype fp16 --seed "$SEED"
  --negative_prompt ""
)

pids=()

run_job () {
  local gpu="$1"
  local model_id="$2"
  local tag="$3"        # sd14 / sd15 / sd21
  local wm_tag="$4"     # PRC_w_att_0_85_clip / GS_w_att / T2S_w_att / TR_w_att_0_88
  local zt_pt="$5"

  local outdir="${IMGROOT}/vis_${tag}_${wm_tag}_coco_seed${SEED}"
  local log="${outdir}.runlog.txt"

  echo "[LAUNCH] GPU${gpu} ${tag} ${wm_tag} -> ${outdir}" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "$PY" \
      --model_id "$model_id" \
      --zT_pt "$zt_pt" \
      --outdir "$outdir" \
      "${COMMON_ARGS[@]}" \
    > "$log" 2>&1 &

  pids+=("$!")   # collect PID of last background job
}

echo "[INFO] Launching 11 COCO generation jobs at once (2 GPUs, ~even split)..." >&2
echo "[INFO] NOTE: skip sd14 + TR_w_att_0_88 because you already generated it." >&2

# -------------------------
# GPU0 (6 jobs): sd14(PRC/GS/T2S) + sd15(PRC/GS/T2S)
# -------------------------
run_job 0 "$SD14" "sd14" "PRC_w_att_0_85_clip" "$ZT_PRC"
run_job 0 "$SD14" "sd14" "GS_w_att"           "$ZT_GS"
run_job 0 "$SD14" "sd14" "T2S_w_att"          "$ZT_T2S"
# sd14 + TR_w_att_0_88 SKIPPED

run_job 0 "$SD15" "sd15" "PRC_w_att_0_85_clip" "$ZT_PRC"
run_job 0 "$SD15" "sd15" "GS_w_att"           "$ZT_GS"
run_job 0 "$SD15" "sd15" "T2S_w_att"          "$ZT_T2S"

# -------------------------
# GPU1 (5 jobs): sd15(TR) + sd21(PRC/GS/T2S/TR)
# -------------------------
run_job 1 "$SD15" "sd15" "TR_w_att_0_88"      "$ZT_TR"

run_job 1 "$SD21" "sd21" "PRC_w_att_0_85_clip" "$ZT_PRC"
run_job 1 "$SD21" "sd21" "GS_w_att"           "$ZT_GS"
run_job 1 "$SD21" "sd21" "T2S_w_att"          "$ZT_T2S"
run_job 1 "$SD21" "sd21" "TR_w_att_0_88"      "$ZT_TR"

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "[ERROR] job failed: pid=$pid" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[DONE] Some jobs failed. Check *.runlog.txt under:" >&2
  echo "       ${IMGROOT}/vis_*_coco_seed${SEED}.runlog.txt" >&2
  exit 1
fi

echo "[DONE] All 11 COCO gen jobs finished OK." >&2
echo "[DONE] Outputs under: ${IMGROOT}/vis_*_coco_seed${SEED}/sliced/*.png" >&2
