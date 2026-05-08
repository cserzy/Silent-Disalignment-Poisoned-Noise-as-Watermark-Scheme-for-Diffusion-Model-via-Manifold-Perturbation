#!/usr/bin/env bash
set -uo pipefail

PY=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/gen_from_zT_bank_multi_models-1_19.py
PROMPTS=/home/yancy/work/dm_backdoor_latent_space/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt
LATDIR=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment
IMGROOT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs

SD14=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers
SD15=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers
SD21=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers

# 两组消融的 zT
ZT_DELSSC=${LATDIR}/generate_GS_w_att_delssc.pt
ZT_DELREPAIR=${LATDIR}/generate_GS_w_att_delrepair.pt

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
  local tag="$3"         # sd14 / sd15 / sd21
  local abla="$4"        # w_att_delssc / w_att_delrepair
  local zt_pt="$5"

  local outdir="${IMGROOT}/vis_${tag}_GS_${abla}_seed12345"
  local log="${outdir}.runlog.txt"

  echo "[LAUNCH] GPU${gpu} ${tag} ${abla}" >&2
  mkdir -p "$outdir"

  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "$PY" \
      --model_id "$model_id" \
      --zT_pt "$zt_pt" \
      --outdir "$outdir" \
      "${COMMON_ARGS[@]}" \
    > "$log" 2>&1 &

  pids+=("$!")  # $! 是最后一个后台任务 PID :contentReference[oaicite:2]{index=2}
}

echo "[INFO] Launching 6 GS ablation jobs (3 per GPU)..." >&2

# 你可以按显存/速度自己调整分配，这里默认：
# GPU0: delrepair 三个模型
run_job 0 "$SD14" "sd14" "w_att_delrepair" "$ZT_DELREPAIR"
run_job 0 "$SD15" "sd15" "w_att_delrepair" "$ZT_DELREPAIR"
run_job 0 "$SD21" "sd21" "w_att_delrepair" "$ZT_DELREPAIR"

# GPU1: delssc 三个模型
run_job 1 "$SD14" "sd14" "w_att_delssc" "$ZT_DELSSC"
run_job 1 "$SD15" "sd15" "w_att_delssc" "$ZT_DELSSC"
run_job 1 "$SD21" "sd21" "w_att_delssc" "$ZT_DELSSC"

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then  # wait 等待指定 PID，并返回其退出码 :contentReference[oaicite:3]{index=3}
    echo "[ERROR] job failed: pid=$pid" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[DONE] Some jobs failed. Check logs:" >&2
  echo "       ${IMGROOT}/vis_sd*_GS_w_att_*_seed12345.runlog.txt" >&2
  exit 1
fi

echo "[DONE] All 6 GS ablation gen jobs finished OK." >&2
echo "[DONE] Outputs under: ${IMGROOT}/vis_sd*_GS_w_att_*_seed12345/sliced/*.png" >&2
