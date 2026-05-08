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
  local tag="$3"        # sd14 / sd15 / sd21
  local abla="$4"       # w_att_delssc / w_att_delrepair

  local run_dir="${IMGROOT}/vis_${tag}_GS_${abla}_seed12345"
  local out_dir="${run_dir}/detect"
  local log="${out_dir}/detect_gs.runlog.txt"

  mkdir -p "$out_dir"

  echo "[LAUNCH] GPU${gpu} ${tag} GS_${abla}" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "$DET" \
      --model_id "$model_id" \
      --run_dir  "$run_dir" \
      --out_dir  "$out_dir" \
      --save_zt \
    > "$log" 2>&1 &

  pids+=("$!")  # $! 是最后一个后台任务 PID :contentReference[oaicite:2]{index=2}
}

echo "[INFO] Launching 6 GS detect ablation jobs (3 per GPU)..." >&2

# 你之前生成时就是 GPU0 跑 delrepair，GPU1 跑 delssc；检测这里也同样分配，方便你看 log
# GPU0: delrepair (SD1.4 / SD1.5 / SD2.1)
run_job 0 "$SD14" "sd14" "w_att_delrepair"
run_job 0 "$SD15" "sd15" "w_att_delrepair"
run_job 0 "$SD21" "sd21" "w_att_delrepair"

# GPU1: delssc (SD1.4 / SD1.5 / SD2.1)
run_job 1 "$SD14" "sd14" "w_att_delssc"
run_job 1 "$SD15" "sd15" "w_att_delssc"
run_job 1 "$SD21" "sd21" "w_att_delssc"

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then  # wait 等待后台任务结束并返回退出码 :contentReference[oaicite:3]{index=3}
    echo "[ERROR] job failed: pid=$pid" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[DONE] Some GS detect ablation jobs failed. Check logs under:" >&2
  echo "       ${IMGROOT}/vis_sd*_GS_w_att_*_seed12345/detect/detect_gs.runlog.txt" >&2
  exit 1
fi

echo "[DONE] All GS detect ablation jobs finished OK." >&2
echo "[DONE] Outputs under: ${IMGROOT}/vis_sd*_GS_w_att_*_seed12345/detect/" >&2
