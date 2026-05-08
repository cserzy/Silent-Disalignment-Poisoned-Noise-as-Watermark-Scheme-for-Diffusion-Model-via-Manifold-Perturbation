#!/usr/bin/env bash
set -euo pipefail

# ====== detector (your uploaded detect_T2S.py) ======
DETPY=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/detect/detect_T2S_compat.py

# ====== roots ======
IMGROOT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs
LATDIR=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment

SD14=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers
SD15=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers
SD21=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers

# ====== meta pt (建议用各自消融生成时保存的 meta；没有的话也可以退回默认的 generate_T2S_w_att_meta.pt) ======
META_DELSSC=${LATDIR}/generate_T2S_w_att_delssc_meta.pt
META_DELREPAIR=${LATDIR}/generate_T2S_w_att_delrepair_meta.pt

# ====== 6 run dirs (ONLY THESE you need to adjust if folder names change) ======
RUN_SD14_DELSSC=${IMGROOT}/vis_sd14_T2S_w_att_delssc_seed12345
RUN_SD15_DELSSC=${IMGROOT}/vis_sd15_T2S_w_att_delssc_seed12345
RUN_SD21_DELSSC=${IMGROOT}/vis_sd21_T2S_w_att_delssc_seed12345

RUN_SD14_DELREPAIR=${IMGROOT}/vis_sd14_T2S_w_att_delrepair_seed12345
RUN_SD15_DELREPAIR=${IMGROOT}/vis_sd15_T2S_w_att_delrepair_seed12345
RUN_SD21_DELREPAIR=${IMGROOT}/vis_sd21_T2S_w_att_delrepair_seed12345

# ====== optional common args ======
COMMON_ARGS=(
  --fp16
  --num_inversion_steps 10
  --inv_guidance 1.0
  --resize 512
  # --compute_auc
  # --print_each
)

pids=()

run_job () {
  local gpu="$1"
  local model_id="$2"
  local tag="$3"        # sd14 / sd15 / sd21
  local abla="$4"       # delssc / delrepair
  local run_dir="$5"
  local meta_pt="$6"

  local img_dir="${run_dir}/sliced"
  local images_glob="${img_dir}/*.png"
  local out_json="${run_dir}/${tag}_T2S_w_att_${abla}_detect.json"
  local out_csv="${run_dir}/${tag}_T2S_w_att_${abla}_detect.csv"
  local log="${run_dir}/${tag}_T2S_w_att_${abla}_detect.runlog.txt"

  if [[ ! -d "$img_dir" ]]; then
    echo "[FATAL] missing img_dir: $img_dir" >&2
    exit 2
  fi
  if [[ ! -f "$meta_pt" ]]; then
    echo "[FATAL] missing cluster_meta_pt: $meta_pt" >&2
    exit 2
  fi

  echo "[LAUNCH] GPU${gpu}  ${tag}  T2S_w_att_${abla}  images_glob=${images_glob}" >&2

  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "$DETPY" \
      --cluster_meta_pt "$meta_pt" \
      --images_glob "$images_glob" \
      --model_id "$model_id" \
      --out_json "$out_json" \
      --out_csv "$out_csv" \
      "${COMMON_ARGS[@]}" \
    > "$log" 2>&1 &

  pids+=("$!")  # $! 是最近后台任务 PID :contentReference[oaicite:2]{index=2}
}

echo "[INFO] Launching 6 T2S ablation detect jobs (3 per GPU)..." >&2

# GPU0: delssc
#run_job 0 "$SD14" "sd14" "delssc"    "$RUN_SD14_DELSSC"    "$META_DELSSC"
#run_job 0 "$SD15" "sd15" "delssc"    "$RUN_SD15_DELSSC"    "$META_DELSSC"
#run_job 0 "$SD21" "sd21" "delssc"    "$RUN_SD21_DELSSC"    "$META_DELSSC"

# GPU1: delrepair
run_job 1 "$SD14" "sd14" "delrepair" "$RUN_SD14_DELREPAIR" "$META_DELREPAIR"
#run_job 1 "$SD15" "sd15" "delrepair" "$RUN_SD15_DELREPAIR" "$META_DELREPAIR"
#run_job 1 "$SD21" "sd21" "delrepair" "$RUN_SD21_DELREPAIR" "$META_DELREPAIR"

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "[ERROR] job failed: pid=$pid" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[DONE] Some detect jobs failed. Check logs:" >&2
  echo "       ${IMGROOT}/vis_sd*_T2S_w_att_*_seed12345/*detect.runlog.txt" >&2
  exit 1
fi

echo "[DONE] All 6 detect jobs finished OK." >&2
echo "[DONE] Outputs:" >&2
echo "       ${IMGROOT}/vis_sd*_T2S_w_att_*_seed12345/*detect.json" >&2
echo "       ${IMGROOT}/vis_sd*_T2S_w_att_*_seed12345/*detect.csv" >&2
