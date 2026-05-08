#!/usr/bin/env bash
set -uo pipefail

EVALPY=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/eval_fid_clipscore_coco_one_dir.py
MAP=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/prompts/coco_val2017_captions_1000.map.csv
COCO=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/coco/images/val2017
IMGROOT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs

SEED=12345
B=100

# 你已经跑过的这一组：sd14 + GS_w_att（跳过）
SKIP=${IMGROOT}/vis_sd14_GS_w_att_coco_seed${SEED}

COMMON_ARGS=(
  --coco_map_csv "$MAP"
  --coco_val_dir "$COCO"
  --device cuda
  --dtype fp16
  --bootstrap_B "$B"
  --bootstrap_seed "$SEED"
)

pids=()

run_job () {
  local gpu="$1"
  local run_dir="$2"

  local log="${run_dir}/eval_coco.runlog.txt"
  echo "[LAUNCH] GPU${gpu} eval -> ${run_dir}" >&2

  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "$EVALPY" \
      --run_dir "$run_dir" \
      "${COMMON_ARGS[@]}" \
    >"$log" 2>&1 &

  pids+=("$!")
}

echo "[INFO] Launching 11 eval jobs (skip: ${SKIP}) with B=${B}, seed=${SEED} ..." >&2

# -------------------------
# GPU0: 6 jobs
# -------------------------
run_job 0 "${IMGROOT}/vis_sd14_PRC_w_att_0_85_clip_coco_seed${SEED}"
run_job 0 "${IMGROOT}/vis_sd14_T2S_w_att_coco_seed${SEED}"
run_job 0 "${IMGROOT}/vis_sd14_TR_w_att_0_88_coco_seed${SEED}"

run_job 0 "${IMGROOT}/vis_sd15_PRC_w_att_0_85_clip_coco_seed${SEED}"
run_job 0 "${IMGROOT}/vis_sd15_GS_w_att_coco_seed${SEED}"
run_job 0 "${IMGROOT}/vis_sd15_T2S_w_att_coco_seed${SEED}"

# -------------------------
# GPU1: 5 jobs
# -------------------------
run_job 1 "${IMGROOT}/vis_sd15_TR_w_att_0_88_coco_seed${SEED}"

run_job 1 "${IMGROOT}/vis_sd21_PRC_w_att_0_85_clip_coco_seed${SEED}"
run_job 1 "${IMGROOT}/vis_sd21_GS_w_att_coco_seed${SEED}"
run_job 1 "${IMGROOT}/vis_sd21_T2S_w_att_coco_seed${SEED}"
run_job 1 "${IMGROOT}/vis_sd21_TR_w_att_0_88_coco_seed${SEED}"

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "[ERROR] job failed: pid=$pid" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[DONE] Some eval jobs failed. Check logs:" >&2
  echo "       ${IMGROOT}/vis_*_coco_seed${SEED}/eval_coco.runlog.txt" >&2
  exit 1
fi

echo "[DONE] All 11 eval jobs finished OK." >&2
echo "[DONE] Each run_dir outputs under: run_dir/eval_coco/{clip_summary.json,fid_summary.json,...}" >&2
