#!/usr/bin/env bash
set -euo pipefail

# Ctrl+C / 退出时把后台任务一起干掉
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT INT TERM

PY=python

# ===== TR detector (use your latest working one) =====
DET_TR=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/detect/detect_TR.py

# ===== models =====
SD14=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers
SD15=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers
SD21=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers

# ===== roots =====
IMGROOT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number

# ===== new hyperparam =====
DIST_THR=80

infer_model_id () {
  local rundir="$1"
  if [[ "$rundir" == *"/vis_sd14_"* ]]; then echo "$SD14"; return; fi
  if [[ "$rundir" == *"/vis_sd15_"* ]]; then echo "$SD15"; return; fi
  if [[ "$rundir" == *"/vis_sd21_"* ]]; then echo "$SD21"; return; fi
  echo "[FATAL] cannot infer model from rundir: $rundir" >&2
  exit 2
}

infer_tag () {
  local rundir="$1"
  if [[ "$rundir" == *"/vis_sd14_"* ]]; then echo "sd14"; return; fi
  if [[ "$rundir" == *"/vis_sd15_"* ]]; then echo "sd15"; return; fi
  if [[ "$rundir" == *"/vis_sd21_"* ]]; then echo "sd21"; return; fi
  echo "sd??"
}

# concurrency=2 => alternate GPUs: 0,1,0,1...
pick_gpu_by_index () {
  local idx="$1"
  echo $(( idx % 2 ))
}

run_stage_cmds () {
  local stage_name="$1"; shift
  local -a cmds=("$@")

  echo
  echo "===================="
  echo "[STAGE] ${stage_name}  (concurrency=2; perGPU=1 via 0,1 mapping)"
  echo "===================="

  # NUL-separated to avoid whitespace issues
  printf '%s\0' "${cmds[@]}" | xargs -0 -I{} -P 2 bash -lc '{}'
}

# -----------------------------
# TR run_dirs (w-only): dongman + number
# -----------------------------
TR_W_DIRS=(
  # dongman (w only)
  "${IMGROOT}/vis_sd14_TR_w_dongman_seed12345"
  "${IMGROOT}/vis_sd15_TR_w_dongman_seed12345"
  "${IMGROOT}/vis_sd21_TR_w_dongman_seed12345"

  # number (w only)
  "${IMGROOT}/vis_sd14_TR_w_number_seed12345"
  "${IMGROOT}/vis_sd15_TR_w_number_seed12345"
  "${IMGROOT}/vis_sd21_TR_w_number_seed12345"
)

# -----------------------------
# build TR commands (with --dist_thr 80)
# -----------------------------
TR_CMDS=()
for i in "${!TR_W_DIRS[@]}"; do
  d="${TR_W_DIRS[$i]}"
  gpu="$(pick_gpu_by_index "$i")"
  model_id="$(infer_model_id "$d")"
  tag="$(infer_tag "$d")"

  img_dir="${d}/sliced"
  log="${d}/detect_TR_dist${DIST_THR}.log"
  out_xlsx="${d}/${tag}_TR_detect_dist${DIST_THR}.xlsx"

  TR_CMDS+=("test -d \"${img_dir}\" || (echo \"[FATAL] missing ${img_dir}\" >&2; exit 2); \
CUDA_VISIBLE_DEVICES=${gpu} ${PY} \"${DET_TR}\" \
  --img_dir \"${img_dir}\" \
  --model_id \"${model_id}\" \
  --out_xlsx \"${out_xlsx}\" \
  --dist_thr ${DIST_THR} \
  > \"${log}\" 2>&1")
done

run_stage_cmds "TR(w-only) dist_thr=${DIST_THR}  [dongman+number]" "${TR_CMDS[@]}"

echo
echo "[DONE] TR(w-only) rerun finished."
echo "[HINT] Logs:"
echo "  ${IMGROOT}/vis_sd*_TR_w_*/*detect_TR_dist${DIST_THR}.log"
