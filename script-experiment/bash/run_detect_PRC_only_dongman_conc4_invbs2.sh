#!/usr/bin/env bash
set -euo pipefail
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT INT TERM

PY=python
PRC_PY=/home/yancy/work/dm_backdoor_latent_space/prc_detect_global_official_align-1_18_fixdim-meg.py

# models
SD14=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers
SD15=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers
SD21=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers

IMGROOT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number

# fixed args (match your sample; fpr=1e-2, inv_bs=2)
PRC_FIXED_ARGS=(--steps 50 --guidance 7.5 --dtype fp32 --var 1.5 --fpr 1e-2 --inv_bs 2 --debug)

infer_model_id () {
  local rundir="$1"
  if [[ "$rundir" == *"/vis_sd14_"* ]]; then echo "$SD14"; return; fi
  if [[ "$rundir" == *"/vis_sd15_"* ]]; then echo "$SD15"; return; fi
  if [[ "$rundir" == *"/vis_sd21_"* ]]; then echo "$SD21"; return; fi
  echo "[FATAL] cannot infer model from rundir: $rundir" >&2
  exit 2
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

  # NUL-separated for safety; run at most 2 jobs in parallel
  printf '%s\0' "${cmds[@]}" | xargs -0 -I{} -P 2 bash -lc '{}'
}

# dongman PRC run_dirs (6)
PRC_DIRS=(
  "${IMGROOT}/vis_sd14_PRC_w_att_0_85_dongman_seed12345"
  "${IMGROOT}/vis_sd14_PRC_w_dongman_seed12345"
  "${IMGROOT}/vis_sd15_PRC_w_att_0_85_dongman_seed12345"
  "${IMGROOT}/vis_sd15_PRC_w_dongman_seed12345"
  "${IMGROOT}/vis_sd21_PRC_w_att_0_85_dongman_seed12345"
  "${IMGROOT}/vis_sd21_PRC_w_dongman_seed12345"
)

PRC_CMDS=()
for i in "${!PRC_DIRS[@]}"; do
  d="${PRC_DIRS[$i]}"
  gpu="$(pick_gpu_by_index "$i")"
  model_id="$(infer_model_id "$d")"
  log="${d}/prc_detect.rerun_invbs2_fpr1e-2.P2.log"

  PRC_CMDS+=("mkdir -p \"${d}\"; \
if [[ -f \"${d}/detect_results_prcGLOBAL.csv\" ]]; then mv -f \"${d}/detect_results_prcGLOBAL.csv\" \"${d}/detect_results_prcGLOBAL.csv.bak_$(date +%Y%m%d_%H%M%S)\"; fi; \
CUDA_VISIBLE_DEVICES=${gpu} ${PY} \"${PRC_PY}\" --run_dir \"${d}\" --model_id \"${model_id}\" ${PRC_FIXED_ARGS[*]} > \"${log}\" 2>&1")
done

run_stage_cmds "PRC (rerun key fixed)" "${PRC_CMDS[@]}"

echo
echo "[DONE] PRC rerun finished."
echo "Logs:"
echo "  ${IMGROOT}/vis_sd*_PRC_*/prc_detect.rerun_invbs2_fpr1e-2.P2.log"
