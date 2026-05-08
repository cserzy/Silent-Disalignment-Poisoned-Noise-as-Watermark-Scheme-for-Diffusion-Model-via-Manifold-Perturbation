#!/usr/bin/env bash
set -euo pipefail

# ========= common paths =========
SD14="/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers"
PROMPTS="/home/yancy/work/dm_backdoor_latent_space/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt"

EXP_ROOT="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19"
EXPORT_DIR="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment-number"

LOGROOT="/home/yancy/work/dm_backdoor_latent_space/logs/lam_sweep_GS_T2S_queue_like_yours"
mkdir -p "$LOGROOT" "$EXPORT_DIR"

# ========= GS script =========
PY_GS="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/generate_zT/generate_GS_zT_w_att.py"

GS_BASE_ARGS=(
  --model_id "$SD14"
  --prompts "$PROMPTS"
  --margin 0.3
  --steps 30 --cfg 7.5 --height 512 --width 512
  --ssc_d_wm 256
  --gs_seed 12345
  --gs_key_hex aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  --gs_nonce_zero
  --gs_ch 4 --gs_hw 4
  --export_zt_only
  --n_zt 16 --zt_seed 12345
  --export_latents_dir "$EXPORT_DIR"
)

# ========= T2S script (patched) =========
PY_T2S="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/generate_zT/generate_T2S_zT_w_att.py"
T2S_CLUSTER_PT="$EXP_ROOT/latents_experiment/generate_T2S_w.pt"

T2S_BASE_ARGS=(
  --model_id "$SD14"
  --prompts "$PROMPTS"
  --cluster_pt "$T2S_CLUSTER_PT"
  --outdir "$EXP_ROOT"
  --K 16
  --seed 12345
  --ssc_cal_N 12 --ssc_energy_ratio 0.90 --ssc_mini_steps 6
  --t2s_tau 0.674
  --export_latents_dir "$EXPORT_DIR"
)

# ========= lam lists =========
LAM_LIST=(0.88 0.89 0.91 0.92)

lam_tag() { echo "$1" | sed 's/\./_/g'; }   # 0.88 -> 0_88

run_gs_one () {
  local gpu="$1"
  local lam="$2"
  local tag; tag="$(lam_tag "$lam")"

  local outdir="$LOGROOT/GS/lam${tag}"
  local outname="generate_GS_w_att_${tag}.pt"
  local runlog="$LOGROOT/GS/lam${tag}.gpu${gpu}.runlog.txt"
  mkdir -p "$outdir" "$(dirname "$runlog")"

  echo "[LAUNCH][GPU${gpu}] GS lam1=${lam} -> ${EXPORT_DIR}/${outname}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "$PY_GS" \
    --outdir "$outdir" \
    --lambda1 "$lam" \
    --export_latents_name "$outname" \
    "${GS_BASE_ARGS[@]}" \
    > "$runlog" 2>&1
  echo "[OK][GPU${gpu}] GS lam1=${lam} done"
}

run_t2s_one () {
  local gpu="$1"
  local lam="$2"
  local tag; tag="$(lam_tag "$lam")"

  local runlog="$LOGROOT/T2S/lam${tag}.gpu${gpu}.runlog.txt"
  mkdir -p "$(dirname "$runlog")"

  local out_pt="generate_T2S_w_att_${tag}.pt"
  local out_pre="generate_T2S_w_att_${tag}_pre.pt"
  local meta_pt="generate_T2S_w_att_${tag}_meta.pt"
  local meta_js="generate_T2S_w_att_${tag}_meta.json"

  echo "[LAUNCH][GPU${gpu}] T2S lam1=${lam} -> ${EXPORT_DIR}/${out_pt}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "$PY_T2S" \
    --lam1 "$lam" \
    --out_pt_name "$out_pt" \
    --out_pre_pt_name "$out_pre" \
    --meta_pt_name "$meta_pt" \
    --meta_json_name "$meta_js" \
    "${T2S_BASE_ARGS[@]}" \
    > "$runlog" 2>&1
  echo "[OK][GPU${gpu}] T2S lam1=${lam} done"
}

echo "[INFO] Queue style like your uploaded script: two subshell queues + tee + wait"
echo "[INFO] GS lam list:  ${LAM_LIST[*]}"
echo "[INFO] T2S lam list: ${LAM_LIST[*]}"
echo "[INFO] Logs root: $LOGROOT"
nvidia-smi || true

# ============================
# GPU0 队列：顺序跑（你可以在这里显式排队）
# 这里我按“先 GS 再 T2S”的顺序排
# ============================
(
  # ---- GS on GPU0 ----
  run_gs_one 0 0.88
  run_gs_one 0 0.89

  # ---- T2S on GPU0 ----
  run_t2s_one 0 0.88
  run_t2s_one 0 0.89
) 2>&1 | tee "$LOGROOT/queue_gpu0.log" &

PID0=$!

# ============================
# GPU1 队列：顺序跑
# ============================
(
  # ---- GS on GPU1 ----
  run_gs_one 1 0.91
  run_gs_one 1 0.92

  # ---- T2S on GPU1 ----
  run_t2s_one 1 0.91
  run_t2s_one 1 0.92
) 2>&1 | tee "$LOGROOT/queue_gpu1.log" &

PID1=$!

echo "[INFO] Launched queues: GPU0 PID=$PID0, GPU1 PID=$PID1"
wait "$PID0" "$PID1"

echo "[DONE] All GS+T2S lam sweep queues finished."
echo "[DONE] Exported to: $EXPORT_DIR"
echo "[DONE] Logs: $LOGROOT"
nvidia-smi || true
