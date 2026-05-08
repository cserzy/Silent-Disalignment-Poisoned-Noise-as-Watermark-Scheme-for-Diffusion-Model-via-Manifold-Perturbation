#!/usr/bin/env bash
set -euo pipefail

# Ctrl+C / 退出时把后台任务一起干掉
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT INT TERM

PY=python

# ===== detectors (use your latest working ones) =====
DET_TR=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/detect/detect_TR.py
DET_GS=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/detect/detect_GS.py
DET_T2S=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/detect/detect_T2S_compat.py
PRC_PY=/home/yancy/work/dm_backdoor_latent_space/prc_detect_global_official_align-1_18_fixdim-meg.py

# ===== models =====
SD14=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers
SD15=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers
SD21=/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd2-1-diffusers

# ===== roots =====
IMGROOT=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number
LATDIR=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment-number

# ===== PRC fixed args (keep exactly as your latest script) =====
PRC_FIXED_ARGS=(--steps 50 --guidance 7.5 --dtype fp32 --var 1.5 --fpr 1e-2 --inv_bs 1 --debug)

# ===== T2S common args (keep as compat script style) =====
T2S_COMMON_ARGS=(
  --fp16
  --num_inversion_steps 10
  --inv_guidance 1.0
  --resize 512
)

# ===== T2S meta: number align (best-effort auto-pick, edit if your filenames differ) =====
META_T2S_W_ATT="${LATDIR}/generate_T2S_w_att_number_meta.pt"
META_T2S_W="${LATDIR}/generate_T2S_w_number_meta.pt"

# fallback: some people store meta under latents_experiment (non-number)
FALLBACK_LATDIR=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment
[[ -f "$META_T2S_W_ATT" ]] || META_T2S_W_ATT="${FALLBACK_LATDIR}/generate_T2S_w_att_meta.pt"
[[ -f "$META_T2S_W"     ]] || META_T2S_W="${FALLBACK_LATDIR}/generate_T2S_w_meta.pt"

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

infer_variant () {
  local rundir="$1"
  if [[ "$rundir" == *"_w_att_"* ]]; then echo "w_att"; else echo "w"; fi
}

# GPU assignment pattern ensures: global conc=4 => perGPU=2 always
pick_gpu_by_index () {
  local idx="$1"
  local m=$(( idx % 4 ))
  case "$m" in
    0) echo 0 ;;
    1) echo 0 ;;
    2) echo 1 ;;
    3) echo 1 ;;
  esac
}

run_stage_cmds () {
  local stage_name="$1"; shift
  local -a cmds=("$@")

  echo
  echo "===================="
  echo "[STAGE] ${stage_name}  (concurrency=4; perGPU=2 via 0,0,1,1 mapping)"
  echo "===================="

  # NUL-separated to avoid whitespace issues
  printf '%s\0' "${cmds[@]}" | xargs -0 -I{} -P 4 bash -lc '{}'
}

# -----------------------------
# run_dirs (24)  number
# -----------------------------
TR_DIRS=(
  "${IMGROOT}/vis_sd14_TR_w_att_0_88_number_seed12345"
  "${IMGROOT}/vis_sd14_TR_w_number_seed12345"
  "${IMGROOT}/vis_sd15_TR_w_att_0_88_number_seed12345"
  "${IMGROOT}/vis_sd15_TR_w_number_seed12345"
  "${IMGROOT}/vis_sd21_TR_w_att_0_88_number_seed12345"
  "${IMGROOT}/vis_sd21_TR_w_number_seed12345"
)

GS_DIRS=(
  "${IMGROOT}/vis_sd14_GS_w_att_number_seed12345"
  "${IMGROOT}/vis_sd14_GS_w_number_seed12345"
  "${IMGROOT}/vis_sd15_GS_w_att_number_seed12345"
  "${IMGROOT}/vis_sd15_GS_w_number_seed12345"
  "${IMGROOT}/vis_sd21_GS_w_att_number_seed12345"
  "${IMGROOT}/vis_sd21_GS_w_number_seed12345"
)

T2S_DIRS=(
  "${IMGROOT}/vis_sd14_T2S_w_att_number_seed12345"
  "${IMGROOT}/vis_sd14_T2S_w_number_seed12345"
  "${IMGROOT}/vis_sd15_T2S_w_att_number_seed12345"
  "${IMGROOT}/vis_sd15_T2S_w_number_seed12345"
  "${IMGROOT}/vis_sd21_T2S_w_att_number_seed12345"
  "${IMGROOT}/vis_sd21_T2S_w_number_seed12345"
)

PRC_DIRS=(
  "${IMGROOT}/vis_sd14_PRC_w_att_0_85_number_seed12345"
  "${IMGROOT}/vis_sd14_PRC_w_number_seed12345"
  "${IMGROOT}/vis_sd15_PRC_w_att_0_85_number_seed12345"
  "${IMGROOT}/vis_sd15_PRC_w_number_seed12345"
  "${IMGROOT}/vis_sd21_PRC_w_att_0_85_number_seed12345"
  "${IMGROOT}/vis_sd21_PRC_w_number_seed12345"
)

# -----------------------------
# build TR commands
# -----------------------------
TR_CMDS=()
for i in "${!TR_DIRS[@]}"; do
  d="${TR_DIRS[$i]}"
  gpu="$(pick_gpu_by_index "$i")"
  model_id="$(infer_model_id "$d")"
  tag="$(infer_tag "$d")"
  log="${d}/detect_TR.log"
  out_xlsx="${d}/${tag}_TR_detect.xlsx"
  img_dir="${d}/sliced"

  TR_CMDS+=("mkdir -p \"$d\"; CUDA_VISIBLE_DEVICES=${gpu} ${PY} \"${DET_TR}\" --img_dir \"${img_dir}\" --model_id \"${model_id}\" --out_xlsx \"${out_xlsx}\" > \"${log}\" 2>&1")
done

# -----------------------------
# build GS commands
# -----------------------------
GS_CMDS=()
for i in "${!GS_DIRS[@]}"; do
  d="${GS_DIRS[$i]}"
  gpu="$(pick_gpu_by_index "$i")"
  model_id="$(infer_model_id "$d")"
  out_dir="${d}/detect"
  log="${out_dir}/detect_gs.runlog.txt"

  GS_CMDS+=("mkdir -p \"${out_dir}\"; CUDA_VISIBLE_DEVICES=${gpu} ${PY} \"${DET_GS}\" --model_id \"${model_id}\" --run_dir \"${d}\" --out_dir \"${out_dir}\" --save_zt > \"${log}\" 2>&1")
done

# -----------------------------
# build T2S commands (compat)
# -----------------------------
T2S_CMDS=()
# sanity: meta must exist
if [[ ! -f "$META_T2S_W_ATT" ]]; then
  echo "[FATAL] T2S meta for w_att not found: $META_T2S_W_ATT" >&2
  echo "        Please set META_T2S_W_ATT to your real meta pt (compat detector requires --cluster_meta_pt)." >&2
  exit 2
fi
if [[ ! -f "$META_T2S_W" ]]; then
  echo "[WARN] T2S meta for w not found: $META_T2S_W" >&2
  echo "       Will reuse w_att meta for w runs (edit META_T2S_W if you have a dedicated one)." >&2
  META_T2S_W="$META_T2S_W_ATT"
fi

for i in "${!T2S_DIRS[@]}"; do
  d="${T2S_DIRS[$i]}"
  gpu="$(pick_gpu_by_index "$i")"
  model_id="$(infer_model_id "$d")"
  tag="$(infer_tag "$d")"
  variant="$(infer_variant "$d")"
  img_dir="${d}/sliced"
  images_glob="${img_dir}/*.png"

  meta_pt="$META_T2S_W"
  [[ "$variant" == "w_att" ]] && meta_pt="$META_T2S_W_ATT"

  out_json="${d}/${tag}_T2S_${variant}_detect.json"
  out_csv="${d}/${tag}_T2S_${variant}_detect.csv"
  log="${d}/${tag}_T2S_${variant}_detect.runlog.txt"

  T2S_CMDS+=("test -d \"${img_dir}\" || (echo \"[FATAL] missing ${img_dir}\" >&2; exit 2); CUDA_VISIBLE_DEVICES=${gpu} ${PY} \"${DET_T2S}\" --cluster_meta_pt \"${meta_pt}\" --images_glob \"${images_glob}\" --model_id \"${model_id}\" --out_json \"${out_json}\" --out_csv \"${out_csv}\" ${T2S_COMMON_ARGS[*]} > \"${log}\" 2>&1")
done

# -----------------------------
# build PRC commands
# -----------------------------
PRC_CMDS=()
for i in "${!PRC_DIRS[@]}"; do
  d="${PRC_DIRS[$i]}"
  gpu="$(pick_gpu_by_index "$i")"
  model_id="$(infer_model_id "$d")"
  log="${d}/prc_detect.runlog.txt"

  PRC_CMDS+=("mkdir -p \"${d}\"; CUDA_VISIBLE_DEVICES=${gpu} ${PY} \"${PRC_PY}\" --run_dir \"${d}\" --model_id \"${model_id}\" ${PRC_FIXED_ARGS[*]} > \"${log}\" 2>&1")
done

# =============================
# run stages in order
# =============================
run_stage_cmds "TR"  "${TR_CMDS[@]}"
run_stage_cmds "GS"  "${GS_CMDS[@]}"
run_stage_cmds "T2S" "${T2S_CMDS[@]}"
run_stage_cmds "PRC" "${PRC_CMDS[@]}"

echo
echo "[DONE] All stages finished."
echo "[HINT] Logs examples:"
echo "  TR : ${IMGROOT}/vis_sd*_TR_*/detect_TR.log"
echo "  GS : ${IMGROOT}/vis_sd*_GS_*/detect/detect_gs.runlog.txt"
echo "  T2S: ${IMGROOT}/vis_sd*_T2S_*/*_detect.runlog.txt"
echo "  PRC: ${IMGROOT}/vis_sd*_PRC_*/prc_detect.runlog.txt"
