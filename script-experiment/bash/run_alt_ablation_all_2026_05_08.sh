#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
LATDIR=${EXP_ROOT}/latents_experiment
IMGROOT=${EXP_ROOT}/imgs

cd "${EXP_ROOT}" || exit 1

PROMPTS=${ROOT}/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt
MODEL_ID=${ROOT}/checkpoints/AltDiffusion-fp16

GEN=${EXP_ROOT}/script-experiment/gen_from_zT_bank_alt_diffusion-1_19.py
BLACK_DET=${EXP_ROOT}/script-experiment/detect/detect_black_ratio_alt.py

GS_DET=${EXP_ROOT}/script-experiment/detect/detect_GS_alt.py
TR_DET=${EXP_ROOT}/script-experiment/detect/detect_TR_alt.py
T2S_DET=${EXP_ROOT}/script-experiment/detect/detect_T2S_alt.py
PRC_DET=${EXP_ROOT}/script-experiment/detect/prc_detect_alt_global_official_align.py

WM_META=${LATDIR}/wm_meta

LOCK0=${EXP_ROOT}/.alt_ablation_gpu0.lock
LOCK1=${EXP_ROOT}/.alt_ablation_gpu1.lock

SEED=12345
GEN_STEPS=50
GEN_CFG=7.5
HEIGHT=512
WIDTH=512
DTYPE=fp16
N_PER_PROMPT=4
START_LATENT=0
INV_STEPS=50

PRC_FPR=1e-2
PRC_MASTER_KEY=prc_key_sd14_0124
PRC_MESSAGE_LENGTH=8
PRC_MAX_BP_ITER=5000

mkdir -p "${IMGROOT}"

need_file () {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] missing file: $1" >&2
    exit 1
  fi
}

need_dir () {
  if [[ ! -d "$1" ]]; then
    echo "[ERROR] missing dir: $1" >&2
    exit 1
  fi
}

num_pngs () {
  local run_dir="$1"
  if [[ -d "${run_dir}/sliced" ]]; then
    find "${run_dir}/sliced" -maxdepth 1 -type f -name "*.png" | wc -l | tr -d ' '
  else
    echo 0
  fi
}

has_generation_done () {
  local run_dir="$1"
  local n
  n=$(num_pngs "${run_dir}")
  [[ -f "${run_dir}/sliced/manifest.csv" && "${n}" -gt 0 ]]
}

run_gpu () {
  local gpu="$1"
  local log_file="$2"
  shift 2

  local lock_file
  if [[ "${gpu}" == "0" ]]; then
    lock_file="${LOCK0}"
  else
    lock_file="${LOCK1}"
  fi

  mkdir -p "$(dirname "${log_file}")"

  (
    flock -x 9
    echo "[GPU${gpu}-LOCK] start: $*" >&2
    CUDA_VISIBLE_DEVICES="${gpu}" "$@" > "${log_file}" 2>&1
    echo "[GPU${gpu}-LOCK] done: $*" >&2
  ) 9>"${lock_file}"
}

run_gen_pair () {
  local gpu="$1"
  local tag="$2"
  local zt_pt="$3"

  local out_safeoff="${IMGROOT}/vis_alt_ablate_${tag}_seed${SEED}"
  local out_safeon="${IMGROOT}/vis_alt_ablate_safeon_${tag}_seed${SEED}"

  need_file "${zt_pt}"

  if has_generation_done "${out_safeoff}"; then
    echo "[SKIP][GEN][safe-off] ${tag}: exists, pngs=$(num_pngs "${out_safeoff}")" >&2
  else
    echo "[GEN][GPU${gpu}][safe-off] ${tag}" >&2
    mkdir -p "${out_safeoff}"
    run_gpu "${gpu}" "${out_safeoff}/gen.runlog.txt" \
      python "${GEN}" \
        --model_id "${MODEL_ID}" \
        --prompts "${PROMPTS}" \
        --zT_pt "${zt_pt}" \
        --outdir "${out_safeoff}" \
        --steps ${GEN_STEPS} \
        --cfg ${GEN_CFG} \
        --height ${HEIGHT} \
        --width ${WIDTH} \
        --device cuda \
        --dtype ${DTYPE} \
        --n_per_prompt ${N_PER_PROMPT} \
        --start_latent ${START_LATENT} \
        --seed ${SEED} \
        --disable_safety_checker
  fi

  if has_generation_done "${out_safeon}"; then
    echo "[SKIP][GEN][safe-on] ${tag}: exists, pngs=$(num_pngs "${out_safeon}")" >&2
  else
    echo "[GEN][GPU${gpu}][safe-on] ${tag}" >&2
    mkdir -p "${out_safeon}"
    run_gpu "${gpu}" "${out_safeon}/gen.runlog.txt" \
      python "${GEN}" \
        --model_id "${MODEL_ID}" \
        --prompts "${PROMPTS}" \
        --zT_pt "${zt_pt}" \
        --outdir "${out_safeon}" \
        --steps ${GEN_STEPS} \
        --cfg ${GEN_CFG} \
        --height ${HEIGHT} \
        --width ${WIDTH} \
        --device cuda \
        --dtype ${DTYPE} \
        --n_per_prompt ${N_PER_PROMPT} \
        --start_latent ${START_LATENT} \
        --seed ${SEED}
  fi
}

run_black_detect () {
  local tag="$1"
  local run_dir="${IMGROOT}/vis_alt_ablate_safeon_${tag}_seed${SEED}"
  local out_dir="${run_dir}/black_detect"

  if [[ -f "${out_dir}/black_ratio_alt_summary.json" ]]; then
    echo "[SKIP][BLACK] ${tag}: ${out_dir}/black_ratio_alt_summary.json exists" >&2
    return 0
  fi

  mkdir -p "${out_dir}"
  echo "[BLACK] ${tag}" >&2
  python "${BLACK_DET}" \
    --run_dir "${run_dir}" \
    --out_dir "${out_dir}" \
    > "${out_dir}/black_detect.runlog.txt" 2>&1
}

run_gs_detect () {
  local gpu="$1"
  local tag="$2"
  local run_dir="${IMGROOT}/vis_alt_ablate_${tag}_seed${SEED}"
  local out_dir="${run_dir}/detect_gs_alt"

  if [[ -f "${out_dir}/gs_detect_invert_decode_alt_summary.json" ]]; then
    echo "[SKIP][GS-DETECT] ${tag}: summary exists" >&2
    return 0
  fi

  mkdir -p "${out_dir}"
  echo "[GS-DETECT][GPU${gpu}] ${tag}" >&2
  run_gpu "${gpu}" "${out_dir}/detect_gs_alt.runlog.txt" \
    python "${GS_DET}" \
      --model_id "${MODEL_ID}" \
      --run_dir "${run_dir}" \
      --out_dir "${out_dir}" \
      --dtype ${DTYPE} \
      --inv_steps ${INV_STEPS} \
      --save_zt
}

run_tr_detect () {
  local gpu="$1"
  local tag="$2"
  local run_dir="${IMGROOT}/vis_alt_ablate_${tag}_seed${SEED}"
  local out_dir="${run_dir}/detect_tr_alt"

  if [[ -f "${out_dir}/treering_detect_alt_img.json" ]]; then
    echo "[SKIP][TR-DETECT] ${tag}: json exists" >&2
    return 0
  fi

  mkdir -p "${out_dir}"
  echo "[TR-DETECT][GPU${gpu}] ${tag}" >&2
  run_gpu "${gpu}" "${out_dir}/detect_tr_alt.runlog.txt" \
    python "${TR_DET}" \
      --img_dir "${run_dir}" \
      --model_id "${MODEL_ID}" \
      --mode img \
      --detect_prompt empty \
      --guidance_scale 1.0 \
      --steps 50 \
      --inv_steps ${INV_STEPS} \
      --fp16 1 \
      --out_dir "${out_dir}" \
      --save_zt
}

run_t2s_detect () {
  local gpu="$1"
  local tag="$2"
  local meta_pt="$3"
  local run_dir="${IMGROOT}/vis_alt_ablate_${tag}_seed${SEED}"
  local out_dir="${run_dir}/detect_t2s_alt"

  need_file "${meta_pt}"

  if [[ -f "${out_dir}/t2s_detect_alt_results.json" ]]; then
    echo "[SKIP][T2S-DETECT] ${tag}: json exists" >&2
    return 0
  fi

  mkdir -p "${out_dir}"
  echo "[T2S-DETECT][GPU${gpu}] ${tag}" >&2
  run_gpu "${gpu}" "${out_dir}/detect_t2s_alt.runlog.txt" \
    python "${T2S_DET}" \
      --model_id "${MODEL_ID}" \
      --run_dir "${run_dir}" \
      --out_dir "${out_dir}" \
      --cluster_meta_pt "${meta_pt}" \
      --dtype ${DTYPE} \
      --inv_steps ${INV_STEPS} \
      --save_zt \
      --compute_auc
}

run_prc_detect () {
  local gpu="$1"
  local tag="$2"
  local run_dir="${IMGROOT}/vis_alt_ablate_${tag}_seed${SEED}"
  local out_dir="${run_dir}/detect_prc_alt"

  need_dir "${WM_META}"
  need_file "${PRC_DET}"

  if [[ -f "${out_dir}/detect_results_prcGLOBAL_alt.csv" ]]; then
    echo "[SKIP][PRC-DETECT] ${tag}: csv exists" >&2
    return 0
  fi

  mkdir -p "${out_dir}"
  echo "[PRC-DETECT][GPU${gpu}] ${tag}" >&2
  run_gpu "${gpu}" "${out_dir}/detect_prc_alt.runlog.txt" \
    python "${PRC_DET}" \
      --model_id "${MODEL_ID}" \
      --run_dir "${run_dir}" \
      --meta_root "${WM_META}" \
      --dtype "${DTYPE}" \
      --inv_steps "${INV_STEPS}" \
      --inv_bs 1 \
      --fpr "${PRC_FPR}" \
      --master_key "${PRC_MASTER_KEY}" \
      --message_length "${PRC_MESSAGE_LENGTH}" \
      --max_bp_iter "${PRC_MAX_BP_ITER}" \
      --save_zt \
      --save_zt_dir "${out_dir}/latents_prc_alt" \
      --out_csv "${out_dir}/detect_results_prcGLOBAL_alt.csv"
}

run_gs_resume () {
  echo "========== [GS-RESUME][GPU0] ==========" >&2

  run_gen_pair 0 "gs_delrepair" "${LATDIR}/generate_GS_w_att_delrepair.pt"
  run_black_detect "gs_delrepair"
  run_gs_detect 0 "gs_delrepair"

  run_gen_pair 0 "gs_delssc" "${LATDIR}/generate_GS_w_att_delssc.pt"
  run_black_detect "gs_delssc"
  run_gs_detect 0 "gs_delssc"

  echo "========== [GS-RESUME] done ==========" >&2
}

run_t2s_resume () {
  echo "========== [T2S-RESUME][GPU0] ==========" >&2

  run_gen_pair 0 "t2s_delrepair" "${LATDIR}/generate_T2S_w_att_delrepair.pt"
  run_black_detect "t2s_delrepair"
  run_t2s_detect 0 "t2s_delrepair" "${LATDIR}/generate_T2S_w_att_delrepair_meta.pt"

  run_gen_pair 0 "t2s_delssc" "${LATDIR}/generate_T2S_w_att_delssc.pt"
  run_black_detect "t2s_delssc"
  run_t2s_detect 0 "t2s_delssc" "${LATDIR}/generate_T2S_w_att_delssc_meta.pt"

  echo "========== [T2S-RESUME] done ==========" >&2
}

run_tr_resume () {
  echo "========== [TR-RESUME][GPU1] ==========" >&2

  run_gen_pair 1 "tr_delrepair" "${LATDIR}/generate_TR_w_att_0_88_delrepair.pt"
  run_black_detect "tr_delrepair"
  run_tr_detect 1 "tr_delrepair"

  run_gen_pair 1 "tr_delssc" "${LATDIR}/generate_TR_w_att_0_88_delssc.pt"
  run_black_detect "tr_delssc"
  run_tr_detect 1 "tr_delssc"

  echo "========== [TR-RESUME] done ==========" >&2
}

run_prc_resume () {
  echo "========== [PRC-RESUME][GPU1] ==========" >&2

  run_gen_pair 1 "prc_delrepair" "${LATDIR}/generate_PRC_w_att_0_85_delrepair.pt"
  run_black_detect "prc_delrepair"
  run_prc_detect 1 "prc_delrepair"

  run_gen_pair 1 "prc_delssc" "${LATDIR}/generate_PRC_w_att_0_85_delssc.pt"
  run_black_detect "prc_delssc"
  run_prc_detect 1 "prc_delssc"

  echo "========== [PRC-RESUME] done ==========" >&2
}

run_gs_resume  > "${EXP_ROOT}/script-experiment/bash/log_alt_ablation_resume_gs.txt" 2>&1 &
run_t2s_resume > "${EXP_ROOT}/script-experiment/bash/log_alt_ablation_resume_t2s.txt" 2>&1 &
run_tr_resume  > "${EXP_ROOT}/script-experiment/bash/log_alt_ablation_resume_tr.txt" 2>&1 &
run_prc_resume > "${EXP_ROOT}/script-experiment/bash/log_alt_ablation_resume_prc.txt" 2>&1 &

wait

echo "[DONE] AltDiff ablation resume dual-GPU pipeline finished." >&2
echo "[LOGS]" >&2
echo "  ${EXP_ROOT}/script-experiment/bash/log_alt_ablation_resume_gs.txt" >&2
echo "  ${EXP_ROOT}/script-experiment/bash/log_alt_ablation_resume_t2s.txt" >&2
echo "  ${EXP_ROOT}/script-experiment/bash/log_alt_ablation_resume_tr.txt" >&2
echo "  ${EXP_ROOT}/script-experiment/bash/log_alt_ablation_resume_prc.txt" >&2