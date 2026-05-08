# #!/usr/bin/env bash
# set -uo pipefail

# source /opt/miniconda3/etc/profile.d/conda.sh
# conda activate Hijacking

# ROOT=/home/yancy/work/dm_backdoor_latent_space
# EXP_ROOT=${ROOT}/experiment-1_19
# cd "${EXP_ROOT}" || exit 1

# GEN=${EXP_ROOT}/script-experiment/gen_from_zT_bank_alt_diffusion-1_19.py
# BLACK_DET=${EXP_ROOT}/script-experiment/detect/detect_black_ratio_alt.py
# T2S_DET=${EXP_ROOT}/script-experiment/detect/detect_T2S_alt.py

# MODEL_ID=${ROOT}/checkpoints/AltDiffusion-fp16
# PROMPTS=${ROOT}/prompts/cal_dongman_female_align-jinghua-2026-1_25.txt

# ZT_T2S_W=${ROOT}/experiment-1_19-sd3/latents_experiment/generate_T2S_w.pt
# ZT_T2S_WATT=${EXP_ROOT}/latents_experiment-number/generate_T2S_w_att_dongman.pt

# T2S_META_JSON=${EXP_ROOT}/latents_experiment-number/generate_T2S_w_att_dongman_meta.json

# SEED=12345
# GEN_STEPS=50
# GEN_CFG=7.5
# HEIGHT=512
# WIDTH=512
# DTYPE=fp16
# N_PER_PROMPT=4
# START_LATENT=0
# INV_STEPS=20

# RUN_SAFEOFF_W=${EXP_ROOT}/imgs/vis_alt_dongman_t2s_w_seed${SEED}
# RUN_SAFEOFF_WATT=${EXP_ROOT}/imgs/vis_alt_dongman_t2s_w_att_seed${SEED}

# RUN_SAFEON_W=${EXP_ROOT}/imgs/vis_alt_dongman_t2s_safeon_w_seed${SEED}
# RUN_SAFEON_WATT=${EXP_ROOT}/imgs/vis_alt_dongman_t2s_safeon_w_att_seed${SEED}

# pids=()

# run_gen_job () {
#   local gpu="$1"
#   local tag="$2"
#   local zt_pt="$3"
#   local outdir="$4"
#   local disable_sc="$5"

#   mkdir -p "${outdir}"

#   echo "[LAUNCH][GEN][GPU${gpu}] ${tag} -> ${outdir} | disable_safety_checker=${disable_sc}" >&2

#   if [[ "${disable_sc}" -eq 1 ]]; then
#     CUDA_VISIBLE_DEVICES="${gpu}" \
#       python "${GEN}" \
#         --model_id "${MODEL_ID}" \
#         --prompts "${PROMPTS}" \
#         --zT_pt "${zt_pt}" \
#         --outdir "${outdir}" \
#         --steps ${GEN_STEPS} \
#         --cfg ${GEN_CFG} \
#         --height ${HEIGHT} \
#         --width ${WIDTH} \
#         --device cuda \
#         --dtype ${DTYPE} \
#         --n_per_prompt ${N_PER_PROMPT} \
#         --start_latent ${START_LATENT} \
#         --seed ${SEED} \
#         --disable_safety_checker \
#       > "${outdir}/gen.runlog.txt" 2>&1 &
#   else
#     CUDA_VISIBLE_DEVICES="${gpu}" \
#       python "${GEN}" \
#         --model_id "${MODEL_ID}" \
#         --prompts "${PROMPTS}" \
#         --zT_pt "${zt_pt}" \
#         --outdir "${outdir}" \
#         --steps ${GEN_STEPS} \
#         --cfg ${GEN_CFG} \
#         --height ${HEIGHT} \
#         --width ${WIDTH} \
#         --device cuda \
#         --dtype ${DTYPE} \
#         --n_per_prompt ${N_PER_PROMPT} \
#         --start_latent ${START_LATENT} \
#         --seed ${SEED} \
#       > "${outdir}/gen.runlog.txt" 2>&1 &
#   fi

#   pids+=("$!")
# }

# run_black_job () {
#   local run_dir="$1"
#   local out_dir="${run_dir}/black_detect"

#   mkdir -p "${out_dir}"

#   echo "[LAUNCH][BLACK] ${run_dir}" >&2
#   python "${BLACK_DET}" \
#     --run_dir "${run_dir}" \
#     --out_dir "${out_dir}" \
#     > "${out_dir}/black_detect.runlog.txt" 2>&1 &

#   pids+=("$!")
# }

# run_t2s_job () {
#   local gpu="$1"
#   local run_dir="$2"
#   local out_dir="${run_dir}/detect_t2s_alt"

#   mkdir -p "${out_dir}"

#   echo "[LAUNCH][T2S][GPU${gpu}] ${run_dir}" >&2
#   CUDA_VISIBLE_DEVICES="${gpu}" \
#     python "${T2S_DET}" \
#       --model_id "${MODEL_ID}" \
#       --run_dir "${run_dir}" \
#       --out_dir "${out_dir}" \
#       --cluster_meta_json "${T2S_META_JSON}" \
#       --dtype ${DTYPE} \
#       --inv_steps ${INV_STEPS} \
#       --save_zt \
#     > "${out_dir}/detect_t2s_alt.runlog.txt" 2>&1 &

#   pids+=("$!")
# }

# wait_all () {
#   local fail=0
#   for pid in "${pids[@]}"; do
#     if ! wait "${pid}"; then
#       echo "[ERROR] job failed: pid=${pid}" >&2
#       fail=1
#     fi
#   done
#   pids=()
#   return "${fail}"
# }

# echo "[INFO] Stage 1: generate safe-off T2S runs..." >&2
# #run_gen_job 0 "t2s_w_safeoff" "${ZT_T2S_W}" "${RUN_SAFEOFF_W}" 1
# run_gen_job 0 "t2s_w_att_safeoff" "${ZT_T2S_WATT}" "${RUN_SAFEOFF_WATT}" 1
# wait_all || exit 1

# echo "[INFO] Stage 2: generate safe-on T2S runs..." >&2
# run_gen_job 0 "t2s_w_safeon" "${ZT_T2S_W}" "${RUN_SAFEON_W}" 0
# run_gen_job 0 "t2s_w_att_safeon" "${ZT_T2S_WATT}" "${RUN_SAFEON_WATT}" 0
# wait_all || exit 1

# echo "[INFO] Stage 3: black-rate detection on safe-on runs..." >&2
# run_black_job "${RUN_SAFEON_W}"
# run_black_job "${RUN_SAFEON_WATT}"
# wait_all || exit 1

# echo "[INFO] Stage 4: T2S watermark detection on safe-off runs..." >&2
# run_t2s_job 0 "${RUN_SAFEOFF_W}"
# wait_all || exit 1

# run_t2s_job 0 "${RUN_SAFEOFF_WATT}"

# wait_all || exit 1

# echo "[DONE] Alt Dongman T2S pipeline finished." >&2

#!/usr/bin/env bash
#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
IMGROOT=${EXP_ROOT}/imgs
LATDIR=${EXP_ROOT}/latents_experiment

DET_PY=${EXP_ROOT}/script-experiment/detect/detect_T2S_oms.py
T2S_ROOT=${ROOT}/third_party/T2SMark

SD14=${ROOT}/checkpoints/sd1-4-diffusers
SD15=${ROOT}/checkpoints/sd1-5-diffusers
SD21=${ROOT}/checkpoints/sd2-1-diffusers

T2S_META_PT=${LATDIR}/generate_T2S_w_att_meta.pt
T2S_META_JSON=${LATDIR}/generate_T2S_w_att_meta.json

OMS_WATT_Q=${LATDIR}/oms_Q_T2S_w_att_to_gauss_aligned.pt
OMS_WATT_Q_JSON=${LATDIR}/oms_Q_T2S_w_att_to_gauss_aligned.json

RUN_SD14=${IMGROOT}/vis_sd14_T2S_w_att_oms_gauss_aligned_seed12345
RUN_SD15=${IMGROOT}/vis_sd15_T2S_w_att_oms_gauss_aligned_seed12345
RUN_SD21=${IMGROOT}/vis_sd21_T2S_w_att_oms_gauss_aligned_seed12345

export PYTHONPATH=${T2S_ROOT}:${PYTHONPATH:-}

pids=()

run_bg () {
  local gpu="$1"
  shift
  echo "[LAUNCH][GPU${gpu}] $*" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" "$@" &
  pids+=("$!")
}

wait_all_or_fail () {
  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "[ERROR] detect job failed: pid=$pid" >&2
      fail=1
    fi
  done
  pids=()
  return "$fail"
}

run_det () {
  local gpu="$1"
  local model_id="$2"
  local rundir="$3"
  local tag="$4"

  local outdir="${rundir}/detect_t2s_oms"
  local log="${outdir}/detect_t2s_oms.runlog.txt"
  mkdir -p "${outdir}"

  echo "[QUEUE] ${tag} -> ${outdir}"

  run_bg "${gpu}" \
    python "${DET_PY}" \
      --model_id "${model_id}" \
      --run_dir "${rundir}" \
      --out_dir "${outdir}" \
      --cluster_meta_pt "${T2S_META_PT}" \
      --cluster_meta_json "${T2S_META_JSON}" \
      --oms_q_pt "${OMS_WATT_Q}" \
      --oms_meta_json "${OMS_WATT_Q_JSON}" \
      --t2s_root "${T2S_ROOT}" \
      --inv_steps 50 \
      --dtype fp16 \
      --save_zt_oms \
      --save_zt_restored \
    > "${log}" 2>&1
}

echo
echo "============================================================"
echo "[T2S DETECT ONLY - FIXED] Start"
echo "============================================================"

run_det 0 "${SD14}" "${RUN_SD14}" "sd14"
run_det 1 "${SD15}" "${RUN_SD15}" "sd15"
run_det 1 "${SD21}" "${RUN_SD21}" "sd21"

if ! wait_all_or_fail; then
  echo "[DONE] Some T2S detect jobs failed. Check logs:" >&2
  echo "       ${IMGROOT}/vis_sd*_T2S_w_att_oms_gauss_aligned_seed12345/detect_t2s_oms/detect_t2s_oms.runlog.txt" >&2
  exit 1
fi

echo
echo "============================================================"
echo "[DONE] All T2S detect jobs finished OK."
echo "Outputs:"
echo "  ${RUN_SD14}/detect_t2s_oms"
echo "  ${RUN_SD15}/detect_t2s_oms"
echo "  ${RUN_SD21}/detect_t2s_oms"
echo "============================================================"