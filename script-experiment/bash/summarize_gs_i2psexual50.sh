#!/usr/bin/env bash
set -u
set -o pipefail

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
IMGROOT=${EXP_ROOT}/imgs_i2psexual50
SEED=12345

OUTDIR=${EXP_ROOT}/summary_i2psexual50
mkdir -p "${OUTDIR}"

SUMMARY_TSV=${OUTDIR}/summary_i2psexual50.tsv
LOGTXT=${OUTDIR}/summary_i2psexual50.log.txt

: > "${LOGTXT}"

echo -e "run_tag\trun_dir\tmanifest_exists\tnsfw_report_exists\tdetect_dir_exists\tdetect_csv_exists\tdetect_summary_exists\tnsfw_report_path\tdetect_csv_path\tdetect_summary_path" > "${SUMMARY_TSV}"

find_detect_csv () {
  local d="$1"
  for f in \
    "${d}/gs_detect_invert_decode.csv" \
    "${d}/detect.csv" \
    "${d}/result.csv" \
    "${d}/results.csv"
  do
    if [[ -f "${f}" ]]; then
      echo "${f}"
      return 0
    fi
  done
  echo ""
}

find_detect_summary () {
  local d="$1"
  for f in \
    "${d}/summary.txt" \
    "${d}/summary.json" \
    "${d}/detect_summary.txt" \
    "${d}/detect_summary.json" \
    "${d}/report.txt"
  do
    if [[ -f "${f}" ]]; then
      echo "${f}"
      return 0
    fi
  done
  echo ""
}

collect_one () {
  local model_tag="$1"
  local variant="$2"

  local run_tag="${model_tag}_${variant}"
  local run_dir="${IMGROOT}/vis_i2psexual50_${model_tag}_${variant}_seed${SEED}"

  local manifest="${run_dir}/sliced/manifest.csv"
  local nsfw_report="${run_dir}/nsfw_report/report.xlsx"
  local detect_dir="${run_dir}/detect"

  local manifest_exists=0
  local nsfw_exists=0
  local detect_dir_exists=0
  local detect_csv_exists=0
  local detect_summary_exists=0

  [[ -f "${manifest}" ]] && manifest_exists=1
  [[ -f "${nsfw_report}" ]] && nsfw_exists=1
  [[ -d "${detect_dir}" ]] && detect_dir_exists=1

  local detect_csv=""
  local detect_summary=""

  if [[ "${detect_dir_exists}" -eq 1 ]]; then
    detect_csv="$(find_detect_csv "${detect_dir}")"
    detect_summary="$(find_detect_summary "${detect_dir}")"
    [[ -n "${detect_csv}" ]] && detect_csv_exists=1
    [[ -n "${detect_summary}" ]] && detect_summary_exists=1
  fi

  echo -e "${run_tag}\t${run_dir}\t${manifest_exists}\t${nsfw_exists}\t${detect_dir_exists}\t${detect_csv_exists}\t${detect_summary_exists}\t${nsfw_report}\t${detect_csv}\t${detect_summary}" >> "${SUMMARY_TSV}"

  {
    echo "============================================================"
    echo "[RUN] ${run_tag}"
    echo "run_dir = ${run_dir}"
    echo "manifest_exists = ${manifest_exists}"
    echo "nsfw_report_exists = ${nsfw_exists}"
    echo "detect_dir_exists = ${detect_dir_exists}"
    echo "detect_csv = ${detect_csv}"
    echo "detect_summary = ${detect_summary}"

    if [[ -f "${detect_summary}" ]]; then
      echo "--- detect summary head ---"
      head -n 20 "${detect_summary}" || true
    fi

    if [[ -f "${detect_csv}" ]]; then
      echo "--- detect csv head ---"
      head -n 5 "${detect_csv}" || true
    fi

    echo
  } >> "${LOGTXT}"
}

# 9 runs
collect_one sd14 gauss
collect_one sd14 gsw
collect_one sd14 gswatt

collect_one sd15 gauss
collect_one sd15 gsw
collect_one sd15 gswatt

collect_one sd21 gauss
collect_one sd21 gsw
collect_one sd21 gswatt

echo "[DONE] summary tsv: ${SUMMARY_TSV}"
echo "[DONE] summary log: ${LOGTXT}"