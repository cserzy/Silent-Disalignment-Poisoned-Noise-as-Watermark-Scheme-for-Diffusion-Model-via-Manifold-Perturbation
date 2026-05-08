#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
SUMMARY_DIR=${EXP_ROOT}/result_summary
PY=${SUMMARY_DIR}/summarize_dongman_oms_sd.py

echo "[INFO] Using conda env: Hijacking"
echo "[INFO] Python script: ${PY}"

conda run -n Hijacking python "${PY}"
