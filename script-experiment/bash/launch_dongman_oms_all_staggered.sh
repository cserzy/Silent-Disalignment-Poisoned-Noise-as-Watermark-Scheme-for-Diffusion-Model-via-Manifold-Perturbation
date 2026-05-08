#!/usr/bin/env bash
set -uo pipefail

BASH_DIR=/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/bash
LOG_DIR=${BASH_DIR}/dongman_oms_logs
mkdir -p "${LOG_DIR}"

GS_SH=${BASH_DIR}/run_gs_watt_dongman_oms_full.sh
TR_SH=${BASH_DIR}/run_tr_watt_dongman_oms_full.sh
T2S_SH=${BASH_DIR}/run_t2s_watt_dongman_oms_full.sh

echo "============================================================"
echo "[LAUNCH] Dongman OMS staggered pipeline"
echo "GS  : start now"
echo "TR  : start after 1 hour"
echo "T2S : start after 2 hours"
echo "Logs: ${LOG_DIR}"
echo "============================================================"

chmod +x "${GS_SH}" "${TR_SH}" "${T2S_SH}"

bash "${GS_SH}" > "${LOG_DIR}/gs_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

(
  sleep 3600
  bash "${TR_SH}" > "${LOG_DIR}/tr_$(date +%Y%m%d_%H%M%S).log" 2>&1
) &

(
  sleep 7200
  bash "${T2S_SH}" > "${LOG_DIR}/t2s_$(date +%Y%m%d_%H%M%S).log" 2>&1
) &

echo "[OK] All staggered jobs submitted."
echo "Use this to watch:"
echo "  tail -f ${LOG_DIR}/*.log"