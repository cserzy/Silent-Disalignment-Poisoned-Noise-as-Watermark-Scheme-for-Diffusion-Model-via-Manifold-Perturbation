#!/usr/bin/env bash
set -euo pipefail

PY=python
OUTDIR="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/result_summary"
mkdir -p "${OUTDIR}"

# ====== 四个汇总脚本路径 ======
GS_SUMMARY="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/result_summary/GS_bit_summary.py"
T2S_SUMMARY="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/result_summary/T2S_bit_summary.py"
PRC_SUMMARY="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/result_summary/PRC_bit_summary-v1.py"
TR_SUMMARY="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/result_summary/TR_bit_summary.py"

# ====== number run_dirs：按方法各 6 个 ======
GS_DIRS=(
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_GS_w_att_number_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_GS_w_number_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_GS_w_att_number_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_GS_w_number_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_GS_w_att_number_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_GS_w_number_seed12345"
)

T2S_DIRS=(
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_T2S_w_att_number_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_T2S_w_number_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_T2S_w_att_number_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_T2S_w_number_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_T2S_w_att_number_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_T2S_w_number_seed12345"
)

TR_DIRS=(
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_TR_w_att_0_88_number_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_TR_w_number_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_TR_w_att_0_88_number_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_TR_w_number_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_TR_w_att_0_88_number_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_TR_w_number_seed12345"
)

echo "========== [1/4] GS summary (number) =========="
${PY} "${GS_SUMMARY}" \
  --run_dirs "${GS_DIRS[@]}" \
  --out_xlsx "${OUTDIR}/GS_bit_accuracy_number.xlsx"

echo "========== [2/4] T2S summary (number) =========="
${PY} "${T2S_SUMMARY}" \
  --run_dirs "${T2S_DIRS[@]}" \
  --out_xlsx "${OUTDIR}/T2S_bit_accuracy_number.xlsx" \
  --verbose

echo "========== [3/4] PRC summary (number, use v1 directly) =========="
# 你已在 PRC_bit_summary-v1.py 内部把 run_dirs 适配到 PRC-number；
# 这里直接指定输出文件名即可。
${PY} "${PRC_SUMMARY}" \
  --imgs_root "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number" \
  --out_xlsx "${OUTDIR}/PRC_bit_accuracy_number.xlsx"

echo "========== [4/4] TR summary (number, wrapper, no source edit) =========="
${PY} - <<'PY'
import os
from pathlib import Path

TR_SUMMARY = Path("/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/result_summary/TR_bit_summary.py")
OUT_XLSX = Path("/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/result_summary/TR_bit_accuracy_number.xlsx")
RUN_DIRS = [
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_TR_w_att_0_88_number_seed12345",
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_TR_w_number_seed12345",
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_TR_w_att_0_88_number_seed12345",
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_TR_w_number_seed12345",
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_TR_w_att_0_88_number_seed12345",
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_TR_w_number_seed12345",
]

import importlib.machinery, importlib.util
loader = importlib.machinery.SourceFileLoader("tr_sum", str(TR_SUMMARY))
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
loader.exec_module(mod)

table = {}
seen = set()
for rd in RUN_DIRS:
    model, variant = mod.parse_tags(rd)
    if (model, variant) in seen:
        raise RuntimeError(f"duplicated (model,variant)={model,variant} rd={rd}")
    seen.add((model, variant))

    xlsx = mod.find_detect_xlsx(rd)
    dr = mod.read_detect_rate(xlsx)
    table.setdefault(model, {})[variant] = dr
    print(f"[OK] {model} {variant} detect_rate={dr:.6f} <- {xlsx}")

OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
mod.write_summary_xlsx(str(OUT_XLSX), table)
print("[DONE] saved:", OUT_XLSX)
PY

echo
echo "[DONE] all number summaries saved to: ${OUTDIR}"
ls -lh "${OUTDIR}" | grep -E "number|GS_bit_accuracy_number|T2S_bit_accuracy_number|PRC_bit_accuracy_number|TR_bit_accuracy_number" || true
