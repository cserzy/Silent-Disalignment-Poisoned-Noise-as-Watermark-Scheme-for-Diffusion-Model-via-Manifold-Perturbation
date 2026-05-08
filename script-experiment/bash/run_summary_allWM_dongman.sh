#!/usr/bin/env bash
set -euo pipefail

PY=python
OUTDIR="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/result_summary"
mkdir -p "${OUTDIR}"

# ====== 你上传的四个汇总脚本（按你本机实际位置改一下即可）======
GS_SUMMARY="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/result_summary/GS_bit_summary.py"
T2S_SUMMARY="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/result_summary/T2S_bit_summary.py"
PRC_SUMMARY="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/result_summary/PRC_bit_summary-v2.py"
TR_SUMMARY="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/result_summary/TR_bit_summary.py"

# ====== dongman run_dirs：按方法各 6 个 ======
GS_DIRS=(
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_GS_w_att_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_GS_w_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_GS_w_att_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_GS_w_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_GS_w_att_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_GS_w_dongman_seed12345"
)

T2S_DIRS=(
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_T2S_w_att_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_T2S_w_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_T2S_w_att_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_T2S_w_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_T2S_w_att_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_T2S_w_dongman_seed12345"
)

PRC_DIRS=(
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_PRC_w_att_0_85_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_PRC_w_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_PRC_w_att_0_85_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_PRC_w_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_PRC_w_att_0_85_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_PRC_w_dongman_seed12345"
)

TR_DIRS=(
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_TR_w_att_0_88_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_TR_w_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_TR_w_att_0_88_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_TR_w_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_TR_w_att_0_88_dongman_seed12345"
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_TR_w_dongman_seed12345"
)

echo "========== [1/4] GS summary =========="
${PY} "${GS_SUMMARY}" \
  --run_dirs "${GS_DIRS[@]}" \
  --out_xlsx "${OUTDIR}/GS_bit_accuracy_dongman.xlsx"

echo "========== [2/4] T2S summary =========="
${PY} "${T2S_SUMMARY}" \
  --run_dirs "${T2S_DIRS[@]}" \
  --out_xlsx "${OUTDIR}/T2S_bit_accuracy_dongman.xlsx" \
  --verbose

echo "========== [3/4] PRC summary (wrapper, no source edit) =========="
${PY} - <<'PY'
import os, argparse, pandas as pd
from openpyxl import load_workbook
from pathlib import Path

PRC_SUMMARY = Path("/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/result_summary/PRC_bit_summary-v2.py")
OUT_XLSX = Path("/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/result_summary/PRC_bit_accuracy_dongman.xlsx")
RUN_DIRS = [
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_PRC_w_att_0_87_dongman_seed12345",
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_PRC_w_dongman_seed12345",
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_PRC_w_att_0_87_dongman_seed12345",
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_PRC_w_dongman_seed12345",
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_PRC_w_att_0_87_dongman_seed12345",
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_PRC_w_dongman_seed12345",
]

# 动态加载你 PRC_bit_summary-v2.py（文件名里有 -，不能直接 import）
import importlib.machinery, importlib.util
loader = importlib.machinery.SourceFileLoader("prc_sum", str(PRC_SUMMARY))
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
loader.exec_module(mod)

rows = []
grouped = {}  # (model, group) -> list(stats)
for rd in RUN_DIRS:
    model = mod.infer_model_name_from_run_dir(rd)
    group = mod.infer_group_from_run_dir(rd)
    csv_path = os.path.join(rd, "detect_results_prcGLOBAL.csv")
    if not (model and group):
        print("[WARN] skip (cannot infer model/group):", rd)
        continue
    if not os.path.isfile(csv_path):
        print("[WARN] missing:", csv_path)
        continue
    det, acc, n = mod.summarize_prc_csv(csv_path)
    rows.append({"model": model, "group": group, "detect_rate": det, "bit_acc": acc, "n": n, "run_dir": rd, "csv_path": csv_path})
    grouped.setdefault((model, group), []).append((det, acc, n))

if not rows:
    raise RuntimeError("No PRC CSV summarized. Check detect_results_prcGLOBAL.csv exists.")

df_detail = pd.DataFrame(rows).sort_values(["model","group","run_dir"]).reset_index(drop=True)

models = ["SD1.4","SD1.5","SD2.1"]
groups = ["w","w_att"]
summary_rows = []
for m in models:
    row = {"model": m}
    for g in groups:
        det, acc, n = mod.weighted_merge(grouped.get((m,g), []))
        row[f"{g}_detect_rate"] = det
        row[f"{g}_bit_acc"] = acc
        row[f"{g}_n"] = n
    summary_rows.append(row)
df_sum = pd.DataFrame(summary_rows)

OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
    df_sum.to_excel(w, sheet_name="summary", index=False)
    df_detail.to_excel(w, sheet_name="detail_all", index=False)
    for g in groups:
        df_detail[df_detail["group"]==g].to_excel(w, sheet_name=g, index=False)

mod.autosize_and_freeze(str(OUT_XLSX))
print("[OK] wrote:", OUT_XLSX)
PY

echo "========== [4/4] TR summary (wrapper, no source edit) =========="
${PY} - <<'PY'
import glob, os
from pathlib import Path
import openpyxl
from openpyxl import Workbook

TR_SUMMARY = Path("/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/result_summary/TR_bit_summary.py")
OUT_XLSX = Path("/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/result_summary/TR_bit_accuracy_dongman.xlsx")
RUN_DIRS = [
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_TR_w_att_0_88_dongman_seed12345",
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_TR_w_dongman_seed12345",
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_TR_w_att_0_88_dongman_seed12345",
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_TR_w_dongman_seed12345",
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_TR_w_att_0_88_dongman_seed12345",
"/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_TR_w_dongman_seed12345",
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

    xlsx = mod.find_detect_xlsx(rd)   # 期望 run_dir 下仅 1 个 *detect*.xlsx
    dr = mod.read_detect_rate(xlsx)   # 从 xlsx/summary sheet 读 detect_rate
    table.setdefault(model, {})[variant] = dr
    print(f"[OK] {model} {variant} detect_rate={dr:.6f} <- {xlsx}")

# 复用原脚本的写表函数，但输出到我们指定的 OUT_XLSX
mod.write_summary_xlsx(str(OUT_XLSX), table)
print("[DONE] saved:", OUT_XLSX)
PY

echo
echo "[DONE] all dongman summaries saved to: ${OUTDIR}"
ls -lh "${OUTDIR}" | grep -E "dongman|GS_bit_accuracy|T2S_bit_accuracy|PRC_bit_accuracy|TR_bit_accuracy" || true
