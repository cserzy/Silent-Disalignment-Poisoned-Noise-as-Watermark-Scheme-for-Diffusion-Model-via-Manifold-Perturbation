#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from glob import glob
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openpyxl import load_workbook


def infer_model_name_from_run_dir(run_dir: str) -> Optional[str]:
    s = os.path.basename(run_dir).lower()
    if "sd14" in s:
        return "SD1.4"
    if "sd15" in s:
        return "SD1.5"
    if "sd21" in s:
        return "SD2.1"
    return None


def infer_ablation_from_run_dir(run_dir: str) -> Optional[str]:
    s = os.path.basename(run_dir).lower()
    if "delssc" in s:
        return "delssc"
    # 兼容你之前可能写错的 derepair
    if "delrepair" in s or "derepair" in s:
        return "delrepair"
    return None


def summarize_prc_csv(csv_path: str) -> Tuple[float, float, int]:
    """
    detect_results_prcGLOBAL.csv
    Columns: detected, bit_acc
    NaN -> 0
    """
    df = pd.read_csv(csv_path)

    need = {"detected", "bit_acc"}
    if not need.issubset(set(df.columns)):
        raise ValueError(
            f"[ERR] CSV 缺少列 detected/bit_acc: {csv_path}\n"
            f"      实际列名: {list(df.columns)}"
        )

    detected = pd.to_numeric(df["detected"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    bit_acc  = pd.to_numeric(df["bit_acc"],  errors="coerce").fillna(0.0).clip(0.0, 1.0)

    return float(detected.mean()), float(bit_acc.mean()), int(len(df))


def weighted_merge(stats: List[Tuple[float, float, int]]) -> Tuple[float, float, int]:
    """
    若同一个 (model, ablation) 下有多个 run_dir（多次运行），做按样本数 n 加权的均值。
    """
    if not stats:
        return 0.0, 0.0, 0
    total_n = sum(n for _, _, n in stats)
    if total_n <= 0:
        return 0.0, 0.0, 0
    det = sum(det * n for det, _, n in stats) / total_n
    acc = sum(acc * n for _, acc, n in stats) / total_n
    return float(det), float(acc), int(total_n)


def autosize_and_freeze(xlsx_path: str) -> None:
    """
    轻量美化：freeze header + autosize column width（openpyxl）。
    冻结窗格是 openpyxl 标准能力。:contentReference[oaicite:2]{index=2}
    """
    wb = load_workbook(xlsx_path)
    for ws in wb.worksheets:
        ws.freeze_panes = "B2" if ws.title.lower() == "summary" else "A2"

        # autosize width
        for col_cells in ws.columns:
            max_len = 0
            col_letter = col_cells[0].column_letter
            for cell in col_cells:
                v = cell.value
                if v is None:
                    continue
                max_len = max(max_len, len(str(v)))
            ws.column_dimensions[col_letter].width = min(max(10, max_len + 2), 80)

    wb.save(xlsx_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs_root",
                    default="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs",
                    help="imgs 根目录，默认 experiment-1_19/imgs")
    ap.add_argument("--out_xlsx",
                    default="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/result_summary/PRC_bit_accuracy-ablation-0_87_v1.xlsx",
                    help="输出 xlsx 路径")
    ap.add_argument("--csv_name", default="detect_results_prcGLOBAL.csv",
                    help="每个 run_dir 下检测 CSV 的文件名（默认 detect_results_prcGLOBAL.csv）")
    args = ap.parse_args()

    imgs_root = os.path.abspath(args.imgs_root)
    out_xlsx = os.path.abspath(args.out_xlsx)
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)

    # 1) 扫描 run_dir：包含 PRC 且包含 delssc/delrepair
    cand_dirs = sorted([d for d in glob(os.path.join(imgs_root, "vis_*")) if os.path.isdir(d)])
    run_dirs: List[str] = []
    for d in cand_dirs:
        s = os.path.basename(d).lower()
        if "_prc_" not in s:
            continue
        if ("delssc" not in s) and ("delrepair" not in s) and ("derepair" not in s):
            continue
        run_dirs.append(d)

    if not run_dirs:
        raise FileNotFoundError(f"[FATAL] No PRC delssc/delrepair run_dirs found under: {imgs_root}")

    # 2) 逐 run_dir 读 CSV，生成明细表
    rows = []
    grouped: Dict[Tuple[str, str], List[Tuple[float, float, int]]] = {}  # (model, ablation) -> list(stats)

    for rd in run_dirs:
        model = infer_model_name_from_run_dir(rd)
        abla = infer_ablation_from_run_dir(rd)
        if model is None or abla is None:
            print(f"[WARN] Skip (cannot infer model/ablation): {rd}")
            continue

        csv_path = os.path.join(rd, args.csv_name)
        if not os.path.isfile(csv_path):
            print(f"[WARN] Missing CSV: {csv_path}")
            continue

        det, acc, n = summarize_prc_csv(csv_path)

        rows.append({
            "model": model,
            "ablation": abla,
            "detect_rate": det,
            "bit_acc": acc,
            "n": n,
            "run_dir": rd,
            "csv_path": csv_path,
        })

        grouped.setdefault((model, abla), []).append((det, acc, n))

    if not rows:
        raise RuntimeError("[FATAL] No valid CSVs were summarized (check paths & csv_name).")

    df_detail = pd.DataFrame(rows).sort_values(["model", "ablation", "run_dir"]).reset_index(drop=True)

    # 3) 汇总表：按 (model, ablation) 加权平均
    models = ["SD1.4", "SD1.5", "SD2.1"]
    ablas = ["delssc", "delrepair"]

    summary_rows = []
    for m in models:
        row = {"model": m}
        for a in ablas:
            det, acc, n = weighted_merge(grouped.get((m, a), []))
            row[f"{a}_detect_rate"] = det
            row[f"{a}_bit_acc"] = acc
            row[f"{a}_n"] = n
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # 4) 分 sheet 输出
    # 多 sheet 写 Excel 用 ExcelWriter 是 pandas 官方推荐方式。:contentReference[oaicite:3]{index=3}
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df_summary.to_excel(w, sheet_name="summary", index=False)
        df_detail.to_excel(w, sheet_name="detail_all", index=False)

        # 额外拆两张：delssc / delrepair
        for a in ablas:
            sub = df_detail[df_detail["ablation"] == a].copy()
            sub.to_excel(w, sheet_name=a, index=False)

    autosize_and_freeze(out_xlsx)
    print(f"[OK] wrote: {out_xlsx}")
    print(f"[OK] summarized {len(df_detail)} run_dirs")


if __name__ == "__main__":
    main()
