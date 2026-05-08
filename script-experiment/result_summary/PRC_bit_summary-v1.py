#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
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


def infer_group_from_run_dir(run_dir: str) -> Optional[str]:
    """
    Group = w / w_att
    兼容：目录可能含 clip 后缀，但不影响分组。
    """
    s = os.path.basename(run_dir).lower()

    # 先判 w_att，避免被 "_w_" 误判
    if "_w_att_" in s or s.endswith("_w_att") or "_w_att" in s:
        return "w_att"

    # 再判 w（注意排除 w_att 已经先返回）
    if "_w_" in s or s.endswith("_w") or "_w" in s:
        return "w"

    return None


def summarize_prc_csv(csv_path: str) -> Tuple[float, float, int]:
    """
    V1 口径（detected-only）：
    - 只需要列 detected
    - NaN -> 0
    - detect_rate = mean(detected)
    - bit_acc 固定为 1.0（不再汇总）
    """
    df = pd.read_csv(csv_path)

    need = {"detected"}
    if not need.issubset(set(df.columns)):
        raise ValueError(
            f"[ERR] CSV 缺少列 detected: {csv_path}\n"
            f"      实际列名: {list(df.columns)}"
        )

    detected = (
        pd.to_numeric(df["detected"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
        .astype(int)
    )

    detect_rate = float(detected.mean())
    bit_acc = 1.0  # V1：不再统计 bit_acc，统一写 1

    return detect_rate, bit_acc, int(len(df))


def weighted_merge(stats: List[Tuple[float, float, int]]) -> Tuple[float, float, int]:
    """
    若同一个 (model, group) 下有多个 run_dir（多次运行），做按样本数 n 加权的均值。
    V1：仅对 detect_rate 做加权；bit_acc 固定 1.0（若 total_n>0）。
    """
    if not stats:
        return 0.0, 0.0, 0

    total_n = sum(n for _, _, n in stats)
    if total_n <= 0:
        return 0.0, 0.0, 0

    det = sum(det * n for det, _, n in stats) / total_n
    acc = 1.0  # V1：只要该组有数据，bit_acc 统一为 1
    return float(det), float(acc), int(total_n)


def autosize_and_freeze(xlsx_path: str) -> None:
    """
    轻量美化：freeze header + autosize column width（openpyxl）
    """
    wb = load_workbook(xlsx_path)
    for ws in wb.worksheets:
        ws.freeze_panes = "B2" if ws.title.lower() == "summary" else "A2"

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
    ap.add_argument(
        "--imgs_root",
        default="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number",
        help="imgs 根目录，默认 experiment-1_19/imgs",
    )
    ap.add_argument(
        "--out_xlsx",
        default="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/result_summary/PRC_bit_accuary-v1-number.xlsx",
        help="输出 xlsx 路径",
    )
    ap.add_argument(
        "--csv_name",
        default="detect_results_prcGLOBAL.csv",
        help="每个 run_dir 下检测 CSV 文件名",
    )
    args = ap.parse_args()

    imgs_root = os.path.abspath(args.imgs_root)
    out_xlsx = os.path.abspath(args.out_xlsx)
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)

    # 1) 写死 run_dir（避免 imgs_root 下其他 vis_* 名字冲突）
    # NOTE: 你后续换主题（dongman/number），直接改这个列表即可。
    run_dirs: List[str] = [
        "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_PRC_w_att_0_85_number_seed12345",
        "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_PRC_w_number_seed12345",
        "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_PRC_w_att_0_85_number_seed12345",
        "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_PRC_w_number_seed12345",
        "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_PRC_w_att_0_85_number_seed12345",
        "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_PRC_w_number_seed12345",
    ]



    # 稍微做个存在性检查（防止手滑）
    missing = [d for d in run_dirs if not os.path.isdir(d)]
    if missing:
        raise FileNotFoundError(
            "[FATAL] These run_dirs do not exist:\n  " + "\n  ".join(missing)
        )

    # 2) 逐 run_dir 读 CSV，生成明细表
    rows = []
    grouped: Dict[Tuple[str, str], List[Tuple[float, float, int]]] = {}  # (model, group) -> list(stats)

    for rd in run_dirs:
        model = infer_model_name_from_run_dir(rd)
        group = infer_group_from_run_dir(rd)
        if model is None or group is None:
            print(f"[WARN] Skip (cannot infer model/group): {rd}")
            continue

        csv_path = os.path.join(rd, args.csv_name)
        if not os.path.isfile(csv_path):
            print(f"[WARN] Missing CSV: {csv_path}")
            continue

        det, acc, n = summarize_prc_csv(csv_path)

        rows.append(
            {
                "model": model,
                "group": group,
                "detect_rate": det,
                "bit_acc": acc,  # V1：恒为 1.0
                "n": n,
                "run_dir": rd,
                "csv_path": csv_path,
            }
        )

        grouped.setdefault((model, group), []).append((det, acc, n))

    if not rows:
        raise RuntimeError("[FATAL] No valid CSVs were summarized (check paths & csv_name).")

    df_detail = (
        pd.DataFrame(rows)
        .sort_values(["model", "group", "run_dir"])
        .reset_index(drop=True)
    )

    # 3) 汇总表：按 (model, group) 加权平均（V1：仅 det 有意义）
    models = ["SD1.4", "SD1.5", "SD2.1"]
    groups = ["w", "w_att"]

    summary_rows = []
    for m in models:
        row = {"model": m}
        for g in groups:
            det, acc, n = weighted_merge(grouped.get((m, g), []))
            row[f"{g}_detect_rate"] = det
            row[f"{g}_bit_acc"] = acc  # V1：有数据则为 1.0
            row[f"{g}_n"] = n
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # 4) 分 sheet 输出（保持与 v2 一致）
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df_summary.to_excel(w, sheet_name="summary", index=False)
        df_detail.to_excel(w, sheet_name="detail_all", index=False)

        for g in groups:
            sub = df_detail[df_detail["group"] == g].copy()
            sub.to_excel(w, sheet_name=g, index=False)

    autosize_and_freeze(out_xlsx)
    print(f"[OK] wrote: {out_xlsx}")
    print(f"[OK] summarized {len(df_detail)} run_dirs")
    print(f"[OK] imgs_root={imgs_root}")


if __name__ == "__main__":
    main()
