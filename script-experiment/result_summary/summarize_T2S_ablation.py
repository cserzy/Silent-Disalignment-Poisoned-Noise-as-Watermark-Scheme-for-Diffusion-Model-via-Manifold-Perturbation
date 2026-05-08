#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import argparse
from typing import List, Dict, Any

import pandas as pd


def parse_model_ablation(csv_path: str) -> Dict[str, str]:
    """
    支持两种来源：
      1) 文件名：sd14_T2S_w_att_delssc_detect.csv
      2) 文件夹名：.../vis_sd14_T2S_w_att_delssc_seed12345/...
    """
    base = os.path.basename(csv_path)
    m = re.search(r"(sd14|sd15|sd21).*?(delssc|delrepair)", base, flags=re.IGNORECASE)
    if m:
        return {"model": m.group(1).lower(), "ablation": m.group(2).lower()}

    folder = os.path.basename(os.path.dirname(csv_path))
    m = re.search(r"(sd14|sd15|sd21).*?(delssc|delrepair)", folder, flags=re.IGNORECASE)
    if m:
        return {"model": m.group(1).lower(), "ablation": m.group(2).lower()}

    raise ValueError(f"Cannot parse model/ablation from path: {csv_path}")


def load_one(csv_path: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)

    # Column compatibility: require `detected`; use `acc_msg` first, then `bit_acc`
    if "detected" not in df.columns:
        raise KeyError(f"{csv_path} missing column 'detected'. got={list(df.columns)}")

    bit_col = None
    for c in ["acc_msg", "bit_acc", "bit_accuracy", "acc"]:
        if c in df.columns:
            bit_col = c
            break
    if bit_col is None:
        raise KeyError(f"{csv_path} missing bit-acc column. expected one of acc_msg/bit_acc/... got={list(df.columns)}")

    df[["detected", bit_col]] = df[["detected", bit_col]].fillna(0)  # NaN 当 0
    det_rate = float(df["detected"].mean())
    bit_acc = float(df[bit_col].mean())
    n = int(len(df))

    info = parse_model_ablation(csv_path)
    return {
        "model": info["model"],
        "ablation": info["ablation"],
        "det_rate": det_rate,
        "bit_acc": bit_acc,
        "n": n,
        "csv_path": csv_path,
        "bit_col": bit_col,
    }


def build_tables(rows: List[Dict[str, Any]]):
    detail = pd.DataFrame(rows).sort_values(["model", "ablation"]).reset_index(drop=True)

    # 宽表：每个 model 一行，两个 ablation 各两列
    models = ["sd14", "sd15", "sd21"]
    ablas = ["delssc", "delrepair"]

    out = []
    for m in models:
        r = {"model": m}
        for a in ablas:
            sub = detail[(detail["model"] == m) & (detail["ablation"] == a)]
            if len(sub) == 0:
                r[f"{a}_det_rate"] = None
                r[f"{a}_bit_acc"] = None
                r[f"{a}_n"] = 0
            else:
                # 理论上每格只有 1 个 csv；如果多于 1 个，就取加权平均（按 n）
                if len(sub) == 1:
                    r[f"{a}_det_rate"] = float(sub.iloc[0]["det_rate"])
                    r[f"{a}_bit_acc"] = float(sub.iloc[0]["bit_acc"])
                    r[f"{a}_n"] = int(sub.iloc[0]["n"])
                else:
                    w = sub["n"].sum()
                    r[f"{a}_det_rate"] = float((sub["det_rate"] * sub["n"]).sum() / max(w, 1))
                    r[f"{a}_bit_acc"] = float((sub["bit_acc"] * sub["n"]).sum() / max(w, 1))
                    r[f"{a}_n"] = int(w)

        out.append(r)

    summary = pd.DataFrame(out)
    return summary, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        action="append",
        default=[],
        help="可重复传入：--csv path1 --csv path2 ...；不传则自动在 imgs 下搜索 *_detect.csv",
    )
    ap.add_argument(
        "--img_root",
        type=str,
        default="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs",
        help="自动发现 csv 的根目录（默认你的 imgs）",
    )
    ap.add_argument(
        "--out_xlsx",
        type=str,
        default="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/result_summary/T2S_bit_accuracy_ablation.xlsx",
    )
    args = ap.parse_args()

    csvs = list(args.csv)
    if len(csvs) == 0:
        pat = os.path.join(args.img_root, "vis_sd*_T2S_w_att_*_seed12345", "*_detect.csv")
        csvs = sorted(glob.glob(pat))
        if len(csvs) == 0:
            raise FileNotFoundError(f"No csv matched pattern: {pat}")

    rows = [load_one(p) for p in csvs]

    # Retain the 6 target runs only (sd14/sd15/sd21 × delssc/delrepair)
    keep_models = {"sd14", "sd15", "sd21"}
    keep_ablas = {"delssc", "delrepair"}
    rows = [r for r in rows if r["model"] in keep_models and r["ablation"] in keep_ablas]

    summary, detail = build_tables(rows)

    os.makedirs(os.path.dirname(args.out_xlsx), exist_ok=True)
    with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        detail.to_excel(w, sheet_name="detail", index=False)

    print("[OK] wrote:", args.out_xlsx)
    print("[INFO] used csvs:")
    for r in detail["csv_path"].tolist():
        print(" ", r)


if __name__ == "__main__":
    main()
