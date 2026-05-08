#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge NSFW threshold sweep reports across:
  - models: SD1.4 / SD1.5 / SD2.1
  - methods: TR / GS / PRC / T2S
  - status: w / w_att
Output: one xlsx with 7 sheets (thr_0.2 ... thr_0.8)
Each sheet: rows = Model x Method, cols = w, w_att, Diff(w_att - w)

Usage:
  python merge_nsfw_threshold_sweep_20260121.py \
    --out_xlsx /home/yancy/work/dm_backdoor_latent_space/experiment-1_19/result_summary/NSFW_threshold_sweep.xlsx

Optional:
  python merge_nsfw_threshold_sweep_20260121.py --run_dirs_txt runs.txt --out_xlsx out.xlsx
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Default run_dirs (your list)
# -----------------------------
DEFAULT_RUN_DIRS = [
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_GS_w_att_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_GS_w_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_PRC_w_att_0_85_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_PRC_w_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_T2S_w_att_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_T2S_w_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_TR_w_att_0_88_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd14_TR_w_dongman_seed12345",

    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_GS_w_att_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_GS_w_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_PRC_w_att_0_85_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_PRC_w_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_T2S_w_att_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_T2S_w_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_TR_w_att_0_88_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd15_TR_w_dongman_seed12345",

    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_GS_w_att_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_GS_w_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_PRC_w_att_0_85_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_PRC_w_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_T2S_w_att_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_T2S_w_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_TR_w_att_0_88_dongman_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs-number/vis_sd21_TR_w_dongman_seed12345",
]

MODEL_MAP = {
    "sd14": "SD1.4",
    "sd15": "SD1.5",
    "sd21": "SD2.1",
}
METHODS = ["TR", "GS", "PRC", "T2S"]
STATUSES = ["w", "w_att"]

THRESHOLDS = [round(x, 1) for x in np.arange(0.2, 0.9, 0.1)]  # 0.2..0.8


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def parse_run_dir(run_dir: str) -> Tuple[str, str, str]:
    """
    Return (model_name, method, status) from run_dir basename.
    """
    base = os.path.basename(run_dir.rstrip("/"))

    # model token
    m = re.search(r"vis_(sd\d+)_", base)
    if not m:
        raise ValueError(f"Cannot parse model token from run_dir basename: {base}")
    model_token = m.group(1).lower()
    if model_token not in MODEL_MAP:
        raise ValueError(f"Unknown model token '{model_token}' in: {base}")
    model_name = MODEL_MAP[model_token]

    # method
    method = None
    for cand in METHODS:
        if f"_{cand}_" in f"_{base}_":
            method = cand
            break
    if method is None:
        # fallback: substring search
        for cand in METHODS:
            if cand in base:
                method = cand
                break
    if method is None:
        raise ValueError(f"Cannot parse method (TR/GS/PRC/T2S) from: {base}")

    # status: detect w_att first
    if "_w_att" in base:
        status = "w_att"
    elif re.search(r"_w(_|$)", base) is not None:
        status = "w"
    else:
        # PRC sometimes: ..._w_0_85...
        if "_w_" in base:
            status = "w"
        else:
            raise ValueError(f"Cannot parse status (w vs w_att) from: {base}")

    return model_name, method, status


def read_threshold_sweep(report_xlsx: str) -> Dict[float, float]:
    """
    Read ThresholdSweep sheet and return dict: threshold(float)->nsfw_rate(float).
    If multiple rows per threshold, take mean.
    """
    df = pd.read_excel(report_xlsx, sheet_name="ThresholdSweep", engine="openpyxl")
    df = _norm_cols(df)

    thr_col = _find_col(df, ["threshold", "thr", "nsfw_threshold", "tau"])
    rate_col = _find_col(df, ["nsfw_rate", "nsfw_ratio", "rate", "ratio", "nsfw"])

    if thr_col is None or rate_col is None:
        raise ValueError(
            f"[ThresholdSweep] missing required columns in {report_xlsx}\n"
            f"  columns={list(df.columns)}\n"
            f"  need one of threshold cols: threshold/thr/nsfw_threshold/tau\n"
            f"  need one of rate cols: nsfw_rate/nsfw_ratio/rate/ratio/nsfw"
        )

    # Optional: filter out NoWM if exists (you said none, but keep safe)
    label_col = _find_col(df, ["label", "wm_label", "name"])
    if label_col is not None:
        mask_nowm = df[label_col].astype(str).str.lower().str.contains("nowm")
        if mask_nowm.any():
            df = df.loc[~mask_nowm].copy()

    df = df[[thr_col, rate_col]].copy()
    df[thr_col] = pd.to_numeric(df[thr_col], errors="coerce")
    df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")
    df = df.dropna(subset=[thr_col, rate_col])

    # Normalize threshold to 1 decimal (avoid float noise)
    df["_thr_round"] = df[thr_col].round(1)
    g = df.groupby("_thr_round")[rate_col].mean()

    return {float(k): float(v) for k, v in g.items()}


def load_run_dirs(run_dirs_txt: Optional[str]) -> List[str]:
    if run_dirs_txt is None:
        return DEFAULT_RUN_DIRS
    with open(run_dirs_txt, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs_txt", type=str, default=None,
                    help="Optional text file, each line is a run_dir (override default list).")
    ap.add_argument("--out_xlsx", type=str, required=True,
                    help="Output xlsx path, e.g. .../result_summary/NSFW_threshold_sweep.xlsx")
    args = ap.parse_args()

    run_dirs = load_run_dirs(args.run_dirs_txt)

    # key: (model_name, method, status) -> {thr: rate}
    results: Dict[Tuple[str, str, str], Dict[float, float]] = {}
    missing_reports = []

    for rd in run_dirs:
        model_name, method, status = parse_run_dir(rd)
        report_xlsx = os.path.join(rd, "nsfw_report", "report.xlsx")
        if not os.path.exists(report_xlsx):
            missing_reports.append(report_xlsx)
            continue
        try:
            thr2rate = read_threshold_sweep(report_xlsx)
        except Exception as e:
            print(f"[ERROR] failed to read ThresholdSweep: {report_xlsx}\n  -> {e}")
            continue

        key = (model_name, method, status)
        results[key] = thr2rate
        print(f"[OK] {key}  loaded {len(thr2rate)} thresholds from {report_xlsx}")

    if missing_reports:
        print("\n[WARN] Missing report.xlsx files:")
        for p in missing_reports:
            print("  -", p)
        print("")

    os.makedirs(os.path.dirname(args.out_xlsx), exist_ok=True)

    # build sheets
    with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as writer:
        for thr in THRESHOLDS:
            rows = []
            for model_name in ["SD1.4", "SD1.5", "SD2.1"]:
                for method in METHODS:
                    w_key = (model_name, method, "w")
                    a_key = (model_name, method, "w_att")
                    w_val = results.get(w_key, {}).get(thr, np.nan)
                    a_val = results.get(a_key, {}).get(thr, np.nan)
                    diff = (a_val - w_val) if (np.isfinite(w_val) and np.isfinite(a_val)) else np.nan
                    rows.append({
                        "Model": model_name,
                        "Method": method,
                        "w": w_val,
                        "w_att": a_val,
                        "Diff(w_att-w)": diff,
                    })

            df_out = pd.DataFrame(rows)

            # optional: keep nice ordering
            df_out["Method"] = pd.Categorical(df_out["Method"], categories=METHODS, ordered=True)
            df_out["Model"] = pd.Categorical(df_out["Model"], categories=["SD1.4", "SD1.5", "SD2.1"], ordered=True)
            df_out = df_out.sort_values(["Model", "Method"], kind="stable").reset_index(drop=True)

            sheet = f"thr_{thr:.1f}"
            df_out.to_excel(writer, sheet_name=sheet, index=False)

        # add a meta sheet for traceability
        meta_rows = []
        for k, v in sorted(results.items()):
            model_name, method, status = k
            meta_rows.append({
                "Model": model_name,
                "Method": method,
                "Status": status,
                "NumThresholds": len(v),
            })
        df_meta = pd.DataFrame(meta_rows)
        df_meta.to_excel(writer, sheet_name="META", index=False)

    # quick missing-combo summary
    print("\n[SUMMARY] Missing combinations (no data for any threshold):")
    for model_name in ["SD1.4", "SD1.5", "SD2.1"]:
        for method in METHODS:
            for status in STATUSES:
                if (model_name, method, status) not in results:
                    print(f"  - {(model_name, method, status)}")

    print(f"\n[DONE] wrote: {args.out_xlsx}")


if __name__ == "__main__":
    main()
