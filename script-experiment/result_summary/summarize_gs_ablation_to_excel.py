#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize GS ablation detection results (delssc / delrepair) across 6 run_dirs into one Excel.

Input CSV (per run_dir):
  <run_dir>/detect/gs_detect_invert_decode.csv

Required columns:
  - detected  (0/1 or bool; NaN treated as 0)
  - bit_acc   (float; NaN treated as 0)

Output Excel:
  /home/yancy/work/dm_backdoor_latent_space/experiment-1_19/result_summary/GS_bit_accuracy_ablation.xlsx

Sheets:
  - summary: compact table by model x ablation with detect_rate & bit_accuracy
  - detail : concatenated per-image rows from all runs
  - meta   : missing-file report + sample counts
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# ---- defaults (your 6 run_dirs) ----
DEFAULT_RUN_DIRS = [
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs/vis_sd14_GS_w_att_delssc_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs/vis_sd15_GS_w_att_delssc_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs/vis_sd21_GS_w_att_delssc_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs/vis_sd14_GS_w_att_delrepair_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs/vis_sd15_GS_w_att_delrepair_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs/vis_sd21_GS_w_att_delrepair_seed12345",
]

DEFAULT_OUT_XLSX = "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/result_summary/GS_bit_accuracy_ablation.xlsx"
DEFAULT_CSV_RELPATH = "detect/gs_detect_invert_decode.csv"


def parse_model_and_ablation(run_dir: str) -> Tuple[str, str]:
    """
    Parse model + ablation from run_dir basename.

    Examples:
      vis_sd14_GS_w_att_delssc_seed12345    -> (SD1.4, delssc)
      vis_sd21_GS_w_att_delrepair_seed12345 -> (SD2.1, delrepair)
    """
    name = Path(run_dir).name.lower()

    # model
    if "sd14" in name:
        model = "SD1.4"
    elif "sd15" in name:
        model = "SD1.5"
    elif "sd21" in name or "sd2.1" in name:
        model = "SD2.1"
    else:
        model = "UNKNOWN"

    # ablation
    if "delssc" in name:
        abla = "delssc"
    elif "delrepair" in name:
        abla = "delrepair"
    else:
        abla = "UNKNOWN"

    return model, abla


def coerce_detected_to_01(s: pd.Series) -> pd.Series:
    """
    Convert detected column to numeric 0/1.
    NaN treated as 0 (strict).
    """
    if s.dtype == bool:
        return s.fillna(False).astype(int)

    if pd.api.types.is_numeric_dtype(s):
        return (pd.to_numeric(s, errors="coerce").fillna(0) != 0).astype(int)

    # string-like
    ss = s.astype(str).str.strip().str.lower()
    true_set = {"1", "true", "t", "yes", "y"}
    out = ss.isin(true_set).astype(int)
    # NaN-like strings become False -> 0
    return out


def read_one_csv(csv_path: Path) -> pd.DataFrame:
    # pandas.read_csv documentation :contentReference[oaicite:3]{index=3}
    df = pd.read_csv(csv_path)

    if "detected" not in df.columns or "bit_acc" not in df.columns:
        raise KeyError(f"CSV missing required cols detected/bit_acc: {csv_path}")

    df = df.copy()
    df["detected"] = coerce_detected_to_01(df["detected"])

    # bit_acc: numeric; NaN treated as 0 (strict)
    df["bit_acc"] = pd.to_numeric(df["bit_acc"], errors="coerce").fillna(0.0)

    return df


def compute_stats(df: pd.DataFrame) -> Dict[str, float]:
    n = int(len(df))
    detect_rate = float(df["detected"].mean())  # bool/0-1 mean gives proportion :contentReference[oaicite:4]{index=4}
    bit_accuracy = float(df["bit_acc"].mean())  # Series.mean skipna; we already filled NaN->0 :contentReference[oaicite:5]{index=5}
    return {"n_images": n, "detect_rate": detect_rate, "bit_accuracy": bit_accuracy}


def build_summary(stats: Dict[Tuple[str, str], Dict[str, float]]) -> pd.DataFrame:
    models = ["SD1.4", "SD1.5", "SD2.1"]
    ablations = ["delssc", "delrepair"]
    metrics = ["detect_rate", "bit_accuracy"]

    cols = pd.MultiIndex.from_product([ablations, metrics], names=["ablation", "metric"])
    out = pd.DataFrame(index=models, columns=cols, dtype=float)

    for (model, abla), st in stats.items():
        if model not in out.index or abla not in ablations:
            continue
        out.loc[model, (abla, "detect_rate")] = st["detect_rate"]
        out.loc[model, (abla, "bit_accuracy")] = st["bit_accuracy"]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="*", default=DEFAULT_RUN_DIRS, help="6 run_dir paths")
    ap.add_argument("--out_xlsx", default=DEFAULT_OUT_XLSX, help="output xlsx path")
    ap.add_argument("--csv_relpath", default=DEFAULT_CSV_RELPATH, help="relative csv path under run_dir")
    args = ap.parse_args()

    run_dirs: List[str] = list(args.run_dirs)
    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    all_detail: List[pd.DataFrame] = []
    stats_by_key: Dict[Tuple[str, str], Dict[str, float]] = {}
    missing: List[str] = []

    for rd in run_dirs:
        csv_path = Path(rd) / args.csv_relpath
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue

        model, abla = parse_model_and_ablation(rd)
        df = read_one_csv(csv_path)
        st = compute_stats(df)
        stats_by_key[(model, abla)] = st

        # attach identifiers for detail sheet
        df["model"] = model
        df["ablation"] = abla
        df["run_dir"] = rd
        # keep some common columns in front
        front = ["model", "ablation", "run_dir"]
        if "image" in df.columns:
            front.append("image")
        if "image_path" in df.columns:
            front.append("image_path")
        rest = [c for c in df.columns if c not in front]
        df = df[front + rest]
        all_detail.append(df)

    summary_df = build_summary(stats_by_key)
    detail_df = pd.concat(all_detail, ignore_index=True) if all_detail else pd.DataFrame()

    # Write multi-sheet Excel via ExcelWriter :contentReference[oaicite:6]{index=6}
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", merge_cells=False)
        detail_df.to_excel(writer, sheet_name="detail", index=False)

        meta_rows = []
        meta_rows.append({"key": "out_xlsx", "value": str(out_xlsx)})
        meta_rows.append({"key": "run_dirs_count", "value": len(run_dirs)})
        meta_rows.append({"key": "missing_csv_count", "value": len(missing)})

        # sample counts per group
        for (model, abla), st in sorted(stats_by_key.items()):
            meta_rows.append({"key": f"n_images::{model}::{abla}", "value": int(st["n_images"])})

        for p in missing:
            meta_rows.append({"key": "missing_csv", "value": p})

        pd.DataFrame(meta_rows).to_excel(writer, sheet_name="meta", index=False)

    print(f"[OK] wrote: {out_xlsx}")
    if missing:
        print(f"[WARN] missing {len(missing)} csv files; see meta sheet.")


if __name__ == "__main__":
    main()
