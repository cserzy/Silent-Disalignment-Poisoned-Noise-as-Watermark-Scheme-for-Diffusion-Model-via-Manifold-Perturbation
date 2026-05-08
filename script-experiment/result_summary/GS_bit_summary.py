#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize GS detection results across multiple runs into one Excel file.

Expected per-run CSV path:
  <run_dir>/detect/gs_detect_invert_decode.csv

The CSV should contain at least:
  - detected (0/1 or bool)
  - bit_acc (float)

Outputs:
  - An Excel file with:
      * sheet "summary": model x variant (w, w_att) with detect_rate & bit_accuracy
      * sheet "detail": concatenated per-image rows from all runs
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


DEFAULT_RUN_DIRS = [
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs/vis_sd14_GS_w_att_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs/vis_sd14_GS_w_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs/vis_sd15_GS_w_att_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs/vis_sd15_GS_w_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs/vis_sd21_GS_w_att_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs/vis_sd21_GS_w_seed12345",
]

DEFAULT_OUT_XLSX = "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/result_summary/GS_bit_accuracy.xlsx"


def parse_model_variant(run_dir: str) -> Tuple[str, str]:
    """
    Parse model + variant from run_dir name.

    Expected pattern examples:
      vis_sd14_GS_w_seed12345
      vis_sd21_GS_w_att_seed12345

    Returns:
      model: "SD1.4" | "SD1.5" | "SD2.1" | fallback raw token
      variant: "w" | "w_att" | fallback raw token
    """
    name = Path(run_dir).name
    m = re.search(r"vis_(sd14|sd15|sd21)_GS_(w(?:_att)?)_", name, flags=re.IGNORECASE)
    sd_map = {"sd14": "SD1.4", "sd15": "SD1.5", "sd21": "SD2.1"}
    if not m:
        # fallback: try to infer with weaker heuristics
        model = "UNKNOWN"
        if "sd14" in name.lower():
            model = "SD1.4"
        elif "sd15" in name.lower():
            model = "SD1.5"
        elif "sd21" in name.lower() or "sd2.1" in name.lower():
            model = "SD2.1"

        variant = "w_att" if "_w_att_" in name.lower() or "w_att" in name.lower() else ("w" if "_w_" in name.lower() or "gs_w" in name.lower() else "UNKNOWN")
        return model, variant

    model = sd_map.get(m.group(1).lower(), m.group(1))
    variant = m.group(2).lower()
    return model, variant


def coerce_detected(series: pd.Series) -> pd.Series:
    """
    Make detected column numeric 0/1.
    Accepts: bool, 0/1, "true"/"false", etc.
    """
    if series.dtype == bool:
        return series.astype(int)

    # If already numeric (int/float), map nonzero to 1
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0).astype(float) != 0).astype(int)

    # Otherwise treat as string
    s = series.astype(str).str.strip().str.lower()
    true_set = {"1", "true", "t", "yes", "y"}
    return s.isin(true_set).astype(int)


def summarize_one_run(csv_path: Path, run_dir: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = pd.read_csv(csv_path)  # pandas.read_csv citeturn0search0

    # Required cols
    for col in ("detected", "bit_acc"):
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in {csv_path}")

    df = df.copy()
    df["detected"] = coerce_detected(df["detected"])
    df["bit_acc"] = pd.to_numeric(df["bit_acc"], errors="coerce")

    # Basic metrics
    n = int(len(df))
    detect_rate = float(df["detected"].mean())  # mean over 0/1 gives proportion
    bit_accuracy = float(df["bit_acc"].mean(skipna=True))

    # Extra (optional) for debugging / deeper insight
    bit_acc_detected_only = float(df.loc[df["detected"] == 1, "bit_acc"].mean(skipna=True)) if (df["detected"] == 1).any() else float("nan")

    stats = dict(
        n_images=n,
        detect_rate=detect_rate,
        bit_accuracy=bit_accuracy,
        bit_acc_detected_only=bit_acc_detected_only,
    )

    # Normalize a minimal detail schema
    model, variant = parse_model_variant(run_dir)
    df["model"] = model
    df["variant"] = variant
    df["run_dir"] = run_dir

    # Keep a stable set of columns first, then keep the rest
    front = ["model", "variant", "run_dir"]
    if "image" in df.columns:
        front.append("image")
    elif "image_path" in df.columns:
        front.append("image_path")
    keep = front + [c for c in df.columns if c not in front]
    df = df[keep]

    return df, stats


def build_summary_table(stats_by_key: Dict[Tuple[str, str], Dict[str, float]]) -> pd.DataFrame:
    models = ["SD1.4", "SD1.5", "SD2.1"]
    variants = ["w", "w_att"]

    # MultiIndex columns: (variant, metric)
    cols = pd.MultiIndex.from_product([variants, ["detect_rate", "bit_accuracy"]])
    out = pd.DataFrame(index=models, columns=cols, dtype=float)

    for (model, variant), st in stats_by_key.items():
        if model not in out.index or variant not in variants:
            continue
        out.loc[model, (variant, "detect_rate")] = st["detect_rate"]
        out.loc[model, (variant, "bit_accuracy")] = st["bit_accuracy"]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dirs",
        nargs="*",
        default=DEFAULT_RUN_DIRS,
        help="List of run_dir paths. Each run_dir should contain detect/gs_detect_invert_decode.csv",
    )
    ap.add_argument(
        "--out_xlsx",
        default=DEFAULT_OUT_XLSX,
        help="Output Excel path (xlsx).",
    )
    ap.add_argument(
        "--csv_relpath",
        default="detect/gs_detect_invert_decode.csv",
        help="Relative path under each run_dir to the per-image CSV.",
    )
    ap.add_argument(
        "--sheet_summary",
        default="summary",
        help="Sheet name for the compact summary table.",
    )
    ap.add_argument(
        "--sheet_detail",
        default="detail",
        help="Sheet name for concatenated per-image rows.",
    )
    args = ap.parse_args()

    run_dirs: List[str] = list(args.run_dirs)
    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    stats_by_key: Dict[Tuple[str, str], Dict[str, float]] = {}
    detail_frames: List[pd.DataFrame] = []

    missing = []
    for rd in run_dirs:
        rd_p = Path(rd)
        csv_path = rd_p / args.csv_relpath
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue

        model, variant = parse_model_variant(str(rd_p))
        detail_df, st = summarize_one_run(csv_path, str(rd_p))
        stats_by_key[(model, variant)] = st
        detail_frames.append(detail_df)

    summary_df = build_summary_table(stats_by_key)

    detail_df_all = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()

    # Write Excel with two sheets (ExcelWriter recommended for multi-sheet) citeturn0search2turn0search1
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name=args.sheet_summary, merge_cells=False)
        detail_df_all.to_excel(writer, sheet_name=args.sheet_detail, index=False)

        # Add a small 'meta' sheet with missing files, if any
        meta_rows = []
        if missing:
            meta_rows.append({"key": "missing_csv_count", "value": len(missing)})
            for p in missing:
                meta_rows.append({"key": "missing_csv", "value": p})
        meta_rows.append({"key": "run_dirs_count", "value": len(run_dirs)})
        meta_rows.append({"key": "out_xlsx", "value": str(out_xlsx)})
        pd.DataFrame(meta_rows).to_excel(writer, sheet_name="meta", index=False)

    print(f"[OK] Wrote: {out_xlsx}")
    if missing:
        print(f"[WARN] Missing {len(missing)} CSV files. See 'meta' sheet for paths.")


if __name__ == "__main__":
    main()
