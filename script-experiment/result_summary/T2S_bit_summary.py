#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize T2S detection CSVs into one Excel.

- Only uses Image results:
    mode == "image_inversion"
  If "mode" column missing, falls back to rows where "image" is non-empty.

- Metrics:
    detect_rate = mean(detected)
    bit_accuracy = mean(acc_msg)

Outputs:
  <out_xlsx> with sheets:
    - summary: rows=model, cols=(w/w_att x metrics)
    - per_run: one row per run_dir with file path and stats

Run:
  python summarize_T2S_to_xlsx.py
or
  python summarize_T2S_to_xlsx.py --run_dirs ... --out_xlsx ...
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_RUN_DIRS = [
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs/vis_sd14_T2S_w_att_clip_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs/vis_sd15_T2S_w_att_clip_seed12345",
    "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/imgs/vis_sd21_T2S_w_att_clip_seed12345",
]

DEFAULT_OUT_XLSX = "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/result_summary/T2S_bit_accuracy.xlsx"
DEFAULT_PRIMARY_CSV = "t2s_detect_aligned.csv"


def parse_model_and_variant(run_dir: str) -> Tuple[str, str]:
    """
    Parse:
      sd14 -> SD1.4
      sd15 -> SD1.5
      sd21 -> SD2.1
      *_w_att_* -> w_att
      *_w_*     -> w
    """
    p = Path(run_dir).name.lower()

    if "sd14" in p:
        model = "SD1.4"
    elif "sd15" in p:
        model = "SD1.5"
    elif "sd21" in p:
        model = "SD2.1"
    else:
        model = "UNKNOWN"

    if "_w_att_" in p or "w_att" in p:
        variant = "w_att"
    elif "_w_" in p or p.endswith("_w_seed12345") or "t2s_w_" in p:
        variant = "w"
    else:
        # last resort
        variant = "w" if "w_att" not in p else "w_att"

    return model, variant


def find_csv_in_run_dir(run_dir: str, primary_name: str = DEFAULT_PRIMARY_CSV) -> Optional[str]:
    rd = Path(run_dir)
    if not rd.exists():
        return None

    # 1) primary file
    cand = rd / primary_name
    if cand.exists():
        return str(cand)

    # 2) any csv that contains "detect" or "t2s"
    csvs = sorted(rd.glob("*.csv"))
    if not csvs:
        # search recursively a bit
        csvs = sorted(rd.rglob("*.csv"))

    if not csvs:
        return None

    def score_path(x: Path) -> int:
        s = x.name.lower()
        score = 0
        if "t2s" in s:
            score += 3
        if "detect" in s:
            score += 3
        if "aligned" in s:
            score += 2
        if "dual" in s:
            score += 1
        return score

    csvs_sorted = sorted(csvs, key=lambda x: (score_path(x), x.name), reverse=True)
    return str(csvs_sorted[0])


def _coerce_numeric(series: pd.Series) -> pd.Series:
    # robust numeric conversion: strings -> float, invalid -> NaN
    return pd.to_numeric(series, errors="coerce")


def filter_image_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only Image results:
      - if 'mode' exists: mode == 'image_inversion'
      - else: if 'image' exists: non-empty image path
      - else: keep all (fallback)
    """
    d = df.copy()
    # Normalize column names
    d.columns = [c.strip() for c in d.columns]

    if "mode" in d.columns:
        return d[d["mode"].astype(str).str.strip().str.lower() == "image_inversion"].copy()

    if "image" in d.columns:
        # keep non-empty
        img = d["image"].astype(str).str.strip()
        return d[(img != "") & (img.str.lower() != "none")].copy()

    return d


def compute_metrics(df_img: pd.DataFrame) -> Dict[str, float]:
    """
    detect_rate = mean(detected)
    bit_accuracy = mean(acc_msg)
    """
    out: Dict[str, float] = {"n": 0, "detect_rate": float("nan"), "bit_accuracy": float("nan")}

    if df_img is None or len(df_img) == 0:
        return out

    out["n"] = int(len(df_img))

    if "detected" in df_img.columns:
        det = _coerce_numeric(df_img["detected"])
        out["detect_rate"] = float(det.mean())
    else:
        out["detect_rate"] = float("nan")

    if "acc_msg" in df_img.columns:
        acc = _coerce_numeric(df_img["acc_msg"])
        out["bit_accuracy"] = float(acc.mean())
    else:
        # fallback to oracle if needed
        if "acc_msg_oracle" in df_img.columns:
            acc = _coerce_numeric(df_img["acc_msg_oracle"])
            out["bit_accuracy"] = float(acc.mean())
        else:
            out["bit_accuracy"] = float("nan")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="*", default=DEFAULT_RUN_DIRS, help="List of run directories.")
    ap.add_argument("--out_xlsx", type=str, default=DEFAULT_OUT_XLSX, help="Output Excel path.")
    ap.add_argument("--primary_csv", type=str, default=DEFAULT_PRIMARY_CSV, help="Preferred CSV filename in each run_dir.")
    ap.add_argument("--verbose", action="store_true", help="Print per-run details.")
    args = ap.parse_args()

    per_run_rows: List[Dict[str, object]] = []

    for rd in args.run_dirs:
        model, variant = parse_model_and_variant(rd)
        csv_path = find_csv_in_run_dir(rd, primary_name=args.primary_csv)

        row: Dict[str, object] = {
            "run_dir": rd,
            "model": model,
            "variant": variant,
            "csv_path": csv_path if csv_path else "",
            "n_image": 0,
            "detect_rate": np.nan,
            "bit_accuracy": np.nan,
            "status": "OK",
        }

        if not csv_path:
            row["status"] = "CSV_NOT_FOUND"
            per_run_rows.append(row)
            if args.verbose:
                print(f"[WARN] csv not found in: {rd}")
            continue

        try:
            df = pd.read_csv(csv_path)
            df_img = filter_image_rows(df)
            met = compute_metrics(df_img)
            row["n_image"] = met["n"]
            row["detect_rate"] = met["detect_rate"]
            row["bit_accuracy"] = met["bit_accuracy"]

            if args.verbose:
                print(f"[OK] {model} {variant} n={row['n_image']} "
                      f"det_rate={row['detect_rate']:.4f} bit_acc={row['bit_accuracy']:.4f} :: {csv_path}")

        except Exception as e:
            row["status"] = f"ERROR: {repr(e)}"
            if args.verbose:
                print(f"[ERR] failed reading {csv_path}: {e}")

        per_run_rows.append(row)

    df_per = pd.DataFrame(per_run_rows)

    # Build summary table: rows=model, columns=(w/w_att)*(detect_rate, bit_accuracy)
    models_order = ["SD1.4", "SD1.5", "SD2.1"]
    variants_order = ["w", "w_att"]

    summary_rows: List[Dict[str, object]] = []
    for m in models_order:
        r: Dict[str, object] = {"model": m}
        for v in variants_order:
            sub = df_per[(df_per["model"] == m) & (df_per["variant"] == v)]
            # If multiple matches, take the first non-NaN by preference (rare but safe)
            det = sub["detect_rate"].dropna()
            acc = sub["bit_accuracy"].dropna()
            nimg = sub["n_image"].dropna()
            r[f"{v}_detect_rate"] = float(det.iloc[0]) if len(det) else np.nan
            r[f"{v}_bit_accuracy"] = float(acc.iloc[0]) if len(acc) else np.nan
            r[f"{v}_n_image"] = int(nimg.iloc[0]) if len(nimg) else 0
        summary_rows.append(r)

    df_sum = pd.DataFrame(summary_rows)

    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    # Write Excel (two sheets)
    with pd.ExcelWriter(out_xlsx) as writer:
        df_sum.to_excel(writer, sheet_name="summary", index=False)
        df_per.to_excel(writer, sheet_name="per_run", index=False)

    print(f"[OK] wrote: {out_xlsx}")
    print("Sheets: summary, per_run")
    # Quick console view
    print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
