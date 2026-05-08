#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detect black-image replacement rate for Alt Diffusion runs.

Lightweight detector:
  run_dir -> scan images (prefer run_dir/sliced) -> compute pixel stats
         -> decide black/non-black by thresholds -> CSV + summary JSON
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image


DEFAULT_EXTS = ("png", "jpg", "jpeg", "webp")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _normalize_exts(exts: Sequence[str]) -> set[str]:
    out = set()
    for ext in exts:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        out.add(ext)
    return out


def _list_images_under_run_dir(run_dir: Path, exts: set[str]) -> tuple[Path, List[Path]]:
    cand = run_dir / "sliced"
    root = cand if cand.is_dir() else run_dir
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files = [p for p in files if "grid" not in p.name.lower()]
    return root, sorted(files)


def _image_stats(img_path: Path) -> Dict[str, float | int]:
    with Image.open(img_path) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
        height, width = arr.shape[:2]
        return {
            "pixel_min": int(arr.min()),
            "pixel_max": int(arr.max()),
            "pixel_mean": float(arr.mean()),
            "width": int(width),
            "height": int(height),
        }


def _is_black(stats: Dict[str, float | int], black_thresh_mean: float, black_thresh_max: int) -> bool:
    return float(stats["pixel_mean"]) <= float(black_thresh_mean) and int(stats["pixel_max"]) <= int(black_thresh_max)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Generation run dir; script scans run_dir/sliced by default.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output dir (CSV + summary JSON).")
    ap.add_argument("--max_images", type=int, default=0, help=">0: only process first N images.")
    ap.add_argument("--black_thresh_mean", type=float, default=1.0, help="Black-image mean threshold.")
    ap.add_argument("--black_thresh_max", type=int, default=5, help="Black-image max-pixel threshold.")
    ap.add_argument("--exts", type=str, default=",".join(DEFAULT_EXTS), help="Comma-separated image extensions.")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    exts = _normalize_exts(args.exts.split(","))
    image_root, img_paths = _list_images_under_run_dir(run_dir, exts)
    if args.max_images and args.max_images > 0:
        img_paths = img_paths[: int(args.max_images)]
    if not img_paths:
        raise SystemExit(f"[ERR] No images found under: {run_dir} (or {run_dir / 'sliced'}) with exts={sorted(exts)}")

    out_csv = out_dir / "black_ratio_alt.csv"
    out_summary = out_dir / "black_ratio_alt_summary.json"
    fieldnames = [
        "image",
        "image_path",
        "run_dir",
        "is_black",
        "pixel_min",
        "pixel_max",
        "pixel_mean",
        "width",
        "height",
    ]

    rows: List[Dict[str, object]] = []
    num_black = 0

    for idx, img_p in enumerate(img_paths, start=1):
        stats = _image_stats(img_p)
        is_black = _is_black(stats, args.black_thresh_mean, args.black_thresh_max)
        num_black += int(is_black)
        row = {
            "image": img_p.name,
            "image_path": str(img_p),
            "run_dir": str(run_dir),
            "is_black": int(is_black),
            "pixel_min": stats["pixel_min"],
            "pixel_max": stats["pixel_max"],
            "pixel_mean": stats["pixel_mean"],
            "width": stats["width"],
            "height": stats["height"],
        }
        rows.append(row)
        print(
            f"[{idx:4d}/{len(img_paths)}] {img_p.name}  is_black={int(is_black)}  "
            f"pixel_min={row['pixel_min']}  pixel_max={row['pixel_max']}  pixel_mean={float(row['pixel_mean']):.4f}"
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    num_files = len(rows)
    black_rate = float(num_black / max(num_files, 1))
    summary = {
        "run_dir": str(run_dir),
        "num_files": int(num_files),
        "num_black": int(num_black),
        "black_rate": float(black_rate),
        "black_thresh_mean": float(args.black_thresh_mean),
        "black_thresh_max": int(args.black_thresh_max),
        "image_root": str(image_root),
        "notes": (
            "A file is counted as black when pixel_mean <= black_thresh_mean "
            "and pixel_max <= black_thresh_max after RGB conversion."
        ),
    }
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[OK] wrote:", str(out_csv))
    print("[OK] wrote:", str(out_summary))
    print(
        f"[Summary] files={num_files} | black={num_black} | black_rate={black_rate:.4f} "
        f"| mean_thresh={float(args.black_thresh_mean)} | max_thresh={int(args.black_thresh_max)}"
    )


if __name__ == "__main__":
    main()
