#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_save_dtype(dtype_str: str) -> torch.dtype:
    s = str(dtype_str).lower().strip()
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    if s == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported --dtype: {dtype_str}. Use one of: fp16, bf16, fp32.")


def dtype_name(dt: torch.dtype) -> str:
    if dt == torch.float16:
        return "fp16"
    if dt == torch.bfloat16:
        return "bf16"
    if dt == torch.float32:
        return "fp32"
    return str(dt)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GAUSS zT bank aligned to vis_generate-slice_multi-12.05.py No-WM raw Gaussian logic."
    )

    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--out_pt",
        type=str,
        default="",
        help="Output .pt path. Default: <outdir>/generate_GAUSS_w_aligned_vis.pt",
    )
    parser.add_argument("--bank_num", type=int, default=16)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="cpu", help="Sampling device. Default cpu.")
    parser.add_argument("--project_root", type=str, default="", help="Optional. For logging/path bookkeeping only.")
    parser.add_argument("--out_meta_pt", type=str, default="", help="Optional meta .pt output path.")
    parser.add_argument("--out_meta_json", type=str, default="", help="Optional meta .json output path.")
    parser.add_argument(
        "--save_alias_zt_bank",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to also save alias key 'zT_bank'. Default 1.",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.out_pt.strip():
        out_pt = Path(args.out_pt)
    else:
        out_pt = outdir / "generate_GAUSS_w_aligned_vis.pt"
    if out_pt.parent:
        out_pt.parent.mkdir(parents=True, exist_ok=True)

    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError(f"height and width must be divisible by 8, got height={args.height}, width={args.width}")
    if args.bank_num <= 0:
        raise ValueError(f"bank_num must be > 0, got {args.bank_num}")

    req_device = str(args.device).strip().lower()
    if req_device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        generation_device = torch.device("cpu")
    else:
        generation_device = torch.device(req_device)

    save_dtype = parse_save_dtype(args.dtype)

    # Align with vis script style: set global seed once, then sample.
    set_seed(int(args.seed))

    latent_shape = (
        int(args.bank_num),
        4,
        int(args.height) // 8,
        int(args.width) // 8,
    )

    # Core alignment: one-shot torch.randn over full bank, sampled in float32.
    latents = torch.randn(latent_shape, device=generation_device, dtype=torch.float32)

    # Save to CPU, then cast to requested save dtype (default fp32).
    latents_to_save = latents.detach().to("cpu").to(save_dtype)

    meta = {
        "method": "GAUSS_aligned_to_vis_nowm",
        "seed": int(args.seed),
        "bank_num": int(args.bank_num),
        "latent_shape": list(latent_shape),
        "generation_device": str(generation_device),
        "sample_dtype": "float32",
        "save_dtype": dtype_name(save_dtype),
        "project_root": str(args.project_root) if args.project_root else "",
        "note": "Aligned to vis_generate-slice_multi-12.05.py No-WM raw Gaussian before scheduler.init_noise_sigma.",
    }

    payload = {
        "latents": latents_to_save,
        "meta": meta,
    }
    if int(args.save_alias_zt_bank) == 1:
        payload["zT_bank"] = latents_to_save

    torch.save(payload, out_pt)

    if args.out_meta_pt.strip():
        out_meta_pt = Path(args.out_meta_pt)
        if out_meta_pt.parent:
            out_meta_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(meta, out_meta_pt)

    if args.out_meta_json.strip():
        out_meta_json = Path(args.out_meta_json)
        if out_meta_json.parent:
            out_meta_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_meta_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[DONE] GAUSS bank generated.")
    print(f"Saved pt: {out_pt}")
    print(f"latents shape: {tuple(latents_to_save.shape)}")
    print(f"sample_dtype: float32, save_dtype: {dtype_name(save_dtype)}")
    print(f"generation_device: {generation_device}")


if __name__ == "__main__":
    main()
