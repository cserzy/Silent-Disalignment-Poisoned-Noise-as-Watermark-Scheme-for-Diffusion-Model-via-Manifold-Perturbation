# -*- coding: utf-8 -*-
"""
generate_GAUSS_zT_bank.py

What this script does:
  - Generate a zT bank of standard Gaussian noise: z ~ N(0, 1)
  - Save as a single .pt file containing a dict with:
      zT_bank: [M,4,H_lat,W_lat]
      shape, seeds, seed_base, gaussian_cfg, note, ...
  - Designed to be a "no-watermark / pure Gaussian baseline" compatible with your pipeline.

Example:
python generate_GAUSS_zT_bank.py \
  --outdir /home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment \
  --height 512 --width 512 \
  --bank_num 16 --seed 12345 \
  --dtype fp32
"""

from __future__ import annotations
import os
import argparse
from typing import List, Dict, Any

import torch


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _parse_dtype(s: str) -> torch.dtype:
    s = str(s).lower().strip()
    if s in ["fp32", "float32", "f32"]:
        return torch.float32
    if s in ["fp16", "float16", "f16"]:
        return torch.float16
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {s} (use fp32/fp16/bf16)")


def main():
    parser = argparse.ArgumentParser(description="Generate standard Gaussian zT bank and save in your pt dict format.")

    # IO
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--out_pt", type=str, default="", help="Output pt path. Default: <outdir>/generate_GAUSS_w.pt")

    # bank shape
    parser.add_argument("--bank_num", type=int, default=16, help="M, number of latents in bank")
    parser.add_argument("--height", type=int, default=512, help="Image height (for latent H=height//8)")
    parser.add_argument("--width", type=int, default=512, help="Image width (for latent W=width//8)")

    # randomness / dtype / device
    parser.add_argument("--seed", type=int, default=12345, help="seed_base, actual seed_i = seed + i")
    parser.add_argument("--dtype", type=str, default="fp32", help="fp32/fp16/bf16 (stored in this dtype)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda (generation device; saved to CPU)")

    # gaussian params (baseline usually mean=0 std=1)
    parser.add_argument("--mean", type=float, default=0.0)
    parser.add_argument("--std", type=float, default=1.0)

    args = parser.parse_args()

    _ensure_dir(args.outdir)

    H_lat = int(args.height) // 8
    W_lat = int(args.width) // 8
    M = int(args.bank_num)
    seed_base = int(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = _parse_dtype(args.dtype)

    z_list: List[torch.Tensor] = []
    seeds: List[int] = []

    # Per-sample generator for strict reproducibility across different PyTorch ops/order.
    for i in range(M):
        s = seed_base + i
        seeds.append(s)

        g = torch.Generator(device=device)
        g.manual_seed(int(s))

        # Standard Gaussian noise in latent space
        z = torch.randn((4, H_lat, W_lat), generator=g, device=device, dtype=torch.float32)
        if args.std != 1.0 or args.mean != 0.0:
            z = z * float(args.std) + float(args.mean)

        # store in requested dtype on CPU
        z_list.append(z.detach().to("cpu").to(dtype))

    z_bank = torch.stack(z_list, dim=0)  # [M,4,H_lat,W_lat]

    out_pt = args.out_pt.strip() if args.out_pt.strip() else os.path.join(args.outdir, "generate_GAUSS_w.pt")

    payload: Dict[str, Any] = {
        "zT_bank": z_bank,
        "shape": list(z_bank.shape),
        "seeds": seeds,
        "seed_base": seed_base,
        "gaussian_cfg": {"mean": float(args.mean), "std": float(args.std)},
        "note": "Pure Gaussian baseline: zT_bank ~ N(mean, std^2). Seeds are incremental (seed_base + i).",
    }
    torch.save(payload, out_pt)

    print("[DONE]")
    print(f"Saved: {out_pt}")
    print(f"zT_bank shape: {tuple(z_bank.shape)}")
    print(f"dtype: {z_bank.dtype}, mean={args.mean}, std={args.std}, seed_base={seed_base}")


if __name__ == "__main__":
    main()
