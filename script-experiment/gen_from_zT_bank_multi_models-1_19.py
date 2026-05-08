#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate images from a fixed z_T bank tensor: [16,4,64,64] for a SINGLE model.

Experiment Setup:
- One model per run (SD1.4 / SD1.5 / SD2.1)
- Load z_T ONCE before the prompt loop
- Each prompt generates 4 images (default uses z_T[0:4])
- Save images directly to out_dir/sliced as Pxx_yy.png (no grid, no slicing)
- Write ONLY one manifest.csv in out_dir/sliced (no global manifest, no subfolders)

Important about scaling:
- By default, assumes z_T is standard normal space, so we scale by scheduler.init_noise_sigma
  (diffusers pipelines do this for initial noise). You can disable via --latents_already_scaled.
"""

import re
import csv
import json
import time
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


# -------------------------
# utils
# -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_prompts(txt_path: str) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]

def parse_method_and_tag(zT_path: str) -> Tuple[str, str]:
    """
    Expect: generate_{METHOD}_{TAG}.pt
      METHOD: TR/GS/PRC/T2S (case-insensitive)
      TAG: w / w_att
    """
    name = Path(zT_path).name
    m = re.match(r"generate_([A-Za-z0-9]+)_([A-Za-z0-9]+)\.pt$", name)
    if not m:
        return "UNK", "UNK"
    method = m.group(1).upper()
    tag = m.group(2).lower()
    if method not in {"TR", "GS", "PRC", "T2S"}:
        method = "UNK"
    if tag not in {"w", "w_att"}:
        tag = "UNK"
    return method, tag

def load_zT_bank(pt_path: str) -> torch.Tensor:
    """
    Load z_T bank tensor, shape [16,4,64,64].
    Supports:
      - raw tensor
      - dict checkpoint (tries common keys)
    """
    obj = torch.load(pt_path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        zT = obj
    elif isinstance(obj, dict):
        for k in ["zT_bank","zT", "z_t", "z_T", "latents", "noise", "z"]:
            if k in obj and isinstance(obj[k], torch.Tensor):
                zT = obj[k]
                break
        else:
            raise KeyError(f"[zT] dict pt but no tensor found. keys={list(obj.keys())}")
    else:
        raise TypeError(f"[zT] Unsupported pt content type: {type(obj)}")

    if zT.ndim != 4:
        raise ValueError(f"[zT] Expect 4D tensor, got {tuple(zT.shape)}")
    if zT.shape[1:] != (4, 64, 64):
        raise ValueError(f"[zT] Expect [*,4,64,64], got {tuple(zT.shape)}")
    if zT.shape[0] < 4:
        raise ValueError(f"[zT] Need at least 4 latents, got N={zT.shape[0]}")

    return zT.float()


def build_pipe(model_id: str, device: str, dtype: torch.dtype) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,   # speed (you will do NSFW scoring separately)
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    return pipe


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_id", type=str, required=True,
                    help="One model path or HF repo id (SD1.4/1.5/2.1 diffusers folder).")
    ap.add_argument("--prompts", type=str, required=True, help="txt: one prompt per line")
    ap.add_argument("--zT_pt", type=str, required=True, help="pt: z_T bank [16,4,64,64]")
    ap.add_argument("--outdir", type=str, required=True, help="output dir (images + manifest in outdir/sliced/)")

    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)

    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])

    # fixed: 4 images per prompt (but keep as hyperparam)
    ap.add_argument("--n_per_prompt", type=int, default=4, help="images per prompt (default 4)")
    ap.add_argument("--start_latent", type=int, default=0, help="start index in z_T bank (default 0)")

    # scaling switch
    ap.add_argument("--latents_already_scaled", action="store_true",
                    help="If set, do NOT multiply latents by scheduler.init_noise_sigma (default: multiply).")

    # negative prompt
    ap.add_argument("--negative_prompt", type=str, default="",
                    help="negative prompt string (default empty)")

    args = ap.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.outdir)
    sliced_dir = out_dir / "sliced"
    ensure_dir(sliced_dir)

    prompts = load_prompts(args.prompts)
    if len(prompts) == 0:
        raise ValueError("No prompts loaded.")

    # load z_T ONCE
    zT_bank = load_zT_bank(args.zT_pt)  # CPU [N,4,64,64]
    N = zT_bank.shape[0]
    B = args.n_per_prompt
    start = args.start_latent
    if start < 0 or start >= N:
        raise ValueError(f"--start_latent out of range: {start} (N={N})")
    if start + B > N:
        raise ValueError(f"Need {B} latents from {start}, but N={N}")

    method, tag = parse_method_and_tag(args.zT_pt)

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    pipe = build_pipe(args.model_id, device=args.device, dtype=torch_dtype)

    # select working latents (size=B)
    zT_batch = zT_bank[start:start+B].clone()  # CPU
    sigma = None
    if not args.latents_already_scaled:
        sigma = float(pipe.scheduler.init_noise_sigma)
        zT_batch = zT_batch * sigma  # diffusers standard: scale initial noise by init_noise_sigma

    # move once
    zT_batch = zT_batch.to(device=args.device, dtype=pipe.unet.dtype)

    manifest_path = sliced_dir / "manifest.csv"
    rows: List[Dict[str, Any]] = []

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "file",
            "prompt_idx",
            "img_idx",
            "prompt",
            "zT_idx",
            "wm_method",
            "wm_tag",
            "model_id",
            "steps",
            "cfg",
            "height",
            "width",
            "seed",
            "dtype",
            "scheduler",
            "latents_scaled_sigma",
            "negative_prompt",
            "zT_pt",
            "sec_per_prompt",
        ])

        for pidx, prompt in enumerate(prompts):
            batch_prompts = [prompt] * B
            negs = [args.negative_prompt] * B  # batch negative prompts

            t0 = time.time()
            images = pipe(
                prompt=batch_prompts,
                negative_prompt=negs,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                height=args.height,
                width=args.width,
                latents=zT_batch,
            ).images
            dt = time.time() - t0

            for j, im in enumerate(images):
                out_name = f"P{pidx:02d}_{j:02d}.png"
                out_path = sliced_dir / out_name
                im.save(out_path)

                row = {
                    "file": str(out_path),
                    "prompt_idx": pidx,
                    "img_idx": j,
                    "prompt": prompt,
                    "zT_idx": start + j,
                    "wm_method": method,
                    "wm_tag": tag,
                    "model_id": args.model_id,
                    "steps": args.steps,
                    "cfg": args.cfg,
                    "height": args.height,
                    "width": args.width,
                    "seed": args.seed,
                    "dtype": args.dtype,
                    "scheduler": pipe.scheduler.__class__.__name__,
                    "latents_scaled_sigma": "" if sigma is None else sigma,
                    "negative_prompt": args.negative_prompt,
                    "zT_pt": args.zT_pt,
                    "sec_per_prompt": round(dt, 4),
                }
                rows.append(row)

                w.writerow([
                    row["file"], row["prompt_idx"], row["img_idx"], row["prompt"], row["zT_idx"],
                    row["wm_method"], row["wm_tag"],
                    row["model_id"], row["steps"], row["cfg"], row["height"], row["width"],
                    row["seed"], row["dtype"], row["scheduler"], row["latents_scaled_sigma"],
                    row["negative_prompt"], row["zT_pt"], row["sec_per_prompt"],
                ])

            print(f"[OK] prompt {pidx:02d}/{len(prompts)-1:02d} -> {B} imgs | {dt:.2f}s")

    # optional run summary
    summary = {
        "model_id": args.model_id,
        "prompts": args.prompts,
        "n_prompts": len(prompts),
        "zT_pt": args.zT_pt,
        "wm_method": method,
        "wm_tag": tag,
        "n_per_prompt": B,
        "start_latent": start,
        "latents_already_scaled": bool(args.latents_already_scaled),
        "latents_scaled_sigma": None if sigma is None else sigma,
        "negative_prompt": args.negative_prompt,
        "steps": args.steps,
        "cfg": args.cfg,
        "height": args.height,
        "width": args.width,
        "dtype": args.dtype,
        "seed": args.seed,
        "outdir": args.outdir,
        "manifest": str(manifest_path),
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n✅ DONE")
    print(f"✅ images -> {sliced_dir}/Pxx_yy.png")
    print(f"✅ manifest -> {manifest_path}")
    print(f"✅ summary -> {out_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
