#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate images from a fixed z_T bank tensor: [16,4,64,64] for a SINGLE model.

COCO tweak (minimal changes vs gen_from_zT_bank_multi_models-1_19.py):
- 1 prompt -> 1 image (default --n_per_prompt=1)
- zT cyclic mapping per prompt:
    prompt_idx p uses zT_idx = (start_latent + p) % N
  (N is usually 16)
- All other logic kept the same:
  - load zT once
  - optional scaling by scheduler.init_noise_sigma unless --latents_already_scaled
  - save to outdir/sliced as Pxx_yy.png
  - write one manifest.csv
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
    Expect (relaxed):
      generate_{METHOD}_{TAG}.pt
      generate_{METHOD}_{TAG}_anything.pt   (e.g., generate_TR_w_att_0_88.pt)
    METHOD: TR/GS/PRC/T2S (case-insensitive)
    TAG: w / w_att
    """
    name = Path(zT_path).name
    m = re.match(r"generate_([A-Za-z0-9]+)_([A-Za-z0-9]+)(?:_.*)?\.pt$", name)
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
        for k in ["zT_bank", "zT", "z_t", "z_T", "latents", "noise", "z"]:
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
    if zT.shape[0] < 1:
        raise ValueError(f"[zT] Need at least 1 latent, got N={zT.shape[0]}")

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

    # COCO default: 1 image per prompt (still keep as hyperparam, but cycle logic triggers when == 1)
    ap.add_argument("--n_per_prompt", type=int, default=1, help="images per prompt (COCO default 1)")
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
    N = int(zT_bank.shape[0])

    start = int(args.start_latent)
    if start < 0 or start >= N:
        raise ValueError(f"--start_latent out of range: {start} (N={N})")

    method, tag = parse_method_and_tag(args.zT_pt)

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    pipe = build_pipe(args.model_id, device=args.device, dtype=torch_dtype)

    sigma = None
    zT_bank_work = zT_bank  # CPU
    if not args.latents_already_scaled:
        sigma = float(pipe.scheduler.init_noise_sigma)
        zT_bank_work = zT_bank_work * sigma

    # move once to GPU (N is small, e.g., 16)
    zT_bank_work = zT_bank_work.to(device=args.device, dtype=pipe.unet.dtype)

    manifest_path = sliced_dir / "manifest.csv"

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

        B = int(args.n_per_prompt)

        # ---------
        # COCO mode: 1 prompt -> 1 image, zT cycles with prompt index
        # ---------
        if B == 1:
            for pidx, prompt in enumerate(prompts):
                z_idx = (start + pidx) % N
                zT_one = zT_bank_work[z_idx:z_idx + 1]  # [1,4,64,64]

                t0 = time.time()
                images = pipe(
                    prompt=[prompt],
                    negative_prompt=[args.negative_prompt],
                    num_inference_steps=args.steps,
                    guidance_scale=args.cfg,
                    height=args.height,
                    width=args.width,
                    latents=zT_one,
                ).images
                dt = time.time() - t0

                im = images[0]
                out_name = f"P{pidx:02d}_{0:02d}.png"
                out_path = sliced_dir / out_name
                im.save(out_path)

                w.writerow([
                    str(out_path), pidx, 0, prompt, z_idx,
                    method, tag,
                    args.model_id, args.steps, args.cfg, args.height, args.width,
                    args.seed, args.dtype, pipe.scheduler.__class__.__name__,
                    "" if sigma is None else sigma,
                    args.negative_prompt, args.zT_pt, round(dt, 4),
                ])

                print(f"[OK] prompt {pidx:02d}/{len(prompts)-1:02d} -> 1 img | zT[{z_idx}] | {dt:.2f}s")

        # ---------
        # Fallback: keep original behavior (B images per prompt using a fixed slice zT[start:start+B])
        # ---------
        else:
            if start + B > N:
                raise ValueError(f"Need {B} latents from {start}, but N={N}")
            zT_batch = zT_bank_work[start:start + B]  # [B,4,64,64]

            for pidx, prompt in enumerate(prompts):
                batch_prompts = [prompt] * B
                negs = [args.negative_prompt] * B

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

                    w.writerow([
                        str(out_path), pidx, j, prompt, start + j,
                        method, tag,
                        args.model_id, args.steps, args.cfg, args.height, args.width,
                        args.seed, args.dtype, pipe.scheduler.__class__.__name__,
                        "" if sigma is None else sigma,
                        args.negative_prompt, args.zT_pt, round(dt, 4),
                    ])

                print(f"[OK] prompt {pidx:02d}/{len(prompts)-1:02d} -> {B} imgs | {dt:.2f}s")

    summary = {
        "model_id": args.model_id,
        "prompts": args.prompts,
        "n_prompts": len(prompts),
        "zT_pt": args.zT_pt,
        "wm_method": method,
        "wm_tag": tag,
        "n_per_prompt": int(args.n_per_prompt),
        "start_latent": int(args.start_latent),
        "coco_cycle_mode": bool(int(args.n_per_prompt) == 1),
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
