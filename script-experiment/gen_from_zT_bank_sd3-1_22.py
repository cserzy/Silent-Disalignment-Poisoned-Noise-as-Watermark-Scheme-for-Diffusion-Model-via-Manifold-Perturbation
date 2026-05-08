#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from diffusers import StableDiffusion3Pipeline


# -------------------------
# Utils
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_prompts(txt_path: str) -> List[str]:
    lines = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            lines.append(s)
    return lines

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
    Load z_T bank tensor, expected 4D: [N, C, H, W].
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
    if zT.shape[0] < 4:
        raise ValueError(f"[zT] Need at least 4 latents, got N={zT.shape[0]}")

    return zT.float()


# -------------------------
# Build SD3 pipeline
# -------------------------
def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    if s == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")

def build_pipe_sd3(model_id: str, device: str, dtype: torch.dtype, cpu_offload: int, disable_t5: int) -> StableDiffusion3Pipeline:
    # SD3: safety checker is not the focus here; you do NSFW scoring later.
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    # Optional: drop T5 encoder for memory
    if int(disable_t5) == 1:
        pipe.text_encoder_3 = None
        pipe.tokenizer_3 = None

    if int(cpu_offload) == 1:
        # recommended for SD3 on commodity GPUs (slower but stable)
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    pipe.set_progress_bar_config(disable=True)
    return pipe

def get_sd3_in_channels(pipe: StableDiffusion3Pipeline) -> int:
    # SD3 usually has a `transformer` module (MMDiT-like).
    # We try to read in_channels robustly.
    if hasattr(pipe, "transformer") and hasattr(pipe.transformer, "config") and hasattr(pipe.transformer.config, "in_channels"):
        return int(pipe.transformer.config.in_channels)
    # fallback (very rare)
    if hasattr(pipe, "unet") and hasattr(pipe.unet, "config") and hasattr(pipe.unet.config, "in_channels"):
        return int(pipe.unet.config.in_channels)
    raise AttributeError("Cannot infer latent in_channels from SD3 pipeline (no transformer.config.in_channels).")


# -------------------------
# Main
# -------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_id", type=str, required=True, help="local path or HF repo id of SD3 diffusers folder")
    ap.add_argument("--prompts", type=str, required=True, help="txt: one prompt per line")
    ap.add_argument("--zT_pt", type=str, required=True, help="pt: z_T bank [N,C,H,W], you will use first B latents")
    ap.add_argument("--outdir", type=str, required=True, help="output dir (images + manifest in outdir/sliced/)")

    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)

    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    # fixed: 4 images per prompt (but keep as hyperparam)
    ap.add_argument("--n_per_prompt", type=int, default=4, help="images per prompt (default 4)")
    ap.add_argument("--start_latent", type=int, default=0, help="start index in z_T bank (default 0)")

    # scaling switch (same semantics as your SD1/2 script)
    ap.add_argument("--latents_already_scaled", action="store_true",
                    help="If set, do NOT multiply latents by scheduler.init_noise_sigma (default: multiply when available).")

    # negative prompt
    ap.add_argument("--negative_prompt", type=str, default="", help="negative prompt string (default empty)")

    # SD3 memory knobs (align with your SD3 script)
    ap.add_argument("--cpu_offload", type=int, default=0, help="enable_model_cpu_offload (slower but saves VRAM)")
    ap.add_argument("--disable_t5", type=int, default=0, help="drop text_encoder_3/tokenizer_3 for memory (slight quality drop)")
    ap.add_argument("--max_sequence_length", type=int, default=512)

    args = ap.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.outdir)
    sliced_dir = out_dir / "sliced"
    ensure_dir(sliced_dir)

    prompts = load_prompts(args.prompts)
    if len(prompts) == 0:
        raise ValueError("No prompts loaded.")

    method, tag = parse_method_and_tag(args.zT_pt)

    torch_dtype = _dtype_from_str(args.dtype)
    pipe = build_pipe_sd3(
        args.model_id,
        device=args.device,
        dtype=torch_dtype,
        cpu_offload=int(args.cpu_offload),
        disable_t5=int(args.disable_t5),
    )

    zT_bank = load_zT_bank(args.zT_pt)  # CPU float32
    start = int(args.start_latent)
    B = int(args.n_per_prompt)
    if start + B > zT_bank.shape[0]:
        raise ValueError(f"[zT] start_latent({start}) + n_per_prompt({B}) exceeds bank size N={zT_bank.shape[0]}")

    # SD3 latent channel check
    expected_c = get_sd3_in_channels(pipe)
    if int(zT_bank.shape[1]) != int(expected_c):
        raise ValueError(
            f"[zT] Channel mismatch for SD3: zT_bank has C={zT_bank.shape[1]}, "
            f"but SD3 pipeline expects in_channels={expected_c}. "
            f"Please regenerate zT bank for SD3 or use matching latents."
        )

    # select working latents (size=B)
    zT_batch = zT_bank[start:start + B].clone()  # CPU
    sigma = None
    if not args.latents_already_scaled:
        # If scheduler has init_noise_sigma, apply standard scaling
        if hasattr(pipe, "scheduler") and hasattr(pipe.scheduler, "init_noise_sigma"):
            sigma = float(pipe.scheduler.init_noise_sigma)
            zT_batch = zT_batch * sigma
        else:
            sigma = None  # no scaling available

    # move once (match pipeline dtype)
    # For cpu_offload, latents can still be on cuda; pipeline will manage modules
    zT_batch = zT_batch.to(device=args.device, dtype=torch_dtype)

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
            "sd3_cpu_offload",
            "sd3_disable_t5",
            "sd3_max_sequence_length",
        ])

        for pidx, prompt in enumerate(prompts):
            batch_prompts = [prompt] * B
            negs = [args.negative_prompt] * B

            t0 = time.time()
            out = pipe(
                prompt=batch_prompts,
                negative_prompt=negs,
                num_inference_steps=int(args.steps),
                guidance_scale=float(args.cfg),
                height=int(args.height),
                width=int(args.width),
                latents=zT_batch,
                max_sequence_length=int(args.max_sequence_length),
            )
            images = out.images
            dt = time.time() - t0

            for j, im in enumerate(images):
                out_name = f"P{pidx:02d}_{j:02d}.png"
                out_path = sliced_dir / out_name
                im.save(out_path)

                w.writerow([
                    str(out_path),
                    pidx,
                    j,
                    prompt,
                    start + j,
                    method,
                    tag,
                    args.model_id,
                    args.steps,
                    args.cfg,
                    args.height,
                    args.width,
                    args.seed,
                    args.dtype,
                    getattr(pipe.scheduler, "__class__", type(pipe.scheduler)).__name__,
                    "" if sigma is None else sigma,
                    args.negative_prompt,
                    args.zT_pt,
                    round(dt, 4),
                    int(args.cpu_offload),
                    int(args.disable_t5),
                    int(args.max_sequence_length),
                ])

    print(f"[DONE] outdir={out_dir}")
    print(f"[DONE] images_dir={sliced_dir}")
    print(f"[DONE] manifest={manifest_path}")


if __name__ == "__main__":
    main()
