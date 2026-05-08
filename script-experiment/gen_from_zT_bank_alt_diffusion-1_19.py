#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate images from a fixed z_T bank tensor for a SINGLE Alt Diffusion model.

Requirements (your experiment setting):
- One model per run
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
import inspect
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, PNDMScheduler, UNet2DConditionModel
from diffusers.pipelines.deprecated.alt_diffusion import AltDiffusionPipeline
from diffusers.pipelines.deprecated.alt_diffusion.modeling_roberta_series import (
    RobertaSeriesModelWithTransformation,
)
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, XLMRobertaTokenizer


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
    Load z_T bank tensor, shape [N,C,H,W].
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
        raise ValueError(f"[zT] Expect 4D tensor [N,C,H,W], got {tuple(zT.shape)}")
    if zT.shape[0] < 1:
        raise ValueError(f"[zT] Need at least 1 latent, got N={zT.shape[0]}")

    return zT.float()


def build_pipe(model_id: str, device: str, dtype: torch.dtype, disable_safety_checker: bool = False):
    model_path = Path(model_id)

    print(f"[load] tokenizer <- {model_path / 'tokenizer'}")
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    print(f"[load] text_encoder <- {model_path / 'text_encoder'}")
    text_encoder = RobertaSeriesModelWithTransformation.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype
    )

    print(f"[load] vae <- {model_path / 'vae'}")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)

    print(f"[load] unet <- {model_path / 'unet'}")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)

    print(f"[load] scheduler <- {model_path / 'scheduler'}")
    scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")

    if disable_safety_checker:
        print("[load] safety_checker disabled by --disable_safety_checker")
        feature_extractor = None
        safety_checker = None
        requires_safety_checker = False
    else:
        print(f"[load] feature_extractor <- {model_path / 'feature_extractor'}")
        feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor")

        print(f"[load] safety_checker <- {model_path / 'safety_checker'}")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            model_id, subfolder="safety_checker", torch_dtype=dtype
        )
        requires_safety_checker = True

    pipe = AltDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=safety_checker,
        feature_extractor=feature_extractor,
        image_encoder=None,
        requires_safety_checker=requires_safety_checker,
    )

    # Keep original script style when possible: prefer DPM from scheduler config.
    # For compatibility, fall back to checkpoint default scheduler if replacement fails.
    scheduler_switched_to_dpm = False
    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        scheduler_switched_to_dpm = True
    except Exception as e:
        print(f"[WARN] Failed to switch scheduler to DPMSolverMultistepScheduler: {e}")
        print("[WARN] Keep pipeline default scheduler for compatibility.")

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe, scheduler_switched_to_dpm


def validate_latent_shape_with_pipe(zT_bank: torch.Tensor, pipe: AltDiffusionPipeline, args: argparse.Namespace):
    if not hasattr(pipe, "unet") or pipe.unet is None:
        raise RuntimeError("[pipe] AltDiffusionPipeline has no unet; cannot validate latent shape.")
    if not hasattr(pipe.unet, "config"):
        raise RuntimeError("[pipe] unet has no config; cannot validate latent shape.")
    if not hasattr(pipe.unet.config, "in_channels"):
        raise RuntimeError("[pipe] unet.config has no in_channels; cannot validate latent channels.")

    latent_channels = int(pipe.unet.config.in_channels)

    if not hasattr(pipe, "vae_scale_factor"):
        raise RuntimeError("[pipe] pipeline has no vae_scale_factor; cannot validate latent spatial size.")
    vae_scale_factor = int(pipe.vae_scale_factor)
    if vae_scale_factor <= 0:
        raise RuntimeError(f"[pipe] invalid vae_scale_factor={vae_scale_factor}")

    if (args.height % vae_scale_factor) != 0 or (args.width % vae_scale_factor) != 0:
        raise ValueError(
            "[shape] height/width must be divisible by vae_scale_factor. "
            f"got height={args.height}, width={args.width}, vae_scale_factor={vae_scale_factor}"
        )

    expected_h = args.height // vae_scale_factor
    expected_w = args.width // vae_scale_factor
    actual = tuple(zT_bank.shape)
    sample_size = getattr(pipe.unet.config, "sample_size", None)

    if zT_bank.shape[1] != latent_channels or zT_bank.shape[2] != expected_h or zT_bank.shape[3] != expected_w:
        raise ValueError(
            "[shape] zT latent shape mismatch for Alt Diffusion checkpoint.\n"
            f"  actual zT shape: {actual}\n"
            f"  expected latent_channels: {latent_channels}\n"
            f"  expected_h: {expected_h}\n"
            f"  expected_w: {expected_w}\n"
            f"  vae_scale_factor: {vae_scale_factor}\n"
            f"  unet.config.sample_size: {sample_size}\n"
            f"  unet.config.in_channels: {latent_channels}"
        )

    return latent_channels, expected_h, expected_w, vae_scale_factor, sample_size


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_id", type=str, required=True,
                    help="One model path or HF repo id (Alt Diffusion diffusers folder).")
    ap.add_argument("--prompts", type=str, required=True, help="txt: one prompt per line")
    ap.add_argument("--zT_pt", type=str, required=True, help="pt: z_T bank [N,C,H,W]")
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

    # Keep CLI compatibility with original script.
    # NOTE: if current AltDiffusionPipeline signature does not accept negative_prompt, this field is ignored at call-time.
    ap.add_argument("--negative_prompt", type=str, default="",
                    help="negative prompt string (default empty)")
    ap.add_argument("--disable_safety_checker", action="store_true",
                    help="Disable safety_checker and feature_extractor for debugging.")

    args = ap.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.outdir)
    sliced_dir = out_dir / "sliced"
    ensure_dir(sliced_dir)

    prompts = load_prompts(args.prompts)
    if len(prompts) == 0:
        raise ValueError(f"No prompts loaded from: {args.prompts}")

    # load z_T ONCE
    zT_bank = load_zT_bank(args.zT_pt)  # CPU [N,C,H,W]
    N = zT_bank.shape[0]
    B = args.n_per_prompt
    start = args.start_latent
    if B <= 0:
        raise ValueError(f"--n_per_prompt must be positive, got {B}")
    if start < 0 or start >= N:
        raise ValueError(f"--start_latent out of range: {start} (N={N})")
    if start + B > N:
        raise ValueError(f"Need {B} latents from index {start}, but N={N} (start+B={start + B})")

    method, tag = parse_method_and_tag(args.zT_pt)

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    pipe, scheduler_switched_to_dpm = build_pipe(
        args.model_id,
        device=args.device,
        dtype=torch_dtype,
        disable_safety_checker=args.disable_safety_checker,
    )
    print(
        f"[safety] enabled={pipe.safety_checker is not None} "
        f"feature_extractor_loaded={pipe.feature_extractor is not None}"
    )

    latent_channels, expected_h, expected_w, vae_scale_factor, sample_size = validate_latent_shape_with_pipe(
        zT_bank=zT_bank,
        pipe=pipe,
        args=args,
    )

    # Validate pipeline call signature for external latents and negative_prompt compatibility.
    call_sig = inspect.signature(pipe.__call__)
    call_params = set(call_sig.parameters.keys())
    supports_external_latents = "latents" in call_params
    supports_negative_prompt = "negative_prompt" in call_params

    if not supports_external_latents:
        raise RuntimeError(
            "Current AltDiffusionPipeline does NOT support external `latents` input in __call__. "
            "This script requires fixed z_T bank injection and will stop instead of pseudo-compat mode."
        )

    if not supports_negative_prompt:
        print(
            "[WARN] Current AltDiffusionPipeline.__call__ does not accept `negative_prompt`; "
            "the --negative_prompt argument will be ignored."
        )

    # select working latents (size=B)
    zT_batch = zT_bank[start:start + B].clone()  # CPU
    sigma = None
    if not args.latents_already_scaled:
        sigma = float(pipe.scheduler.init_noise_sigma)
        zT_batch = zT_batch * sigma  # keep original script behavior

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
            "pipe_class",
            "supports_external_latents",
            "latent_channels",
            "vae_scale_factor",
        ])

        for pidx, prompt in enumerate(prompts):
            batch_prompts = [prompt] * B
            negs = [args.negative_prompt] * B

            call_kwargs = {
                "prompt": batch_prompts,
                "num_inference_steps": args.steps,
                "guidance_scale": args.cfg,
                "height": args.height,
                "width": args.width,
                "latents": zT_batch,
            }
            if supports_negative_prompt:
                call_kwargs["negative_prompt"] = negs

            t0 = time.time()
            result = pipe(**call_kwargs)
            dt = time.time() - t0
            images = result.images

            if hasattr(result, "nsfw_content_detected"):
                nsfw_content_detected = result.nsfw_content_detected
                print(f"[safety] prompt {pidx:02d} nsfw_content_detected={nsfw_content_detected}")
            else:
                nsfw_content_detected = None
                print(f"[safety] prompt {pidx:02d} no nsfw_content_detected field found")

            for j, im in enumerate(images):
                im_arr = np.asarray(im)
                is_black = bool(im_arr.max() == 0)
                nsfw_flag = None
                if nsfw_content_detected is not None and j < len(nsfw_content_detected):
                    nsfw_flag = bool(nsfw_content_detected[j])
                print(
                    f"[safety] image P{pidx:02d}_{j:02d} "
                    f"nsfw_flag={nsfw_flag} is_black={is_black} "
                    f"pixel_min={int(im_arr.min())} pixel_max={int(im_arr.max())} "
                    f"pixel_mean={float(im_arr.mean()):.4f}"
                )

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
                    "pipe_class": pipe.__class__.__name__,
                    "supports_external_latents": bool(supports_external_latents),
                    "latent_channels": latent_channels,
                    "vae_scale_factor": vae_scale_factor,
                }
                rows.append(row)

                w.writerow([
                    row["file"], row["prompt_idx"], row["img_idx"], row["prompt"], row["zT_idx"],
                    row["wm_method"], row["wm_tag"],
                    row["model_id"], row["steps"], row["cfg"], row["height"], row["width"],
                    row["seed"], row["dtype"], row["scheduler"], row["latents_scaled_sigma"],
                    row["negative_prompt"], row["zT_pt"], row["sec_per_prompt"],
                    row["pipe_class"], row["supports_external_latents"], row["latent_channels"], row["vae_scale_factor"],
                ])

            print(f"[OK] prompt {pidx:02d}/{len(prompts)-1:02d} -> {B} imgs | {dt:.2f}s")

    # optional run summary
    summary = {
        "model_id": args.model_id,
        "pipe_class": pipe.__class__.__name__,
        "scheduler_switched_to_dpm": bool(scheduler_switched_to_dpm),
        "supports_external_latents": bool(supports_external_latents),
        "supports_negative_prompt": bool(supports_negative_prompt),
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
        "latent_channels": latent_channels,
        "expected_latent_h": expected_h,
        "expected_latent_w": expected_w,
        "vae_scale_factor": vae_scale_factor,
        "unet_sample_size": sample_size,
        "outdir": args.outdir,
        "manifest": str(manifest_path),
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[DONE]")
    print(f"[DONE] images -> {sliced_dir}/Pxx_yy.png")
    print(f"[DONE] manifest -> {manifest_path}")
    print(f"[DONE] summary -> {out_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
