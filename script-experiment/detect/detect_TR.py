#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_TR.py  (fixed-threshold version)

Changes vs previous:
  - Remove wrong-key stats and global threshold calibration.
  - Use a fixed distance threshold: --dist_thr (default 75.0).
  - Invert/detect per image and print per-image result in terminal for easy debugging.

Still kept:
  - mode: img / pt / both
  - unknown prompt (default empty), optional manifest_first/manual
  - Excel output: summary + detect_result
"""

import os
import re
import csv
import json
import glob
import argparse
import hashlib
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch

from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DDIMInverseScheduler,
    DPMSolverMultistepScheduler,
)

from openpyxl import Workbook


# -------------------------
# Manifest / IO utils
# -------------------------

def find_manifest(base_dir: str) -> Optional[str]:
    cands = [
        os.path.join(base_dir, "manifest.csv"),
        os.path.join(base_dir, "sliced", "manifest.csv"),
    ]
    parent = os.path.dirname(os.path.abspath(base_dir))
    cands += [
        os.path.join(parent, "manifest.csv"),
        os.path.join(parent, "sliced", "manifest.csv"),
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    return None


def read_manifest_first_prompt(manifest_path: str) -> Optional[str]:
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            first = next(r, None)
            if first is None:
                return None
            return str(first.get("prompt", "") or "")
    except Exception:
        return None


def collect_images(img_dir: str) -> List[str]:
    img_dir = img_dir.rstrip("/")
    direct = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if direct:
        return direct

    sliced = sorted(glob.glob(os.path.join(img_dir, "sliced", "*.png")))
    if sliced:
        return sliced

    return sorted(glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True))


def collect_pts(base_dir: str, pt_dir: Optional[str]) -> List[str]:
    cand_dirs: List[str] = []
    if pt_dir:
        cand_dirs.append(pt_dir)
    cand_dirs += [
        os.path.join(base_dir, "zt"),
        os.path.join(base_dir, "saved_zT_post"),
        os.path.join(base_dir, "saved_zT_pre"),
        base_dir,
    ]
    all_pts: List[str] = []
    for d in cand_dirs:
        if not d or (not os.path.isdir(d)):
            continue
        p1 = sorted(glob.glob(os.path.join(d, "*_zT_refined.pt")))
        if p1:
            return p1
        p2 = sorted(glob.glob(os.path.join(d, "*_zT_*.pt")))
        if p2:
            return p2
        p3 = sorted(glob.glob(os.path.join(d, "*.pt")))
        if p3:
            all_pts.extend(p3)

    if all_pts:
        return all_pts

    rec = sorted(glob.glob(os.path.join(base_dir, "**", "saved_zT_post", "*.pt"), recursive=True))
    if rec:
        refined = [p for p in rec if p.endswith("_zT_refined.pt")]
        return sorted(refined) if refined else rec

    return []


def parse_Pxx_yy(filename: str) -> Tuple[Optional[int], Optional[int]]:
    b = os.path.basename(filename)
    m = re.search(r"[Pp](\d+)[_\-](\d+)\.png$", b)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


# -------------------------
# Tree-Ring: mask/pattern/dist
# -------------------------

def fft2_shift(x: torch.Tensor) -> torch.Tensor:
    X = torch.fft.fft2(x)
    X = torch.fft.fftshift(X, dim=(-2, -1))
    return X


def build_mask_4d(
    B: int, C: int, H: int, W: int,
    mask_shape: str,
    radius: int,
    channel: int,
    mask_mode: str,
    device: torch.device,
) -> torch.Tensor:
    yy = torch.arange(H, device=device).view(H, 1).repeat(1, W)
    xx = torch.arange(W, device=device).view(1, W).repeat(H, 1)
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    dy = yy.float() - cy
    dx = xx.float() - cx

    if mask_shape.lower() == "circle":
        rr = torch.sqrt(dx * dx + dy * dy)
        disk_r = rr <= float(radius)
        if mask_mode == "disk":
            m2 = disk_r
        elif mask_mode == "ringband":
            inner = rr <= float(max(radius - 1, 0))
            m2 = disk_r & (~inner)
        else:
            raise ValueError(f"unknown mask_mode={mask_mode}")
    elif mask_shape.lower() == "square":
        ax = dx.abs()
        ay = dy.abs()
        disk_r = (ax <= float(radius)) & (ay <= float(radius))
        if mask_mode == "disk":
            m2 = disk_r
        elif mask_mode == "ringband":
            inner = (ax <= float(max(radius - 1, 0))) & (ay <= float(max(radius - 1, 0)))
            m2 = disk_r & (~inner)
        else:
            raise ValueError(f"unknown mask_mode={mask_mode}")
    else:
        raise ValueError(f"unknown mask_shape={mask_shape}")

    mask = m2.view(1, 1, H, W).repeat(B, C, 1, 1)
    if int(channel) >= 0:
        keep = torch.zeros((1, C, 1, 1), device=device, dtype=torch.bool)
        keep[:, int(channel):int(channel) + 1] = True
        mask = mask & keep
    return mask


def effective_seed(base_seed: int, key_bytes: Optional[bytes], seed_from_key: bool) -> int:
    if (not seed_from_key) or (not key_bytes):
        return int(base_seed)
    h = hashlib.sha256(b"treering|" + key_bytes).digest()
    k = int.from_bytes(h[:4], byteorder="little", signed=False)
    return int(base_seed) ^ k


def load_key_bytes(key_path: Optional[str], key_hex: Optional[str]) -> Optional[bytes]:
    if key_hex:
        key_hex = key_hex.strip().lower()
        key_hex = re.sub(r"^0x", "", key_hex)
        return bytes.fromhex(key_hex)
    if key_path:
        with open(key_path, "rb") as f:
            return f.read()
    return None


def build_pattern_fft(
    seed: int,
    shape: Tuple[int, int, int, int],
    pattern: str,
    const: float,
    device: torch.device,
    gen_cuda_plus1: bool = False,  # aligned default False
) -> torch.Tensor:
    """
    Return complex FFT-domain patch aligned with injection=complex:
      patch_fft = fftshift(fft2(randn(seed)))
    """
    B, C, H, W = shape
    g = torch.Generator(device=device)
    seed_eff = int(seed) + (1 if (gen_cuda_plus1 and device.type == "cuda") else 0)
    g.manual_seed(seed_eff)

    if pattern.lower() in ("ring", "rand"):
        z = torch.randn((B, C, H, W), generator=g, device=device)
        patch_fft = fft2_shift(z).to(torch.complex64)
        return patch_fft

    if pattern.lower() == "const":
        z = torch.full((B, C, H, W), float(const), device=device)
        patch_fft = fft2_shift(z).to(torch.complex64)
        return patch_fft

    raise ValueError(f"unknown pattern={pattern}")


def dist_fft(latents: torch.Tensor, patch_fft: torch.Tensor, mask: torch.Tensor) -> float:
    Z = fft2_shift(latents).to(torch.complex64)
    diff = (Z[mask] - patch_fft[mask]).abs()
    return float(diff.mean().item())


# -------------------------
# DDIM inversion (image -> noisy latent)
# -------------------------

def pil_to_tensor(im: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    arr = np.array(im).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    t = t * 2.0 - 1.0
    return t.to(device=device, dtype=dtype)


@torch.no_grad()
def ddim_invert_to_noise_latent(
    pipe: StableDiffusionPipeline,
    image: Image.Image,
    steps: int,
    prompt: str,
    guidance_scale: float,
    inv_scheduler: DDIMInverseScheduler,
    dtype: torch.dtype,
) -> torch.Tensor:
    device = pipe._execution_device if hasattr(pipe, "_execution_device") else pipe.device

    do_cfg = float(guidance_scale) > 1.0
    if hasattr(pipe, "encode_prompt"):
        prompt_embeds, negative_embeds = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt="",
        )
    else:
        prompt_embeds = pipe._encode_prompt(
            prompt,
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt="",
        )
        negative_embeds = None

    if do_cfg:
        if negative_embeds is None:
            text_embeds = prompt_embeds
        else:
            text_embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
    else:
        text_embeds = prompt_embeds

    im_t = pil_to_tensor(image, device=device, dtype=dtype)
    lat0 = pipe.vae.encode(im_t).latent_dist.sample()
    lat0 = lat0 * pipe.vae.config.scaling_factor

    inv_scheduler.set_timesteps(int(steps), device=device)
    lat = lat0

    for t in inv_scheduler.timesteps:
        lat_in = torch.cat([lat] * 2, dim=0) if do_cfg else lat
        noise_pred = pipe.unet(lat_in, t, encoder_hidden_states=text_embeds).sample
        if do_cfg:
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + float(guidance_scale) * (noise_text - noise_uncond)
        lat = inv_scheduler.step(noise_pred, t, lat).prev_sample

    return lat


# -------------------------
# Pipe / sigma
# -------------------------

def build_pipe(model_id: str, device: torch.device, fp16: bool) -> StableDiffusionPipeline:
    dtype = torch.float16 if fp16 else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    return pipe


def get_sigma(pipe: StableDiffusionPipeline, sigma_scheduler: str) -> float:
    if sigma_scheduler == "dpm":
        sch = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif sigma_scheduler == "ddim":
        sch = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"unknown sigma_scheduler={sigma_scheduler}")
    return float(sch.init_noise_sigma)


# -------------------------
# Excel writer
# -------------------------

def write_excel_two_sheets(
    out_xlsx: str,
    summary_rows: List[Tuple[str, Any]],
    detail_header: List[str],
    detail_rows: List[List[Any]],
) -> None:
    wb = Workbook()

    ws_sum = wb.active
    ws_sum.title = "summary"
    ws_sum.append(["metric", "value"])
    for k, v in summary_rows:
        ws_sum.append([k, v])

    ws_det = wb.create_sheet("detect_result")
    ws_det.append(detail_header)
    for r in detail_rows:
        ws_det.append(r)

    os.makedirs(os.path.dirname(out_xlsx) or ".", exist_ok=True)
    wb.save(out_xlsx)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()

    # inputs
    ap.add_argument("--img_dir", type=str, required=True, help="directory containing pngs (sliced or direct)")
    ap.add_argument("--model_id", type=str, required=True)

    # modes
    ap.add_argument("--mode", type=str, default="img", choices=["img", "pt", "both"])
    ap.add_argument("--pt_dir", type=str, default=None, help="optional directory containing saved zT .pt")
    ap.add_argument("--max_items", type=int, default=-1, help="limit number of images/pts for quick debug")

    # inversion
    ap.add_argument("--detect_prompt", type=str, default="empty", choices=["empty", "manifest_first", "manual"])
    ap.add_argument("--prompt", type=str, default="", help="used when detect_prompt=manual")
    ap.add_argument("--guidance_scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=50)

    # latent scaling
    ap.add_argument("--latent_mode", type=str, default="zT", choices=["zT", "sigma_scaled"])
    ap.add_argument("--sigma_scheduler", type=str, default="dpm", choices=["dpm", "ddim"])

    # Fixed detection threshold
    ap.add_argument("--dist_thr", type=float, default=75.0, help="fixed dist threshold (no wrong-key)")

    # treering params (ALIGNED DEFAULTS)
    ap.add_argument("--mask_mode", type=str, default="disk", choices=["disk", "ringband"])
    ap.add_argument("--tr_w_seed", type=int, default=12345)         # IMPORTANT
    ap.add_argument("--tr_seed_from_key", type=int, default=0)      # IMPORTANT: keep 0 to avoid seed drift
    ap.add_argument("--key_path", type=str, default=None)
    ap.add_argument("--key_hex", type=str, default=None)

    ap.add_argument("--tr_w_pattern", type=str, default="ring", choices=["ring", "rand", "const"])
    ap.add_argument("--tr_w_pattern_const", type=float, default=0.0)
    ap.add_argument("--tr_w_mask_shape", type=str, default="circle", choices=["circle", "square"])
    ap.add_argument("--tr_w_radius", type=int, default=9)
    ap.add_argument("--tr_w_channel", type=int, default=-1)

    # perf / output
    ap.add_argument("--fp16", type=int, default=1)
    ap.add_argument("--out_xlsx", type=str, default=None, help="xlsx output path; default <img_dir>/treering_detect.xlsx")
    ap.add_argument("--out_json", type=str, default=None, help="optional json dump path")

    args = ap.parse_args()

    img_dir = args.img_dir
    assert os.path.isdir(img_dir), f"img_dir not found: {img_dir}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = build_pipe(args.model_id, device=device, fp16=bool(args.fp16))
    dtype = torch.float16 if (bool(args.fp16) and device.type == "cuda") else torch.float32

    # prompt selection (unknown prompt default: empty)
    manifest_path = find_manifest(img_dir)
    if args.detect_prompt == "empty":
        detect_prompt = ""
    elif args.detect_prompt == "manifest_first":
        if not manifest_path:
            raise FileNotFoundError("detect_prompt=manifest_first but manifest.csv not found near img_dir")
        detect_prompt = read_manifest_first_prompt(manifest_path) or ""
    else:
        detect_prompt = str(args.prompt)

    sigma = get_sigma(pipe, sigma_scheduler=args.sigma_scheduler)

    # Key bytes remain optional; seed_from_key=0 preserves the default behavior
    key_bytes = load_key_bytes(args.key_path, args.key_hex)

    # build mask + patch (precompute)
    B, C, H, W = (1, 4, 64, 64)
    mask = build_mask_4d(
        B, C, H, W,
        mask_shape=args.tr_w_mask_shape,
        radius=args.tr_w_radius,
        channel=args.tr_w_channel,
        mask_mode=args.mask_mode,
        device=device,
    )

    seed_correct = effective_seed(args.tr_w_seed, key_bytes, bool(args.tr_seed_from_key))
    correct_patch = build_pattern_fft(
        seed=seed_correct,
        shape=(B, C, H, W),
        pattern=args.tr_w_pattern,
        const=args.tr_w_pattern_const,
        device=device,
        gen_cuda_plus1=False,  # aligned
    )

    # prepare items list
    img_paths: List[str] = []
    pt_paths: List[str] = []

    if args.mode in ("img", "both"):
        img_paths = collect_images(img_dir)
        if args.max_items > 0:
            img_paths = img_paths[: int(args.max_items)]
        if not img_paths:
            raise FileNotFoundError(f"No PNG images found under img_dir={img_dir}")

    if args.mode in ("pt", "both"):
        pt_paths = collect_pts(img_dir, pt_dir=args.pt_dir)
        if args.max_items > 0:
            pt_paths = pt_paths[: int(args.max_items)]
        if not pt_paths:
            raise FileNotFoundError(f"No PT files found (img_dir={img_dir}, pt_dir={args.pt_dir})")

    # build inversion schedulers once
    ddim = DDIMScheduler.from_config(pipe.scheduler.config)
    inv = DDIMInverseScheduler.from_config(ddim.config)

    # -------------------------
    # Detect all items with fixed thr
    # -------------------------
    thr = float(args.dist_thr)

    detail_header = [
        "mode", "file",
        "prompt_id", "img_id", "latent_id",
        "dist", "dist_thr", "ok",
        "prompt_used", "steps", "guidance_scale",
        "latent_mode", "sigma_scheduler", "sigma",
        "seed_used",
    ]
    detail_rows: List[List[Any]] = []

    ok_cnt = 0
    total_cnt = 0

    def record_one(mode: str, file_path: str, lat: torch.Tensor, prompt_used: str):
        nonlocal ok_cnt, total_cnt
        dist = dist_fft(lat, correct_patch, mask)
        ok = bool(dist <= thr)

        pid, iid = parse_Pxx_yy(file_path)
        latent_id = iid if iid is not None else None

        row = [
            mode,
            file_path,
            pid, iid, latent_id,
            float(dist), float(thr), int(ok),
            prompt_used, int(args.steps), float(args.guidance_scale),
            str(args.latent_mode), str(args.sigma_scheduler), float(sigma),
            int(seed_correct),
        ]
        detail_rows.append(row)

        total_cnt += 1
        ok_cnt += int(ok)

        # real-time terminal print for debugging
        print(f"[{total_cnt:05d}] {os.path.basename(file_path)}  dist={dist:.4f}  thr={thr:.4f}  ok={int(ok)}")

    # IMG: invert -> detect per image
    if args.mode in ("img", "both"):
        pbar = tqdm(img_paths, desc="[DETECT][IMG] invert+detect", dynamic_ncols=True)
        for p in pbar:
            im = Image.open(p).convert("RGB")
            lat = ddim_invert_to_noise_latent(
                pipe=pipe,
                image=im,
                steps=int(args.steps),
                prompt=str(detect_prompt),
                guidance_scale=float(args.guidance_scale),
                inv_scheduler=inv,
                dtype=dtype,
            ).detach()
            if args.latent_mode == "sigma_scaled":
                lat = lat * float(sigma)

            record_one("img", p, lat, str(detect_prompt))
            pbar.set_postfix(ok=f"{ok_cnt}/{total_cnt}", rate=f"{(ok_cnt/max(total_cnt,1)):.3f}")

    # PT: detect directly from zT per file
    if args.mode in ("pt", "both"):
        pbar = tqdm(pt_paths, desc="[DETECT][PT] detect", dynamic_ncols=True)
        for p in pbar:
            obj = torch.load(p, map_location=device)
            if torch.is_tensor(obj):
                zT = obj
            elif isinstance(obj, dict):
                found = None
                for k in ["zT", "z_t", "z_T", "latents", "noise", "z", "zT_bank"]:
                    if k in obj and torch.is_tensor(obj[k]):
                        found = obj[k]
                        break
                if found is None:
                    raise TypeError(f"Unsupported pt dict format: keys={list(obj.keys())}")
                zT = found
            else:
                raise TypeError(f"Unsupported pt format: {p}, type={type(obj)}")

            if zT.dim() == 3:
                zT = zT.unsqueeze(0)
            if zT.shape[0] != 1:
                zT = zT[:1]
            zT = zT.to(device=device, dtype=dtype)

            lat = zT
            if args.latent_mode == "sigma_scaled":
                lat = lat * float(sigma)

            record_one("pt", p, lat, "(pt)")
            pbar.set_postfix(ok=f"{ok_cnt}/{total_cnt}", rate=f"{(ok_cnt/max(total_cnt,1)):.3f}")

    detect_rate = float(ok_cnt / max(total_cnt, 1))

    # -------------------------
    # Output: Excel (summary + detect_result), optional JSON
    # -------------------------
    out_xlsx = args.out_xlsx
    if not out_xlsx:
        out_xlsx = os.path.join(img_dir, "treering_detect.xlsx")

    summary_rows = [
        ("img_dir", img_dir),
        ("model_id", args.model_id),
        ("mode", args.mode),
        ("n_total", total_cnt),
        ("n_detected", ok_cnt),
        ("detect_rate", detect_rate),
        ("dist_thr", float(thr)),
        ("prompt_mode", str(args.detect_prompt)),
        ("prompt_used", str(detect_prompt)),
        ("steps", int(args.steps)),
        ("guidance_scale", float(args.guidance_scale)),
        ("latent_mode", str(args.latent_mode)),
        ("sigma_scheduler", str(args.sigma_scheduler)),
        ("sigma", float(sigma)),
        # treering params
        ("tr_w_seed", int(args.tr_w_seed)),
        ("tr_seed_from_key", int(args.tr_seed_from_key)),
        ("tr_w_pattern", str(args.tr_w_pattern)),
        ("tr_w_mask_shape", str(args.tr_w_mask_shape)),
        ("tr_w_radius", int(args.tr_w_radius)),
        ("tr_w_channel", int(args.tr_w_channel)),
        ("mask_mode", str(args.mask_mode)),
    ]

    write_excel_two_sheets(
        out_xlsx=out_xlsx,
        summary_rows=summary_rows,
        detail_header=detail_header,
        detail_rows=detail_rows,
    )
    print(f"\n[OK] Saved Excel: {out_xlsx}")

    if args.out_json:
        out_json = args.out_json
    else:
        out_json = os.path.join(img_dir, f"treering_detect_{args.mode}.json")

    out_obj = {
        "summary": dict(summary_rows),
        "detail_header": detail_header,
        "detail_rows": detail_rows,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
