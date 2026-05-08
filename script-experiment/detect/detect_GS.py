#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GS (Gaussian Shading, qbin/sign-bit) end-to-end detector.

One-piece script:
  image -> DDIM inversion (unknown prompt) -> optional save recovered zT (.pt)
       -> decode GS bits from zT sign -> per-image detection result + CSV summary

As requested, the *only required* CLI args are:
  --model_id  --run_dir  --out_dir

All GS watermark parameters are embedded below as defaults for the current experiment setup.

Implementation notes:
  - Inversion uses official-style unknown-prompt mode: prompt="", guidance_scale=1.0.
  - Bit packing uses NumPy packbits/unpackbits default (MSB-first, bitorder='big').

Dependencies (expected to be available in the local project environment):
  - inverse_stable_diffusion.py (InversableStableDiffusionPipeline)
  - image_utils.py (transform_img)
  - diffusers, torch, numpy, pycryptodome, pillow
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from Crypto.Cipher import ChaCha20
from PIL import Image
from tqdm import tqdm

from diffusers import DPMSolverMultistepScheduler

# These imports are expected from the official GS repository or the local project copy.
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from image_utils import transform_img


# ---------------------------
# Embedded GS watermark defaults aligned with the current experiment setup
# ---------------------------

GS_KEY_HEX_DEFAULT = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"  # 32B (64 hex)
GS_NONCE_ZERO_DEFAULT = True  # 12B all-zero nonce
GS_SEED_DEFAULT = 12345
GS_CH_DEFAULT = 4
GS_HW_DEFAULT = 4
GS_PACK_MODE_DEFAULT = "official"  # np.packbits/unpackbits default (bitorder='big')
DETECT_THRESH_DEFAULT = 0.60


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ---------------------------
# IO helpers
# ---------------------------


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _list_images_under_run_dir(run_dir: Path) -> List[Path]:
    """Prefer run_dir/sliced; fallback to run_dir if no sliced exists."""
    cand = run_dir / "sliced"
    root = cand if cand.is_dir() else run_dir
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    # filter typical grid images
    files = [p for p in files if "grid" not in p.name.lower()]
    return sorted(files)


def _parse_hex_bytes(hex_str: str, nbytes: int, name: str) -> bytes:
    b = bytes.fromhex(hex_str)
    if len(b) != nbytes:
        raise ValueError(f"{name} must be {nbytes} bytes ({2*nbytes} hex chars), got {len(b)} bytes")
    return b


# ---------------------------
# GS bit construction helpers
# ---------------------------


def diffuse_bits_to_chw(base_bits: np.ndarray, C: int, H: int, W: int, fc: int, fhw: int) -> np.ndarray:
    """Spread base_bits over (C,H,W) by tiling.

    base -> reshape (C/fc, H/fhw, W/fhw)
        -> tile channels by fc
        -> tile spatial by fhw x fhw
    """
    assert base_bits.ndim == 1
    assert C % fc == 0 and H % fhw == 0 and W % fhw == 0, (C, H, W, fc, fhw)
    c0 = C // fc
    h0 = H // fhw
    w0 = W // fhw
    assert base_bits.size == c0 * h0 * w0, (base_bits.size, c0, h0, w0)
    small = base_bits.reshape(c0, h0, w0)
    bits_c = np.tile(small, (fc, 1, 1))
    bits_chw = np.tile(bits_c, (1, fhw, fhw))
    return bits_chw.astype(np.int8)


def chacha20_xor_bits_official(bits_chw: np.ndarray, key32: bytes, nonce12: bytes) -> np.ndarray:
    """Official-style: packbits/unpackbits MSB-first + ChaCha20.encrypt()."""
    flat = bits_chw.reshape(-1).astype(np.uint8)
    packed = np.packbits(flat)  # bitorder='big' by default
    cipher = ChaCha20.new(key=key32, nonce=nonce12)
    out_bytes = cipher.encrypt(packed.tobytes())
    out_bits = np.unpackbits(np.frombuffer(out_bytes, dtype=np.uint8))[: flat.size]
    return out_bits.astype(np.int8).reshape(bits_chw.shape)


def expected_bits(
    C: int,
    H: int,
    W: int,
    *,
    gs_seed: int,
    fc: int,
    fhw: int,
    key32: bytes,
    nonce12: bytes,
) -> np.ndarray:
    """Return expected GS bits as (C,H,W) in {0,1}."""
    assert (C * H * W) % (fc * fhw * fhw) == 0, (C, H, W, fc, fhw)
    k_bits = (C * H * W) // (fc * fhw * fhw)
    rng = np.random.default_rng(int(gs_seed))
    base = rng.integers(0, 2, size=int(k_bits), dtype=np.int8)
    bits = diffuse_bits_to_chw(base, C=C, H=H, W=W, fc=fc, fhw=fhw)
    return chacha20_xor_bits_official(bits, key32=key32, nonce12=nonce12)


def decode_bits_from_zt_sign(zT: torch.Tensor) -> np.ndarray:
    """Decode bits from zT by sign: bit=1 iff z>=0."""
    z = zT.detach().cpu().float().numpy()
    return (z >= 0).astype(np.int8)


def bit_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, int]:
    assert pred.shape == gt.shape
    ok = (pred == gt)
    acc = float(ok.mean())
    ber = float((~ok).mean())
    return acc, ber, int(pred.size)


# ---------------------------
# DDIM inversion
# ---------------------------


@torch.inference_mode()
def invert_one_image_to_zT(
    pipe: InversableStableDiffusionPipeline,
    img_pil: Image.Image,
    *,
    inv_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return recovered noisy latent zT via official-style forward diffusion (DDIM inversion)."""
    img_t = transform_img(img_pil, target_size=512).unsqueeze(0).to(device=device, dtype=dtype)
    z0 = pipe.get_image_latents(img_t, sample=False)
    text_embeddings = pipe.get_text_embedding("")  # unknown prompt mode
    zT = pipe.forward_diffusion(
        latents=z0,
        text_embeddings=text_embeddings,
        guidance_scale=1.0,
        num_inference_steps=int(inv_steps),
    )
    return zT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True, help="Diffusers SD checkpoint path or hub id.")
    ap.add_argument("--run_dir", type=str, required=True, help="Generation run dir; script scans run_dir/sliced by default.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output dir (CSV + optional recovered latents).")

    # Optional parameters not required by the default pipeline
    ap.add_argument("--inv_steps", type=int, default=50, help="DDIM inversion steps (default 50).")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="Compute dtype.")
    ap.add_argument("--max_images", type=int, default=0, help=">0: only process first N images.")
    ap.add_argument("--save_zt", action="store_true", help="Save recovered zT (.pt) under out_dir/latents.")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_lat_dir = out_dir / "latents"
    _ensure_dir(out_dir)
    if args.save_zt:
        _ensure_dir(out_lat_dir)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Embedded GS params
    key32 = _parse_hex_bytes(GS_KEY_HEX_DEFAULT, 32, "gs_key")
    nonce12 = bytes(12) if GS_NONCE_ZERO_DEFAULT else None
    if nonce12 is None:
        raise RuntimeError("This detector embeds nonce_zero=True by default; update code if you need a custom nonce.")

    # Load pipeline
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    img_paths = _list_images_under_run_dir(run_dir)
    if args.max_images and args.max_images > 0:
        img_paths = img_paths[: args.max_images]
    if not img_paths:
        raise SystemExit(f"[ERR] No images found under: {run_dir} (or {run_dir/'sliced'})")

    # Output CSV
    out_csv = out_dir / "gs_detect_invert_decode.csv"
    fieldnames = [
        "image",
        "image_path",
        "run_dir",
        "inv_steps",
        "zT_path",
        "C",
        "H",
        "W",
        "gs_seed",
        "gs_ch",
        "gs_hw",
        "pack_mode",
        "bit_acc",
        "ber",
        "nbits",
        "detected",
    ]

    rows: List[Dict[str, object]] = []

    gt_cached = None
    gt_shape = None

    for idx, img_p in enumerate(tqdm(img_paths, desc="GS detect (invert+decode)"), start=1):
        img = Image.open(img_p).convert("RGB")
        zT = invert_one_image_to_zT(pipe, img, inv_steps=args.inv_steps, device=device, dtype=dtype)
        zT_cpu = zT.detach().float().cpu()  # (1,C,H,W)
        _, C, H, W = map(int, zT_cpu.shape)

        # cache expected bits once per (C,H,W)
        if gt_cached is None or gt_shape != (C, H, W):
            gt_cached = expected_bits(
                C=C,
                H=H,
                W=W,
                gs_seed=int(GS_SEED_DEFAULT),
                fc=int(GS_CH_DEFAULT),
                fhw=int(GS_HW_DEFAULT),
                key32=key32,
                nonce12=nonce12,
            )
            gt_shape = (C, H, W)

        zT_path_str = ""
        if args.save_zt:
            zT_path = out_lat_dir / (img_p.stem + "_inv_zT.pt")
            torch.save(
                {
                    "zT": zT_cpu,
                    "image": img_p.name,
                    "image_path": str(img_p),
                    "run_dir": str(run_dir),
                    "gs": {
                        "seed": int(GS_SEED_DEFAULT),
                        "ch": int(GS_CH_DEFAULT),
                        "hw": int(GS_HW_DEFAULT),
                        "key_hex": GS_KEY_HEX_DEFAULT.lower(),
                        "nonce_hex": "00" * 12,
                        "pack_order": GS_PACK_MODE_DEFAULT,
                    },
                },
                zT_path,
            )
            zT_path_str = str(zT_path)

        pred = decode_bits_from_zt_sign(zT_cpu[0])
        acc, ber, nbits = bit_metrics(pred, gt_cached)

        detected = 1 if acc >= float(DETECT_THRESH_DEFAULT) else 0

        print(
            f"[{idx:4d}/{len(img_paths)}] {img_p.name}  detected={detected}  "
            f"bit_acc={acc:.6f}  ber={ber:.6f}"
        )

        rows.append(
            {
                "image": img_p.name,
                "image_path": str(img_p),
                "run_dir": str(run_dir),
                "inv_steps": int(args.inv_steps),
                "zT_path": zT_path_str,
                "C": C,
                "H": H,
                "W": W,
                "gs_seed": int(GS_SEED_DEFAULT),
                "gs_ch": int(GS_CH_DEFAULT),
                "gs_hw": int(GS_HW_DEFAULT),
                "pack_mode": GS_PACK_MODE_DEFAULT,
                "bit_acc": acc,
                "ber": ber,
                "nbits": nbits,
                "detected": detected,
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    accs = np.array([float(r["bit_acc"]) for r in rows], dtype=np.float64)
    bers = np.array([float(r["ber"]) for r in rows], dtype=np.float64)
    dets = np.array([int(r["detected"]) for r in rows], dtype=np.int32)

    print("\n[OK] wrote:", str(out_csv))
    print(
        f"[Summary] files={len(rows)} | bit_acc mean={accs.mean():.6f} min={accs.min():.6f} "
        f"| ber mean={bers.mean():.6f} max={bers.max():.6f} "
        f"| detected rate={dets.mean():.4f} (thresh={DETECT_THRESH_DEFAULT})"
    )
    if args.save_zt:
        print("[OK] saved recovered zT to:", str(out_lat_dir))


if __name__ == "__main__":
    main()
