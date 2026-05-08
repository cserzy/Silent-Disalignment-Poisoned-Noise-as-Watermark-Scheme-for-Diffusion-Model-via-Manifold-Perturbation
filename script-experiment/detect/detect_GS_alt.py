#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Alt Diffusion GS detector.

Minimal Alt-specific adaptation of detect_GS.py:
  image -> Alt VAE encode -> DDIM inverse with Alt UNet/text encoder (unknown prompt)
       -> recovered zT -> decode GS bits from zT sign
       -> per-image detection result + CSV + JSON summary

Notes:
  - This script keeps the GS bit construction / decode / metrics logic from detect_GS.py.
  - The inversion layer is Alt-specific and implemented with a minimal DDIM inverse path.
  - Safety checker is disabled by default because Alt experiments already showed it can black out images.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from Crypto.Cipher import ChaCha20
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL, DDIMInverseScheduler, PNDMScheduler, UNet2DConditionModel
from diffusers.pipelines.deprecated.alt_diffusion import AltDiffusionPipeline
from diffusers.pipelines.deprecated.alt_diffusion.modeling_roberta_series import (
    RobertaSeriesModelWithTransformation,
)
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, XLMRobertaTokenizer

from image_utils import transform_img


GS_KEY_HEX_DEFAULT = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
GS_NONCE_ZERO_DEFAULT = True
GS_SEED_DEFAULT = 12345
GS_CH_DEFAULT = 4
GS_HW_DEFAULT = 4
GS_PACK_MODE_DEFAULT = "official"
DETECT_THRESH_DEFAULT = 0.55

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _list_images_under_run_dir(run_dir: Path) -> List[Path]:
    cand = run_dir / "sliced"
    root = cand if cand.is_dir() else run_dir
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files = [p for p in files if "grid" not in p.name.lower()]
    return sorted(files)


def _parse_hex_bytes(hex_str: str, nbytes: int, name: str) -> bytes:
    b = bytes.fromhex(hex_str)
    if len(b) != nbytes:
        raise ValueError(f"{name} must be {nbytes} bytes ({2*nbytes} hex chars), got {len(b)} bytes")
    return b


def diffuse_bits_to_chw(base_bits: np.ndarray, C: int, H: int, W: int, fc: int, fhw: int) -> np.ndarray:
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
    flat = bits_chw.reshape(-1).astype(np.uint8)
    packed = np.packbits(flat)
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
    assert (C * H * W) % (fc * fhw * fhw) == 0, (C, H, W, fc, fhw)
    k_bits = (C * H * W) // (fc * fhw * fhw)
    rng = np.random.default_rng(int(gs_seed))
    base = rng.integers(0, 2, size=int(k_bits), dtype=np.int8)
    bits = diffuse_bits_to_chw(base, C=C, H=H, W=W, fc=fc, fhw=fhw)
    return chacha20_xor_bits_official(bits, key32=key32, nonce12=nonce12)


def decode_bits_from_zt_sign(zT: torch.Tensor) -> np.ndarray:
    z = zT.detach().cpu().float().numpy()
    return (z >= 0).astype(np.int8)


def bit_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, int]:
    assert pred.shape == gt.shape
    ok = pred == gt
    acc = float(ok.mean())
    ber = float((~ok).mean())
    return acc, ber, int(pred.size)


def build_alt_pipe(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    disable_safety_checker: bool = True,
) -> AltDiffusionPipeline:
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
        print("[load] safety_checker disabled")
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

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


@torch.no_grad()
def invert_one_image_to_zT_alt(
    pipe: AltDiffusionPipeline,
    img_pil: Image.Image,
    *,
    inv_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    prompt: str = "",
) -> torch.Tensor:
    """Approximate Alt inversion: VAE encode -> DDIM inverse with empty-prompt Alt UNet.

    This is not an official Alt inverse pipeline API. It reuses the standard DDIM inverse
    stepping pattern from project detection code and adapts it to Alt components.
    """
    img_t = transform_img(img_pil, target_size=512).unsqueeze(0).to(device=device, dtype=dtype)

    enc_dist = pipe.vae.encode(img_t).latent_dist
    z0 = enc_dist.mode() * pipe.vae.config.scaling_factor

    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        negative_prompt=None,
    )

    inv_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inv_scheduler.set_timesteps(int(inv_steps), device=device)

    lat = z0
    for t in inv_scheduler.timesteps:
        noise_pred = pipe.unet(
            lat,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        lat = inv_scheduler.step(noise_pred, t, lat, return_dict=False)[0]

    return lat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True, help="Alt Diffusion checkpoint path.")
    ap.add_argument("--run_dir", type=str, required=True, help="Generation run dir; scans run_dir/sliced by default.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output dir (CSV + summary + optional latents).")

    ap.add_argument("--inv_steps", type=int, default=50, help="DDIM inverse steps (default 50).")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="Compute dtype.")
    ap.add_argument("--max_images", type=int, default=0, help=">0: only process first N images.")
    ap.add_argument("--save_zt", action="store_true", help="Save recovered zT (.pt) under out_dir/latents.")
    ap.add_argument("--disable_safety_checker", dest="disable_safety_checker", action="store_true",
                    help="Disable safety_checker and feature_extractor.")
    ap.add_argument("--enable_safety_checker", dest="disable_safety_checker", action="store_false",
                    help="Enable safety_checker and feature_extractor.")
    ap.add_argument("--alt_force_manual_pipe", action="store_true",
                    help="Reserved flag for clarity; this script always uses manual Alt pipe loading.")
    ap.set_defaults(disable_safety_checker=True)

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_lat_dir = out_dir / "latents"
    _ensure_dir(out_dir)
    if args.save_zt:
        _ensure_dir(out_lat_dir)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    key32 = _parse_hex_bytes(GS_KEY_HEX_DEFAULT, 32, "gs_key")
    nonce12 = bytes(12) if GS_NONCE_ZERO_DEFAULT else None
    if nonce12 is None:
        raise RuntimeError("This detector embeds nonce_zero=True by default; update code if you need a custom nonce.")

    pipe = build_alt_pipe(
        args.model_id,
        device=device,
        dtype=dtype,
        disable_safety_checker=bool(args.disable_safety_checker),
    )
    print(
        f"[pipe] class={pipe.__class__.__name__} "
        f"safety_checker_enabled={pipe.safety_checker is not None} "
        f"vae_scale_factor={getattr(pipe, 'vae_scale_factor', 'NA')}"
    )

    img_paths = _list_images_under_run_dir(run_dir)
    if args.max_images and args.max_images > 0:
        img_paths = img_paths[: args.max_images]
    if not img_paths:
        raise SystemExit(f"[ERR] No images found under: {run_dir} (or {run_dir / 'sliced'})")

    out_csv = out_dir / "gs_detect_invert_decode_alt.csv"
    out_summary = out_dir / "gs_detect_invert_decode_alt_summary.json"
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
        "invert_mode",
        "approx_inverse",
    ]

    rows: List[Dict[str, object]] = []
    gt_cached = None
    gt_shape = None

    for idx, img_p in enumerate(tqdm(img_paths, desc="GS detect Alt (invert+decode)"), start=1):
        img = Image.open(img_p).convert("RGB")
        zT = invert_one_image_to_zT_alt(pipe, img, inv_steps=args.inv_steps, device=device, dtype=dtype)
        zT_cpu = zT.detach().float().cpu()
        _, C, H, W = map(int, zT_cpu.shape)

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
            print(f"[gs] cached expected bits for shape={(C, H, W)}")

        zT_path_str = ""
        if args.save_zt:
            zT_path = out_lat_dir / (img_p.stem + "_inv_zT_alt.pt")
            torch.save(
                {
                    "zT": zT_cpu,
                    "image": img_p.name,
                    "image_path": str(img_p),
                    "run_dir": str(run_dir),
                    "invert_mode": "alt_ddim_inverse_empty_prompt",
                    "approx_inverse": True,
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
            f"bit_acc={acc:.6f}  ber={ber:.6f}  zT_shape={(1, C, H, W)}"
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
                "invert_mode": "alt_ddim_inverse_empty_prompt",
                "approx_inverse": 1,
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
    summary = {
        "model_id": args.model_id,
        "run_dir": str(run_dir),
        "out_csv": str(out_csv),
        "files": len(rows),
        "inv_steps": int(args.inv_steps),
        "dtype": args.dtype,
        "disable_safety_checker": bool(args.disable_safety_checker),
        "invert_mode": "alt_ddim_inverse_empty_prompt",
        "approx_inverse": True,
        "bit_acc_mean": float(accs.mean()),
        "bit_acc_min": float(accs.min()),
        "ber_mean": float(bers.mean()),
        "ber_max": float(bers.max()),
        "detected_rate": float(dets.mean()),
        "detect_thresh": float(DETECT_THRESH_DEFAULT),
    }
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[OK] wrote:", str(out_csv))
    print("[OK] wrote:", str(out_summary))
    print(
        f"[Summary] files={len(rows)} | bit_acc mean={accs.mean():.6f} min={accs.min():.6f} "
        f"| ber mean={bers.mean():.6f} max={bers.max():.6f} "
        f"| detected rate={dets.mean():.4f} (thresh={DETECT_THRESH_DEFAULT})"
    )
    if args.save_zt:
        print("[OK] saved recovered zT to:", str(out_lat_dir))


if __name__ == "__main__":
    main()
