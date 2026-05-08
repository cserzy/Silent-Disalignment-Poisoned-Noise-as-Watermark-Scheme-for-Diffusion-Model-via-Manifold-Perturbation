#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GS detector with OMS inverse support.

One-piece script:
  image -> DDIM inversion (unknown prompt) -> zT in OMS domain
       -> OMS inverse restore -> zT in original GS domain
       -> decode GS bits from restored zT sign -> per-image result + CSV summary

Required CLI args:
  --model_id  --run_dir  --out_dir  --oms_q_pt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from Crypto.Cipher import ChaCha20
from PIL import Image
from tqdm import tqdm

from diffusers import DPMSolverMultistepScheduler

# These imports are expected from the official GS repository or the local project copy.
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from image_utils import transform_img

# Reuse OMS inverse logic from script-experiment/oms_repair_pt.py.
THIS_DIR = Path(__file__).resolve().parent
OMS_ROOT = THIS_DIR.parent
if str(OMS_ROOT) not in sys.path:
    sys.path.insert(0, str(OMS_ROOT))
try:
    from oms_repair_pt import (  # type: ignore
        flatten_latent_4d,
        load_q_pt,
        resolve_forward_aux_from_q_and_meta,
        run_apply_with_fallback,
        run_blended_inverse_with_fallback,
        unflatten_latent_2d,
    )
except Exception as e:
    raise RuntimeError(
        "Failed to import OMS helpers from script-experiment/oms_repair_pt.py. "
        f"Details: {e}"
    )


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


def _validate_zt_4d(z: torch.Tensor, tag: str) -> None:
    if not isinstance(z, torch.Tensor):
        raise SystemExit(f"[ERR] {tag} is not torch.Tensor (got {type(z)}).")
    if z.ndim != 4:
        raise SystemExit(
            f"[ERR] {tag} expects 4D [N,C,H,W], got shape={tuple(z.shape)}. "
            "Packed/non-4D latent is not supported."
        )


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def apply_oms_inverse_to_zt(
    zt_oms_4d: torch.Tensor,
    q_obj: Dict[str, Any],
    *,
    oms_q_pt: str,
    oms_meta_json: str,
    device: torch.device,
    verbose: bool,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """OMS-domain zT -> original GS-domain zT."""
    _validate_zt_4d(zt_oms_4d, "inversion zT")

    perm = q_obj["perm"]
    inv_perm = q_obj["inv_perm"]
    q_blocks = q_obj["q_blocks"]

    x_all = flatten_latent_4d(zt_oms_4d.to(torch.float32))
    _, d = x_all.shape
    if int(perm.numel()) != d:
        raise SystemExit(
            f"[ERR] OMS q dimension mismatch with inversion latent: q_D={perm.numel()}, latent_D={d}."
        )

    aux = resolve_forward_aux_from_q_and_meta(
        q_obj=q_obj,
        q_pt_path=oms_q_pt,
        q_meta_json=oms_meta_json,
    )
    alpha_eff = float(aux.get("blend_alpha", 1.0))
    match_eff = _to_bool(aux.get("match_target_std", False))
    rescale_factor = float(aux.get("rescale_factor", 1.0))

    if alpha_eff < 0.0 or alpha_eff > 1.0:
        raise SystemExit(f"[ERR] Invalid blend_alpha in OMS q/meta: {alpha_eff}")
    if match_eff and abs(rescale_factor) < 1e-12:
        raise SystemExit(
            f"[ERR] Invalid rescale_factor in OMS q/meta (too close to zero): {rescale_factor}"
        )

    # Undo global std match first when it was enabled in forward.
    x_unscaled = x_all / rescale_factor if match_eff else x_all

    solve_summary: Optional[Dict[str, Any]] = None
    if alpha_eff == 1.0:
        x_restored, used_device = run_apply_with_fallback(
            x_2d_cpu=x_unscaled,
            perm=perm,
            inv_perm=inv_perm,
            q_blocks=q_blocks,
            inverse=True,
            device=device,
            verbose=verbose,
        )
        inverse_mode = "pure_Q_inverse" if not match_eff else "blended_plus_rescale_inverse"
    else:
        x_restored, solve_summary, used_device = run_blended_inverse_with_fallback(
            y_2d_cpu=x_unscaled,
            perm=perm,
            inv_perm=inv_perm,
            q_blocks=q_blocks,
            alpha=alpha_eff,
            device=device,
            verbose=verbose,
        )
        inverse_mode = "blended_inverse" if not match_eff else "blended_plus_rescale_inverse"

    zt_restored = unflatten_latent_2d(x_restored, tuple(zt_oms_4d.shape))
    return zt_restored, {
        "inverse_mode": inverse_mode,
        "blend_alpha": float(alpha_eff),
        "match_target_std": bool(match_eff),
        "rescale_factor": float(rescale_factor),
        "solve_summary": solve_summary,
        "used_device": used_device,
        "param_source": aux.get("source", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True, help="Diffusers SD checkpoint path or hub id.")
    ap.add_argument("--run_dir", type=str, required=True, help="Generation run dir; script scans run_dir/sliced by default.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output dir (CSV + optional recovered latents).")
    ap.add_argument("--oms_q_pt", type=str, required=True, help="OMS q_pt file (e.g., oms_Q_GS.pt).")

    # Optional knobs
    ap.add_argument("--oms_meta_json", type=str, default="", help="Optional OMS meta json fallback for blend/std info.")
    ap.add_argument("--oms_block_mode", type=str, default="flat_chunk", choices=["flat_chunk"], help="OMS q block mode.")
    ap.add_argument("--inv_steps", type=int, default=50, help="DDIM inversion steps (default 50).")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="Compute dtype.")
    ap.add_argument("--max_images", type=int, default=0, help=">0: only process first N images.")
    ap.add_argument("--save_zt", action="store_true", help="Compat flag: save restored zT (same as --save_zt_restored).")
    ap.add_argument("--save_zt_oms", action="store_true", help="Save inversion zT before OMS inverse (OMS domain).")
    ap.add_argument("--save_zt_restored", action="store_true", help="Save OMS-restored zT used for GS decode.")
    ap.add_argument("--verbose", action="store_true", help="Print more logs.")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_lat_dir = out_dir / "latents"
    _ensure_dir(out_dir)

    if args.save_zt and not args.save_zt_restored:
        args.save_zt_restored = True
        print("[INFO] --save_zt is treated as --save_zt_restored in detect_GS_oms.py", flush=True)

    if args.save_zt_oms or args.save_zt_restored:
        _ensure_dir(out_lat_dir)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] model_id={args.model_id}", flush=True)
    print(f"[INFO] run_dir={run_dir}", flush=True)
    print(f"[INFO] out_dir={out_dir}", flush=True)
    print(f"[INFO] oms_q_pt={args.oms_q_pt}", flush=True)
    print(f"[INFO] oms_meta_json_provided={bool(str(args.oms_meta_json).strip())}", flush=True)
    print(f"[INFO] device={device} dtype={args.dtype}", flush=True)

    # Embedded GS params
    key32 = _parse_hex_bytes(GS_KEY_HEX_DEFAULT, 32, "gs_key")
    nonce12 = bytes(12) if GS_NONCE_ZERO_DEFAULT else None
    if nonce12 is None:
        raise RuntimeError("This detector embeds nonce_zero=True by default; update code if you need a custom nonce.")

    q_obj = load_q_pt(args.oms_q_pt, block_mode=args.oms_block_mode)
    oms_aux = resolve_forward_aux_from_q_and_meta(
        q_obj=q_obj,
        q_pt_path=args.oms_q_pt,
        q_meta_json=args.oms_meta_json,
    )
    oms_alpha = float(oms_aux.get("blend_alpha", 1.0))
    oms_match_std = _to_bool(oms_aux.get("match_target_std", False))
    oms_rescale = float(oms_aux.get("rescale_factor", 1.0))
    if oms_alpha == 1.0:
        inverse_hint = "pure_Q_inverse" if not oms_match_std else "blended_plus_rescale_inverse"
    else:
        inverse_hint = "blended_inverse" if not oms_match_std else "blended_plus_rescale_inverse"
    print(
        f"[INFO] OMS inverse defaults: mode={inverse_hint}, "
        f"blend_alpha={oms_alpha:.6f}, match_target_std={oms_match_std}, rescale_factor={oms_rescale:.8f}",
        flush=True,
    )

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
    out_csv = out_dir / "gs_detect_invert_oms_decode.csv"
    fieldnames = [
        "image",
        "image_path",
        "run_dir",
        "inv_steps",
        "zT_oms_path",
        "zT_restored_path",
        "C",
        "H",
        "W",
        "gs_seed",
        "gs_ch",
        "gs_hw",
        "pack_mode",
        "oms_q_pt",
        "oms_blend_alpha",
        "oms_match_target_std",
        "oms_rescale_factor",
        "bit_acc",
        "ber",
        "nbits",
        "detected",
    ]

    rows: List[Dict[str, object]] = []

    gt_cached = None
    gt_shape = None

    for idx, img_p in enumerate(tqdm(img_paths, desc="GS detect (invert+OMS-inverse+decode)"), start=1):
        img = Image.open(img_p).convert("RGB")
        zt_oms = invert_one_image_to_zT(pipe, img, inv_steps=args.inv_steps, device=device, dtype=dtype)
        zt_oms_cpu = zt_oms.detach().float().cpu()  # (1,C,H,W)
        print(f"[{idx:4d}/{len(img_paths)}] inversion done: {img_p.name} shape={tuple(zt_oms_cpu.shape)}", flush=True)

        zt_oms_path_str = ""
        if args.save_zt_oms:
            zt_oms_path = out_lat_dir / f"{img_p.stem}_inv_zT_oms.pt"
            torch.save(
                {
                    "zT": zt_oms_cpu,
                    "image": img_p.name,
                    "image_path": str(img_p),
                    "run_dir": str(run_dir),
                    "domain": "oms",
                    "oms_q_pt": str(args.oms_q_pt),
                },
                zt_oms_path,
            )
            zt_oms_path_str = str(zt_oms_path)

        zt_restored_cpu, inv_info = apply_oms_inverse_to_zt(
            zt_oms_cpu,
            q_obj=q_obj,
            oms_q_pt=args.oms_q_pt,
            oms_meta_json=args.oms_meta_json,
            device=device,
            verbose=args.verbose,
        )
        print(
            f"[{idx:4d}/{len(img_paths)}] OMS inverse: mode={inv_info['inverse_mode']} "
            f"alpha={inv_info['blend_alpha']:.6f} "
            f"match_std={inv_info['match_target_std']} "
            f"rescale={inv_info['rescale_factor']:.8f}",
            flush=True,
        )

        zt_restored_path_str = ""
        if args.save_zt_restored:
            zt_restored_path = out_lat_dir / f"{img_p.stem}_inv_zT_restored.pt"
            torch.save(
                {
                    "zT": zt_restored_cpu,
                    "image": img_p.name,
                    "image_path": str(img_p),
                    "run_dir": str(run_dir),
                    "domain": "restored_gs",
                    "oms_q_pt": str(args.oms_q_pt),
                    "oms_inverse_info": inv_info,
                },
                zt_restored_path,
            )
            zt_restored_path_str = str(zt_restored_path)

        _, C, H, W = map(int, zt_restored_cpu.shape)

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

        pred = decode_bits_from_zt_sign(zt_restored_cpu[0])
        acc, ber, nbits = bit_metrics(pred, gt_cached)

        detected = 1 if acc >= float(DETECT_THRESH_DEFAULT) else 0

        print(
            f"[{idx:4d}/{len(img_paths)}] {img_p.name}  detected={detected}  "
            f"bit_acc={acc:.6f}  ber={ber:.6f}",
            flush=True,
        )

        rows.append(
            {
                "image": img_p.name,
                "image_path": str(img_p),
                "run_dir": str(run_dir),
                "inv_steps": int(args.inv_steps),
                "zT_oms_path": zt_oms_path_str,
                "zT_restored_path": zt_restored_path_str,
                "C": C,
                "H": H,
                "W": W,
                "gs_seed": int(GS_SEED_DEFAULT),
                "gs_ch": int(GS_CH_DEFAULT),
                "gs_hw": int(GS_HW_DEFAULT),
                "pack_mode": GS_PACK_MODE_DEFAULT,
                "oms_q_pt": str(args.oms_q_pt),
                "oms_blend_alpha": float(inv_info.get("blend_alpha", oms_alpha)),
                "oms_match_target_std": bool(inv_info.get("match_target_std", oms_match_std)),
                "oms_rescale_factor": float(inv_info.get("rescale_factor", oms_rescale)),
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
    if args.save_zt_oms or args.save_zt_restored:
        print("[OK] saved zT files to:", str(out_lat_dir))


if __name__ == "__main__":
    main()
