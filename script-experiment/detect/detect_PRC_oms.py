#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Detect PRC GLOBAL watermarks from images/zT, with OMS inverse before PRC decode.

Assumptions
- Images: <run_dir>/sliced/*.png (no recursion)
- Keys:   <run_dir>/wm_meta/prc_keys.pkl (preferred)
          (also supports a few fallback filenames)
- Prompt: by default uses empty prompt (official default). Optionally can use the
          first prompt in manifest.csv as a fixed prompt for all images.

Batching
- --inv_bs controls how many images are inverted per chunk.

Outputs
- A CSV with per-item detection results, and prints realtime progress.

Notes
- This script expects PRC-Watermark official repo code structure. You can pass
  --prc_repo to add that repo to sys.path so that `import src.*` works.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import sys
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image


# Reuse OMS inverse helpers from script-experiment/oms_repair_pt.py
_SCRIPT_DIR = Path(__file__).resolve().parent
_OMS_ROOT = _SCRIPT_DIR.parent
if str(_OMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_OMS_ROOT))
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

# -----------------------------
# Path helpers / imports
# -----------------------------
def _resolve_existing_dir(p: str, name: str) -> str:
    rp = str(Path(p).expanduser().resolve())
    if not os.path.isdir(rp):
        raise FileNotFoundError(f"{name} not found: {rp}")
    return rp


def _resolve_existing_file(p: str, name: str) -> str:
    rp = str(Path(p).expanduser().resolve())
    if not os.path.isfile(rp):
        raise FileNotFoundError(f"{name} not found: {rp}")
    return rp


def _infer_project_root(run_dir: Optional[str], zt_path: Optional[str]) -> str:
    seeds: List[Path] = [Path(__file__).resolve().parents[2]]
    if run_dir:
        seeds.append(Path(run_dir).expanduser().resolve())
    if zt_path:
        seeds.append(Path(zt_path).expanduser().resolve().parent)
    for seed in seeds:
        for cand in [seed, *seed.parents]:
            if (cand / "script-experiment").is_dir():
                return str(cand)
    return str(seeds[0])


def _is_prc_repo_candidate(base_dir: str) -> bool:
    base = Path(base_dir)
    cands = [
        base / "src" / "inverse_stable_diffusion.py",
        base / "inverse_stable_diffusion.py",
    ]
    return any(p.is_file() for p in cands)


def _resolve_prc_repo(prc_repo_arg: Optional[str], project_root: str) -> Tuple[Optional[str], List[str]]:
    tried: List[str] = []
    if prc_repo_arg:
        p = str(Path(prc_repo_arg).expanduser().resolve())
        tried.append(p)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"--prc_repo not found: {p}")
        if not _is_prc_repo_candidate(p):
            raise FileNotFoundError(
                f"--prc_repo exists but missing PRC modules (inverse_stable_diffusion.py): {p}"
            )
        return p, tried

    script_dir = str(Path(__file__).resolve().parent)
    cands = [
        script_dir,
        str(Path(project_root) / "script-experiment" / "detect"),
        str(Path(project_root) / "third_party" / "PRC-Watermark-main"),
        str(Path(project_root) / "PRC-Watermark-main"),
    ]
    for c in cands:
        cc = str(Path(c).expanduser().resolve())
        tried.append(cc)
        if os.path.isdir(cc) and _is_prc_repo_candidate(cc):
            return cc, tried
    return None, tried


def _add_prc_repo_to_syspath(prc_repo: Optional[str]) -> None:
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    if prc_repo and os.path.isdir(prc_repo):
        if prc_repo not in sys.path:
            sys.path.insert(0, prc_repo)
        src = os.path.join(prc_repo, "src")
        if os.path.isdir(src) and src not in sys.path:
            sys.path.insert(0, src)

_PRC_IMPORT_CACHE: Optional[Tuple[Any, Any, Any]] = None


def _import_official_modules():
    """
    Import PRC official modules with minimal dependencies for detection path.

    Heavy training/data modules (e.g. optim_utils -> datasets -> aiohttp) are
    intentionally NOT imported here.

    Returns:
      InversableStableDiffusionPipeline, prc_lib, prc_gaussians
    """
    global _PRC_IMPORT_CACHE
    if _PRC_IMPORT_CACHE is not None:
        return _PRC_IMPORT_CACHE

    inv_errs: List[str] = []
    prc_errs: List[str] = []
    gauss_errs: List[str] = []

    try:
        try:
            from src.inverse_stable_diffusion import InversableStableDiffusionPipeline
            inv_from = "src.inverse_stable_diffusion"
        except Exception as e1:
            inv_errs.append(repr(e1))
            from inverse_stable_diffusion import InversableStableDiffusionPipeline  # type: ignore
            inv_from = "inverse_stable_diffusion"
    except Exception as e2:
        inv_errs.append(repr(e2))
        raise RuntimeError(
            "Failed to import required PRC inversion module "
            "(InversableStableDiffusionPipeline). Tried: "
            "src.inverse_stable_diffusion, inverse_stable_diffusion. "
            f"Errors: {' | '.join(inv_errs)}"
        )

    try:
        try:
            import prc as prc_lib  # type: ignore
            prc_from = "prc"
        except Exception as e1:
            prc_errs.append(repr(e1))
            from src import prc as prc_lib  # type: ignore
            prc_from = "src.prc"
    except Exception as e2:
        prc_errs.append(repr(e2))
        raise RuntimeError(
            "Failed to import required PRC decode module (prc). Tried: prc, src.prc. "
            f"Errors: {' | '.join(prc_errs)}"
        )

    try:
        try:
            import pseudogaussians as prc_gaussians  # type: ignore
            gauss_from = "pseudogaussians"
        except Exception as e1:
            gauss_errs.append(repr(e1))
            from src import pseudogaussians as prc_gaussians  # type: ignore
            gauss_from = "src.pseudogaussians"
    except Exception as e2:
        gauss_errs.append(repr(e2))
        raise RuntimeError(
            "Failed to import required PRC posterior module (pseudogaussians). "
            "Tried: pseudogaussians, src.pseudogaussians. "
            f"Errors: {' | '.join(gauss_errs)}"
        )

    print(
        "[INFO] imported PRC modules:",
        f"inversion={inv_from}, prc={prc_from}, gaussians={gauss_from}",
        flush=True,
    )
    print(
        "[INFO] skipped heavy modules: optim_utils (unused in PRC detect path)",
        flush=True,
    )

    _PRC_IMPORT_CACHE = (InversableStableDiffusionPipeline, prc_lib, prc_gaussians)
    return _PRC_IMPORT_CACHE


# -----------------------------
# Key loading
# -----------------------------

def _load_prc_keys(meta_dir: str, key_path: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
    """Load PRC decoding key, plus meta.

    We prefer a single combined pickle (to match official workflows): prc_keys.pkl
    Expected shapes:
      - dict with keys like {'encoding_key':..., 'decoding_key':...}
      - tuple/list (encoding_key, decoding_key)
      - or directly a decoding_key object
    """
    tried: List[str] = []

    # preferred combined key
    key_cands = []
    if key_path:
        key_cands.append(str(Path(key_path).expanduser().resolve()))
    key_cands += [
        os.path.join(meta_dir, "prc_keys.pkl"),
        os.path.join(meta_dir, "keys.pkl"),
        os.path.join(meta_dir, "prc_decoding_key.pkl"),
        os.path.join(meta_dir, "decoding_key.pkl"),
        os.path.join(meta_dir, "prc_key.pkl"),
    ]

    obj = None
    chosen = None
    for p in key_cands:
        tried.append(p)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                obj = pickle.load(f)
            chosen = p
            break

    if obj is None:
        raise FileNotFoundError(
            "Missing PRC key file. Tried:\n  " + "\n  ".join(tried)
        )

    decoding_key = None
    if isinstance(obj, dict):
        # common dict layouts
        for k in ("decoding_key", "decoding", "dk"):
            if k in obj:
                decoding_key = obj[k]
                break
        if decoding_key is None and "encoding_key" in obj and "decoding_key" in obj:
            decoding_key = obj["decoding_key"]
    elif isinstance(obj, (tuple, list)) and len(obj) >= 2:
        # assume (encoding, decoding)
        decoding_key = obj[1]
    else:
        # assume it's already the decoding key
        decoding_key = obj

    if decoding_key is None:
        raise RuntimeError(f"Loaded {chosen} but could not extract decoding_key.")

    meta: Dict[str, Any] = {"key_file": chosen}

    # optional meta json produced by your generator
    for mp in [
        os.path.join(meta_dir, "prc_global_meta.json"),
        os.path.join(meta_dir, "prc_meta.json"),
        os.path.join(meta_dir, "meta.json"),
    ]:
        if os.path.isfile(mp):
            try:
                with open(mp, "r", encoding="utf-8") as f:
                    meta.update(json.load(f))
            except Exception:
                pass

    return decoding_key, meta


def _infer_key_n_and_target_size(decoding_key) -> Tuple[Optional[int], Optional[int], Optional[Tuple[int, int]]]:
    """
    Infer expected flattened dimension n and target image size for inversion from decoding_key.

    decoding_key in PRC official code is usually a tuple like:
      (generator_matrix, parity_check_matrix, one_time_pad, fpr, noise_rate, test_bits, g, max_bp_iter, t)
    Some forks store generator_matrix as (k, n), some as (n, k). We infer n robustly.
    """
    try:
        gen = decoding_key[0]
        shp = tuple(getattr(gen, "shape", ()))
        if len(shp) >= 2:
            # robust to both (k,n) and (n,k) layouts
            n = int(max(int(shp[0]), int(shp[1])))
        else:
            n = None
    except Exception:
        n = None

    # For Stable Diffusion latent size, n = 4*(H/8)*(W/8)
    # If n is known, guess the closest square-like size among common: 512-> 4*64*64=16384, 768->4*96*96=36864
    hlat = None
    target = None
    if n is not None:
        if n == 4 * 64 * 64:
            hlat = 64
            target = (512, 512)
        elif n == 4 * 96 * 96:
            hlat = 96
            target = (768, 768)
        else:
            # attempt infer hlat assuming square: n = 4*hlat*hlat
            import math
            h = int(round(math.sqrt(n / 4)))
            if 4 * h * h == n:
                hlat = h
                target = (h * 8, h * 8)

    return n, hlat, target


def _resolve_meta_dir(
    wm_meta_dir: Optional[str],
    meta_root: Optional[str],
    run_dir: Optional[str],
    zt_dir: Optional[str],
    project_root: str,
) -> Tuple[str, List[str]]:
    tried: List[str] = []

    explicit = wm_meta_dir or meta_root
    if explicit:
        p = str(Path(explicit).expanduser().resolve())
        tried.append(p)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"wm_meta/meta dir not found: {p}")
        return p, tried

    cands: List[str] = []
    if run_dir:
        cands.extend(
            [
                os.path.join(run_dir, "wm_meta"),
                os.path.join(run_dir, "latents_experiment", "wm_meta"),
            ]
        )
    if zt_dir:
        cands.extend(
            [
                os.path.join(zt_dir, "wm_meta"),
                os.path.join(Path(zt_dir).parent, "wm_meta"),
            ]
        )
    cands.extend(
        [
            os.path.join(project_root, "latents_experiment", "wm_meta"),
            os.path.join(project_root, "wm_meta"),
        ]
    )
    for c in cands:
        cc = str(Path(c).expanduser().resolve())
        tried.append(cc)
        if os.path.isdir(cc):
            return cc, tried

    raise FileNotFoundError(
        "Could not resolve wm_meta dir. Tried:\n  " + "\n  ".join(tried)
    )


def _resolve_manifest_csv(run_dir: str) -> Tuple[Optional[str], List[str]]:
    tried = [
        str(Path(run_dir) / "sliced" / "manifest.csv"),
        str(Path(run_dir) / "manifest.csv"),
    ]
    for p in tried:
        if os.path.isfile(p):
            return p, tried
    return None, tried


def _ensure_parent_dir(path_like: str) -> str:
    out = str(Path(path_like).expanduser().resolve())
    parent = str(Path(out).parent)
    try:
        os.makedirs(parent, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Cannot create output directory: {parent}. Error: {repr(e)}")
    return out


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int,)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def collect_images_from_run_dir(run_dir: str) -> List[str]:
    """Prefer run_dir/sliced/*.png, then run_dir/*.png, then recursive *.png."""
    rd = Path(run_dir)
    if not rd.is_dir():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    sliced = sorted(str(p) for p in (rd / "sliced").glob("*.png"))
    if sliced:
        return sliced
    direct = sorted(str(p) for p in rd.glob("*.png"))
    if direct:
        return direct
    return sorted(str(p) for p in rd.rglob("*.png"))


def apply_oms_inverse_to_latent(
    zt_oms: torch.Tensor,
    q_obj: Dict[str, Any],
    *,
    oms_q_pt: str,
    oms_meta_json: str,
    device: torch.device,
    verbose: bool,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Restore latent from OMS domain to original PRC domain.
    Reuses the same inverse chain as oms_repair_pt.py.
    """
    if not torch.is_tensor(zt_oms):
        raise TypeError(f"Expected tensor zt_oms, got {type(zt_oms)}")
    if zt_oms.ndim != 4:
        raise ValueError(f"Expected 4D latent [N,C,H,W], got shape={tuple(zt_oms.shape)}")

    x_all = flatten_latent_4d(zt_oms.to(torch.float32))
    perm = q_obj["perm"]
    inv_perm = q_obj["inv_perm"]
    q_blocks = q_obj["q_blocks"]

    if int(perm.numel()) != int(x_all.shape[1]):
        raise ValueError(f"OMS q dimension mismatch: q_D={perm.numel()}, latent_D={x_all.shape[1]}")

    aux = resolve_forward_aux_from_q_and_meta(
        q_obj=q_obj,
        q_pt_path=oms_q_pt,
        q_meta_json=oms_meta_json,
    )
    alpha_eff = float(aux.get("blend_alpha", 1.0))
    match_eff = _to_bool(aux.get("match_target_std", False))
    rescale_factor = float(aux.get("rescale_factor", 1.0))

    if not (0.0 <= alpha_eff <= 1.0):
        raise ValueError(f"Invalid OMS blend_alpha={alpha_eff}")
    if match_eff and abs(rescale_factor) < 1e-12:
        raise ValueError(f"Invalid OMS rescale_factor={rescale_factor}")

    # Undo forward std scaling first.
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

    zt_restored = unflatten_latent_2d(x_restored, tuple(zt_oms.shape))
    info = {
        "inverse_mode": inverse_mode,
        "blend_alpha": float(alpha_eff),
        "match_target_std": bool(match_eff),
        "rescale_factor": float(rescale_factor),
        "solve_summary": solve_summary,
        "used_device": used_device,
        "param_source": aux.get("source", {}),
    }
    return zt_restored, info


# -----------------------------
# SD pipe + inversion (image mode)
# -----------------------------

def _resolve_torch_dtype(dtype_str: str) -> torch.dtype:
    s = (dtype_str or "fp32").lower()
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def _pil_to_latent_input_batch(
    images: List[Image.Image],
    target_size: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Convert PIL images to latent-input tensor expected by VAE/decoder_inv:
      shape [B,3,H,W], value range [-1,1].
    """
    import numpy as np

    if not images:
        raise ValueError("images must be a non-empty list")

    if target_size is not None:
        width = int(target_size[0])
        height = int(target_size[1])
    else:
        width, height = images[0].size

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid target_size resolved: {(width, height)}")

    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.BICUBIC  # Pillow>=9
    else:
        resample = Image.BICUBIC

    arrs: List[Any] = []
    for im in images:
        if not isinstance(im, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image in images, got {type(im)}")
        if im.mode != "RGB":
            im = im.convert("RGB")
        if im.size != (width, height):
            im = im.resize((width, height), resample=resample)
        a = np.asarray(im, dtype=np.float32) / 255.0  # [H,W,3], [0,1]
        a = a * 2.0 - 1.0  # [-1,1]
        a = np.transpose(a, (2, 0, 1))  # [3,H,W]
        arrs.append(a)

    batch = np.stack(arrs, axis=0)  # [B,3,H,W]
    t = torch.from_numpy(batch)
    if (device is not None) or (dtype is not None):
        t = t.to(device=device if device is not None else t.device,
                 dtype=dtype if dtype is not None else t.dtype)
    return t


def _make_pipe(model_id: str, torch_dtype: torch.dtype, cache_dir: Optional[str], solver_order: int):
    from diffusers import DPMSolverMultistepScheduler

    pipe = _import_official_modules()[0].from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Version-compatible scheduler init: reuse existing scheduler config.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        solver_order=int(solver_order),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def exact_inversion_batch(
    pipe,
    images: List[Image.Image],
    prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    inv_order: int = 0,
    decoder_inv: bool = True,
    target_size: Optional[Tuple[int, int]] = None,
    debug: bool = False,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Invert images back to latents using official inverse pipeline.

    Returns: latents tensor (B,4,64,64) on CUDA.
    """
    device = pipe.device
    imgs_t = _pil_to_latent_input_batch(
        images=images,
        target_size=target_size,
        device=device,
        dtype=torch.float32,
    )

    # Align inversion input with VAE weights to avoid conv dtype/device mismatch.
    try:
        vae_param = next(pipe.vae.parameters())
        vae_device = vae_param.device
        vae_dtype = vae_param.dtype
    except StopIteration:
        vae_device = device
        vae_dtype = imgs_t.dtype
    imgs_t = imgs_t.to(device=vae_device, dtype=vae_dtype)

    if verbose:
        already = bool(getattr(exact_inversion_batch, "_printed_dtype_align", False))
        if not already:
            branch = "decoder_inv(torch.enable_grad)" if decoder_inv else "inverse(torch.no_grad)"
            print(
                "[INV][verbose] dtype/device align: "
                f"imgs_t_shape={tuple(imgs_t.shape)}, "
                f"imgs_t={imgs_t.dtype}/{imgs_t.device}, "
                f"vae={vae_dtype}/{vae_device}, "
                f"branch={branch}",
                flush=True,
            )
            setattr(exact_inversion_batch, "_printed_dtype_align", True)

    if debug:
        print(f"[DBG] imgs_t.shape={tuple(imgs_t.shape)} dtype={imgs_t.dtype} device={imgs_t.device}")

    # decoder_inv: decode-then-invert (official default)
    if decoder_inv:
        with torch.enable_grad():
            latents = pipe.decoder_inv(imgs_t)
        if debug:
            print(f"[DBG] latents.shape={tuple(latents.shape)} dtype={latents.dtype} device={latents.device}")
        return latents

    # otherwise use diffusion inversion (slower)
    with torch.no_grad():
        latents = pipe.inverse(
            prompt=[prompt] * len(images),
            image=images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            inv_order=inv_order,
        )
    if isinstance(latents, dict) and "latents" in latents:
        latents = latents["latents"]
    if debug:
        print(f"[DBG] inverse latents.shape={tuple(latents.shape)} dtype={latents.dtype} device={latents.device}")
    return latents


# -----------------------------
# Detection core
# -----------------------------

def _bits_acc_prefix(decoded_bits: List[int], expected_bits) -> float:
    import numpy as np
    decoded = np.array(decoded_bits, dtype=np.int64)
    expected = np.array(expected_bits, dtype=np.int64)
    m = min(len(decoded), len(expected))
    if m <= 0:
        return float("nan")
    return float((decoded[:m] == expected[:m]).mean())


def _detect_one_batch(
    decoding_key,
    prc_gaussians,
    prc_lib,
    reversed_latents: torch.Tensor,
    var: float,
    fpr: float,
    max_bp_iter: int = 5000,
    key_n_expected: int | None = None,
    expected_bits=None,
    names: Optional[List[str]] = None,
    print_each: bool = True,
    show_k: int = 64,
    verbose: bool = False,
) -> Tuple[List[bool], List[float], List[Optional[str]], List[float]]:
    """
    Return:
      flags:        List[bool]
      scores:       List[float] (nan if unavailable)
      decoded_strs: List[str|None]  (None if Decode failed)
      bit_accs:     List[float] (prefix acc vs expected_bits; nan if expected_bits is None)
    """
    import numpy as np

    bs = reversed_latents.shape[0]
    flags: List[bool] = []
    scores: List[float] = []
    decoded_strs: List[Optional[str]] = []
    bit_accs: List[float] = []

    for i in range(bs):
        name = names[i] if (names is not None and i < len(names)) else f"idx{i}"

        # PRC decode is numpy/scipy style; detach from graph before conversion ops.
        z = reversed_latents[i].detach().to(dtype=torch.float64).flatten().cpu()
        if verbose:
            already = bool(getattr(_detect_one_batch, "_printed_decode_tensor_state", False))
            if not already:
                print(
                    "[PRC][verbose] decode input tensor state: "
                    f"requires_grad={bool(z.requires_grad)}, dtype={z.dtype}, device={z.device}",
                    flush=True,
                )
                setattr(_detect_one_batch, "_printed_decode_tensor_state", True)

        if key_n_expected is not None and int(z.numel()) != int(key_n_expected):
            raise RuntimeError(
                f"PRC dimension mismatch: key expects n={key_n_expected}, but inverted latents have n={int(z.numel())}. "
                f"(name={name})"
            )

        # posteriors (soft values)
        post = prc_gaussians.recover_posteriors(z, variances=float(var))
        post = torch.from_numpy(np.array(post)) if not isinstance(post, torch.Tensor) else post

        # ---- Detect ----
        try:
            out = prc_lib.Detect(decoding_key, post, false_positive_rate=float(fpr))
        except TypeError:
            try:
                out = prc_lib.Detect(decoding_key, post, float(fpr))
            except TypeError:
                out = prc_lib.Detect(decoding_key, post)

        score = float("nan")
        flag = False
        if isinstance(out, tuple) and len(out) >= 1:
            flag = bool(out[0])
            if len(out) >= 2:
                try:
                    score = float(out[1])
                except Exception:
                    pass
        else:
            flag = bool(out)

        # ---- Decode ----
        dec_str = None
        acc = float("nan")
        try:
            try:
                decoded_bits = prc_lib.Decode(decoding_key, post, max_bp_iter=int(max_bp_iter))
            except TypeError:
                try:
                    decoded_bits = prc_lib.Decode(decoding_key, post, int(max_bp_iter))
                except TypeError:
                    decoded_bits = prc_lib.Decode(decoding_key, post)
            if isinstance(decoded_bits, (list, tuple)):
                dec_str = "".join(str(int(b)) for b in decoded_bits)
                if expected_bits is not None:
                    acc = _bits_acc_prefix(decoded_bits, expected_bits)
        except Exception:
            dec_str = None

        flags.append(flag)
        scores.append(score)
        decoded_strs.append(dec_str)
        bit_accs.append(acc)

        if print_each:
            prefix = (dec_str[:show_k] if dec_str is not None else "None")
            if expected_bits is not None:
                print(f"[ZT] {name} detected={int(flag)} score={score:.6g} bit_acc={acc:.3f} dec[:{show_k}]={prefix}")
            else:
                print(f"[ZT] {name} detected={int(flag)} score={score:.6g} dec[:{show_k}]={prefix}")

    return flags, scores, decoded_strs, bit_accs


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    # ---- input modes ----
    ap.add_argument("--run_dir", default=None,
                    help="Your generation run directory (contains sliced/, wm_meta/). Required for image->inversion mode.")
    ap.add_argument("--zT_pt", default=None,
                    help="Optional: direct zT mode. Path to a torch pt tensor with shape [N,4,64,64]. If set, skip image inversion.")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory for CSV and optional latent dumps. Default: run_dir (image mode) or dirname(zT_pt) (zT mode).")
    ap.add_argument("--meta_root", default=None,
                    help="Where to load PRC key/meta files. "
                         "Default: <run_dir>/wm_meta (image mode) or <dirname(zT_pt)>/wm_meta (zT mode).")
    ap.add_argument("--wm_meta_dir", default=None, help="Alias of --meta_root; explicit wm_meta dir has highest priority.")
    ap.add_argument("--key_path", default=None, help="Explicit PRC key pkl path. Highest priority.")

    ap.add_argument("--model_id", default=None, help="Stable Diffusion model path or HF id (required for image->inversion mode)")
    ap.add_argument("--prc_repo", default=None, help="Path to PRC-Watermark-main (to import src/*). Optional.")
    ap.add_argument("--cache_dir", default=None, help="HF cache dir (optional)")

    ap.add_argument("--prompt_mode", choices=["empty", "first_manifest"], default="empty",
                    help="empty: official default. first_manifest: use manifest's first prompt as fixed prompt.")
    ap.add_argument("--prompt", default=None, help="Optional override prompt (fixed for all images).")

    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--inv_steps", type=int, default=-1, help="Alias of --steps.")
    ap.add_argument("--guidance", "--guidance_scale", dest="guidance", type=float, default=7.5)
    ap.add_argument("--solver_order", type=int, default=1)
    ap.add_argument("--inv_order", type=int, default=0, help="Keep 0 to match official decode.py default")
    ap.add_argument("--decoder_inv", type=int, default=1, help="1: decode-then-invert (official default), 0: sample latents")

    ap.add_argument("--dtype", default="fp32", help="fp32 (official default) or fp16/bf16")
    ap.add_argument("--inv_bs", type=int, default=1, help="Batch size (image inversion chunk size OR zT chunk size)")

    ap.add_argument("--var", type=float, default=1.5, help="Variance parameter in pseudogaussians.recover_posteriors")
    ap.add_argument(
        "--fpr",
        type=float,
        default=1e-2,
        help="False positive rate used for PRC Detect; overrides metadata-derived/default key value when provided.",
    )
    ap.add_argument(
        "--max_bp_iter",
        type=int,
        default=5000,
        help="PRC BP decoder max iterations for prc_lib.Decode(...).",
    )

    ap.add_argument("--out_csv", default=None,
                    help="Output CSV path. Default: <run_dir>/detect_results_prcGLOBAL.csv (image mode) "
                         "or <dirname(zT_pt)>/detect_results_prcGLOBAL_from_zT.csv (zT mode)")
    ap.add_argument("--oms_q_pt", type=str, required=True, help="OMS q pt path.")
    ap.add_argument("--oms_meta_json", type=str, default="", help="Optional OMS meta json fallback.")
    ap.add_argument("--save_zt_oms", action="store_true", help="Save zT in OMS domain before OMS inverse.")
    ap.add_argument("--save_zt_restored", action="store_true", help="Save zT after OMS inverse restoration.")
    ap.add_argument("--debug", action="store_true", help="Print debug shapes for key and latents")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of items to process (0=all)")
    ap.add_argument("--max_images", type=int, default=0, help="Alias of --limit.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--zT_start", type=int, default=0, help="(zT mode) start index in zT bank")

    ap.add_argument("--master_key", type=str, default="prc_key_yx_0504",
                    help="Used ONLY for bit-accuracy ground-truth message derivation if prc_message_bits.txt is missing.")
    ap.add_argument("--message_length", type=int, default=8,
                    help="Number of message bits for bit-accuracy evaluation.")

    args = ap.parse_args()
    args.oms_q_pt = _resolve_existing_file(args.oms_q_pt, "oms_q_pt")
    if int(args.inv_steps) > 0:
        args.steps = int(args.inv_steps)
    if int(args.max_images) > 0:
        if int(args.limit) <= 0:
            args.limit = int(args.max_images)
        else:
            args.limit = min(int(args.limit), int(args.max_images))

    direct_zt = args.zT_pt is not None
    if int(args.inv_bs) <= 0:
        raise ValueError("--inv_bs must be >= 1")

    zt_path: Optional[str] = None
    zt_dir: Optional[str] = None
    run_dir: Optional[str] = None

    if direct_zt:
        zt_path = _resolve_existing_file(args.zT_pt, "zT_pt")
        zt_dir = str(Path(zt_path).parent)
        run_dir = zt_dir
    else:
        if not args.run_dir:
            raise ValueError("--run_dir is required when --zT_pt is not provided.")
        run_dir = _resolve_existing_dir(args.run_dir, "run_dir")
        if not args.model_id:
            raise ValueError("--model_id is required in image mode.")

    if args.out_dir:
        out_dir = _resolve_existing_dir(args.out_dir, "out_dir") if os.path.isdir(str(args.out_dir)) else str(Path(args.out_dir).expanduser().resolve())
    else:
        out_dir = run_dir
    os.makedirs(out_dir, exist_ok=True)
    out_lat_dir = os.path.join(out_dir, "latents")
    if args.save_zt_oms or args.save_zt_restored:
        os.makedirs(out_lat_dir, exist_ok=True)

    project_root = _infer_project_root(run_dir=run_dir, zt_path=zt_path)
    prc_repo, prc_repo_tried = _resolve_prc_repo(args.prc_repo, project_root=project_root)
    if prc_repo is None:
        raise FileNotFoundError(
            "Could not resolve PRC repo/module directory. "
            "Please pass --prc_repo explicitly. Tried:\n  " + "\n  ".join(prc_repo_tried)
        )
    _add_prc_repo_to_syspath(prc_repo)

    # ---- imports (PRC official code / local aligned modules) ----
    _, prc_lib, prc_gaussians = _import_official_modules()

    # ---- resolve wm_meta / key ----
    meta_dir, meta_tried = _resolve_meta_dir(
        wm_meta_dir=args.wm_meta_dir,
        meta_root=args.meta_root,
        run_dir=(run_dir if not direct_zt else None),
        zt_dir=zt_dir,
        project_root=project_root,
    )
    decoding_key, meta = _load_prc_keys(meta_dir, key_path=args.key_path)
    resolved_key_path = str(meta.get("key_file", ""))

    key_n, key_hlat, key_target = _infer_key_n_and_target_size(decoding_key)
    if args.debug:
        print(f"[DBG] key_n={key_n} key_hlat={key_hlat} key_target_size={key_target}")

    # ---- expected bits (bit-accuracy) ----
    import numpy as np

    def expected_bits_from_master(master_key: str, L: int) -> np.ndarray:
        digest = hashlib.sha256(master_key.encode("utf-8") + b"::prc_msg").digest()
        bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8)).astype(np.int64)
        if bits.size < L:
            reps = int(np.ceil(L / bits.size))
            bits = np.tile(bits, reps)
        return bits[:L].copy()

    msg_file = os.path.join(meta_dir, "prc_message_bits.txt")
    expected_bits = None
    if os.path.isfile(msg_file):
        raw = open(msg_file, "r", encoding="utf-8").read().strip().replace(" ", "")
        if not raw:
            raise RuntimeError(f"Message bits file is empty: {msg_file}")
        if any(ch not in ("0", "1") for ch in raw):
            raise RuntimeError(f"Message bits file contains non-binary chars: {msg_file}")
        expected_bits = np.array([int(ch) for ch in raw], dtype=np.int64)[: int(args.message_length)]
    elif args.master_key:
        expected_bits = expected_bits_from_master(args.master_key, int(args.message_length))
    else:
        raise RuntimeError(
            f"Missing message bits file and empty --master_key. Expected message bits at: {msg_file}"
        )

    if expected_bits is None or int(expected_bits.size) <= 0:
        raise RuntimeError(
            "Could not resolve expected message bits. "
            "Check prc_message_bits.txt or pass valid --master_key/--message_length."
        )

    if direct_zt:
        out_csv = args.out_csv or os.path.join(out_dir, "detect_results_prcGLOBAL_oms_from_zT.csv")
    else:
        out_csv = args.out_csv or os.path.join(out_dir, "detect_results_prcGLOBAL_oms.csv")
    out_csv = _ensure_parent_dir(out_csv)

    # ---- load OMS q ----
    q_obj = load_q_pt(args.oms_q_pt, block_mode="flat_chunk")
    oms_aux = resolve_forward_aux_from_q_and_meta(
        q_obj=q_obj,
        q_pt_path=args.oms_q_pt,
        q_meta_json=args.oms_meta_json,
    )
    oms_alpha = float(oms_aux.get("blend_alpha", 1.0))
    oms_match_std = _to_bool(oms_aux.get("match_target_std", False))
    oms_rescale = float(oms_aux.get("rescale_factor", 1.0))
    if oms_alpha == 1.0:
        oms_inverse_hint = "pure_Q_inverse" if not oms_match_std else "blended_plus_rescale_inverse"
    else:
        oms_inverse_hint = "blended_inverse" if not oms_match_std else "blended_plus_rescale_inverse"

    mode_str = "zT" if direct_zt else "image"
    print("[CFG] mode=", mode_str)
    print("[CFG] project_root=", project_root)
    print("[CFG] model_id=", args.model_id if args.model_id else "(N/A in zT mode)")
    print("[CFG] run_dir=", run_dir)
    print("[CFG] zT_pt=", zt_path if zt_path else "")
    print("[CFG] out_dir=", out_dir)
    print("[CFG] wm_meta_dir=", meta_dir)
    print("[CFG] oms_q_pt=", args.oms_q_pt)
    print("[CFG] oms_meta_json_provided=", bool(str(args.oms_meta_json).strip()))
    print("[CFG] oms_inverse_default_mode=", oms_inverse_hint)
    print("[CFG] oms_blend_alpha=", oms_alpha, "oms_match_target_std=", oms_match_std, "oms_rescale_factor=", oms_rescale)
    print("[CFG] key_path=", resolved_key_path)
    print("[CFG] prc_repo=", prc_repo)
    print("[CFG] out_csv=", out_csv)
    print("[CFG] wm_meta_search_tried=", len(meta_tried), "locations")
    print("[CFG] guidance_scale=", args.guidance)
    if key_n is not None:
        print("[KEY] expected_n=", key_n, "target_size=", key_target)

    csv_cols = [
        "image",
        "image_path",
        "run_dir",
        "zT_pt",
        "model_id",
        "inv_steps",
        "zT_oms_path",
        "zT_restored_path",
        "wm_meta_dir",
        "oms_q_pt",
        "oms_blend_alpha",
        "oms_match_target_std",
        "oms_rescale_factor",
        "oms_inverse_mode",
        "detected",
        "score",
        "prompt_used",
        "steps",
        "guidance",
        "var",
        "fpr",
        "max_bp_iter",
        "key_file",
        "decode_ok",
        "bit_acc",
    ]

    # ---- zT mode ----
    if direct_zt:
        obj = torch.load(zt_path, map_location="cpu")
        if isinstance(obj, dict):
            zT = None
            for k in ("zT", "zT_att", "latents", "z"):
                if k in obj:
                    zT = obj[k]
                    break
            if zT is None:
                raise KeyError(f"--zT_pt is a dict but cannot find latent tensor key; keys={list(obj.keys())[:30]}")
        else:
            zT = obj

        if not isinstance(zT, torch.Tensor):
            raise TypeError(f"--zT_pt must contain a torch.Tensor (or dict containing one), got {type(zT)}")

        if zT.ndim != 4 or zT.shape[1:] != (4, 64, 64):
            raise ValueError(f"zT must have shape [N,4,64,64], got {tuple(zT.shape)}")

        N = int(zT.shape[0])
        start = int(args.zT_start)
        if start < 0 or start >= N:
            raise ValueError(f"--zT_start out of range: {start} (N={N})")

        total = N - start
        if int(args.limit) > 0:
            total = min(total, int(args.limit))
        end = start + total
        zT = zT[start:end].contiguous()

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(csv_cols)

        print("[CFG] zT_range=", f"[{start}:{end})", "count=", total, "bs=", int(args.inv_bs))
        print("[CFG] var=", args.var, "fpr=", args.fpr, "max_bp_iter=", args.max_bp_iter)
        print("[KEY] ", resolved_key_path)

        detected_cnt = 0
        bit_acc_sum = 0.0
        bit_acc_cnt = 0

        n = int(zT.shape[0])
        bs = max(1, int(args.inv_bs))

        for i in range(0, n, bs):
            zt_oms_batch = zT[i:i + bs].to(torch.float32)
            batch_names = [f"zT:{start + i + j:04d}" for j in range(zt_oms_batch.shape[0])]

            zt_restored_batch, inv_info = apply_oms_inverse_to_latent(
                zt_oms_batch,
                q_obj=q_obj,
                oms_q_pt=args.oms_q_pt,
                oms_meta_json=args.oms_meta_json,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                verbose=bool(args.verbose),
            )
            if args.verbose:
                m = float(zt_restored_batch.mean().item())
                s = float(zt_restored_batch.std(unbiased=False).item())
                solved = None
                if isinstance(inv_info.get("solve_summary"), dict):
                    solved = inv_info["solve_summary"].get("all_blocks_solved")
                print(
                    f"[ZT][verbose] OMS inverse mode={inv_info['inverse_mode']} restored_mean={m:.8f} restored_std={s:.8f} all_blocks_solved={solved}",
                    flush=True,
                )

            flags, scores, decoded_strs, bit_accs = _detect_one_batch(
                decoding_key=decoding_key,
                prc_gaussians=prc_gaussians,
                prc_lib=prc_lib,
                reversed_latents=zt_restored_batch,
                var=float(args.var),
                fpr=float(args.fpr),
                max_bp_iter=int(args.max_bp_iter),
                key_n_expected=key_n,
                expected_bits=expected_bits,
                names=batch_names,
                print_each=False,
                show_k=64,
                verbose=bool(args.verbose),
            )

            with open(out_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for j, name in enumerate(batch_names):
                    global_idx = start + i + j
                    det = int(bool(flags[j]))
                    sc = scores[j]
                    acc = bit_accs[j]
                    dec_ok = int(decoded_strs[j] is not None)
                    detected_cnt += det
                    if expected_bits is not None and not (acc != acc):  # not nan
                        bit_acc_sum += float(acc)
                        bit_acc_cnt += 1

                    zt_oms_path = ""
                    zt_restored_path = ""
                    if args.save_zt_oms:
                        zt_oms_path = os.path.join(out_lat_dir, f"zT_{global_idx:04d}_inv_zT_oms.pt")
                        torch.save({"zT": zt_oms_batch[j:j + 1].detach().float().cpu(), "index": int(global_idx), "domain": "oms"}, zt_oms_path)
                    if args.save_zt_restored:
                        zt_restored_path = os.path.join(out_lat_dir, f"zT_{global_idx:04d}_inv_zT_restored.pt")
                        torch.save(
                            {
                                "zT": zt_restored_batch[j:j + 1].detach().float().cpu(),
                                "index": int(global_idx),
                                "domain": "restored_prc",
                                "oms_inverse_info": inv_info,
                            },
                            zt_restored_path,
                        )

                    if expected_bits is not None:
                        print(
                            f"[{i+j+1:5d}/{n}] {name}  oms_mode={inv_info['inverse_mode']}  detected={det}  score={sc:.6g}  bit_acc={acc:.3f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[{i+j+1:5d}/{n}] {name}  oms_mode={inv_info['inverse_mode']}  detected={det}  score={sc:.6g}",
                            flush=True,
                        )

                    w.writerow([
                        name,
                        "",  # image_path
                        run_dir,
                        zt_path,
                        args.model_id if args.model_id else "",
                        int(args.steps),
                        zt_oms_path,
                        zt_restored_path,
                        meta_dir,
                        args.oms_q_pt,
                        float(inv_info.get("blend_alpha", oms_alpha)),
                        int(bool(inv_info.get("match_target_std", oms_match_std))),
                        float(inv_info.get("rescale_factor", oms_rescale)),
                        str(inv_info.get("inverse_mode", "")),
                        det,
                        sc,
                        "",  # prompt_used N/A
                        int(args.steps),
                        float(args.guidance),
                        float(args.var),
                        float(args.fpr),
                        int(args.max_bp_iter),
                        resolved_key_path,
                        dec_ok,
                        (float(acc) if (expected_bits is not None) else float("nan")),
                    ])

        detect_rate = detected_cnt / float(n) if n > 0 else 0.0
        avg_bit_acc = (bit_acc_sum / float(bit_acc_cnt)) if bit_acc_cnt > 0 else float("nan")
        if expected_bits is not None:
            print(f"[SUMMARY] detect_rate={detect_rate:.4f}  avg_bit_acc={avg_bit_acc:.4f}  n={n}")
        else:
            print(f"[SUMMARY] detect_rate={detect_rate:.4f}  n={n}")

        print(f"[DONE] wrote {out_csv} | {n} zT samples")
        return

    # ---- image->inversion->OMS inverse mode ----
    img_paths = collect_images_from_run_dir(run_dir)
    if args.limit and int(args.limit) > 0:
        img_paths = img_paths[: int(args.limit)]
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No PNGs found in run_dir: {run_dir}")

    torch_dtype = _resolve_torch_dtype(args.dtype)
    pipe = _make_pipe(args.model_id, torch_dtype=torch_dtype, cache_dir=args.cache_dir, solver_order=args.solver_order)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(csv_cols)

    # Prompt selection (official default: empty prompt)
    fixed_prompt = ""
    manifest_path = None
    manifest_tried: List[str] = []
    if args.prompt is not None:
        fixed_prompt = str(args.prompt)
    elif args.prompt_mode == "first_manifest":
        manifest_path, manifest_tried = _resolve_manifest_csv(run_dir)
        if manifest_path:
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    first = next(reader)
                    if first and "prompt" in first and first["prompt"] is not None:
                        fixed_prompt = str(first["prompt"])
            except Exception:
                fixed_prompt = ""

    print("[CFG] images=", len(img_paths), "inv_bs=", args.inv_bs)
    print("[CFG] prompt_mode=", args.prompt_mode, "prompt_used=", repr(fixed_prompt))
    if args.prompt_mode == "first_manifest":
        print("[CFG] manifest_found=", bool(manifest_path), "manifest_csv=", manifest_path if manifest_path else "")
        if (not manifest_path) and args.debug:
            print("[DBG] manifest_search_tried=", manifest_tried)
    print("[CFG] steps=", args.steps, "guidance=", args.guidance, "dtype=", str(torch_dtype).replace("torch.", ""))
    print("[CFG] var=", args.var, "fpr=", args.fpr, "max_bp_iter=", args.max_bp_iter)
    print("[KEY] ", resolved_key_path)

    n = len(img_paths)

    for i in range(0, n, int(args.inv_bs)):
        batch_paths = img_paths[i:i + int(args.inv_bs)]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        batch_names = [os.path.basename(p) for p in batch_paths]

        # inversion gives zT in OMS domain
        zt_oms_batch = exact_inversion_batch(
            pipe=pipe,
            images=batch_imgs,
            prompt=fixed_prompt,
            guidance_scale=float(args.guidance),
            num_inference_steps=int(args.steps),
            inv_order=int(args.inv_order),
            decoder_inv=bool(int(args.decoder_inv)),
            target_size=key_target,
            debug=bool(args.debug),
            verbose=bool(args.verbose),
        )
        print(f"[IMG] inversion done: batch_start={i} batch_size={len(batch_paths)}", flush=True)

        zt_restored_batch, inv_info = apply_oms_inverse_to_latent(
            zt_oms_batch,
            q_obj=q_obj,
            oms_q_pt=args.oms_q_pt,
            oms_meta_json=args.oms_meta_json,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            verbose=bool(args.verbose),
        )
        print(
            f"[IMG] OMS inverse: mode={inv_info['inverse_mode']} alpha={inv_info['blend_alpha']:.6f} "
            f"match_std={inv_info['match_target_std']} rescale={inv_info['rescale_factor']:.8f}",
            flush=True,
        )
        if args.verbose:
            m = float(zt_restored_batch.mean().item())
            s = float(zt_restored_batch.std(unbiased=False).item())
            solved = None
            if isinstance(inv_info.get("solve_summary"), dict):
                solved = inv_info["solve_summary"].get("all_blocks_solved")
            print(f"[IMG][verbose] restored_mean={m:.8f} restored_std={s:.8f} all_blocks_solved={solved}", flush=True)

        flags, scores, decoded_strs, bit_accs = _detect_one_batch(
            decoding_key=decoding_key,
            prc_gaussians=prc_gaussians,
            prc_lib=prc_lib,
            reversed_latents=zt_restored_batch,
            var=float(args.var),
            fpr=float(args.fpr),
            max_bp_iter=int(args.max_bp_iter),
            key_n_expected=key_n,
            expected_bits=expected_bits,
            names=batch_names,
            print_each=False,
            show_k=64,
            verbose=bool(args.verbose),
        )

        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for j, p in enumerate(batch_paths):
                idx = i + j
                name = os.path.basename(p)
                det = int(bool(flags[j]))
                sc = scores[j]
                dec = decoded_strs[j]
                acc = bit_accs[j]
                dec_ok = int(dec is not None)
                dec_prefix = (dec[:64] if dec is not None else "None")

                zt_oms_path = ""
                zt_restored_path = ""
                if args.save_zt_oms:
                    zt_oms_path = os.path.join(out_lat_dir, f"{Path(p).stem}_inv_zT_oms.pt")
                    torch.save(
                        {
                            "zT": zt_oms_batch[j:j + 1].detach().float().cpu(),
                            "image": name,
                            "image_path": p,
                            "run_dir": run_dir,
                            "domain": "oms",
                            "oms_q_pt": args.oms_q_pt,
                        },
                        zt_oms_path,
                    )
                if args.save_zt_restored:
                    zt_restored_path = os.path.join(out_lat_dir, f"{Path(p).stem}_inv_zT_restored.pt")
                    torch.save(
                        {
                            "zT": zt_restored_batch[j:j + 1].detach().float().cpu(),
                            "image": name,
                            "image_path": p,
                            "run_dir": run_dir,
                            "domain": "restored_prc",
                            "oms_q_pt": args.oms_q_pt,
                            "oms_inverse_info": inv_info,
                        },
                        zt_restored_path,
                    )

                if expected_bits is not None:
                    print(
                        f"[{idx+1:5d}/{n}] {name}  oms_mode={inv_info['inverse_mode']}  detected={det}  score={sc:.6g}  "
                        f"decode={'OK' if dec_ok else 'FAIL'}  bit_acc={acc:.3f}  dec[:64]={dec_prefix}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{idx+1:5d}/{n}] {name}  oms_mode={inv_info['inverse_mode']}  detected={det}  score={sc:.6g}  "
                        f"decode={'OK' if dec_ok else 'FAIL'}  dec[:64]={dec_prefix}",
                        flush=True,
                    )

                w.writerow([
                    name,
                    p,
                    run_dir,
                    "",
                    args.model_id if args.model_id else "",
                    int(args.steps),
                    zt_oms_path,
                    zt_restored_path,
                    meta_dir,
                    args.oms_q_pt,
                    float(inv_info.get("blend_alpha", oms_alpha)),
                    int(bool(inv_info.get("match_target_std", oms_match_std))),
                    float(inv_info.get("rescale_factor", oms_rescale)),
                    str(inv_info.get("inverse_mode", "")),
                    det,
                    sc,
                    fixed_prompt,
                    int(args.steps),
                    float(args.guidance),
                    float(args.var),
                    float(args.fpr),
                    int(args.max_bp_iter),
                    resolved_key_path,
                    dec_ok,
                    (float(acc) if (expected_bits is not None) else float("nan")),
                ])

    print(f"[DONE] wrote {out_csv} | {n} images")


if __name__ == "__main__":
    main()
