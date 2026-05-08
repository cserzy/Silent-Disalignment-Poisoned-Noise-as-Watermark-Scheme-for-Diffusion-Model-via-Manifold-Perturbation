#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Detect PRC GLOBAL watermarks from images, aligned to PRC-Watermark's official decode/inversion.

Assumptions
- Images: <run_dir>/sliced/*.png (no recursion)
- Keys:   <run_dir>/wm_meta/prc_keys.pkl (preferred)
          (also supports a few fallback filenames)
- Prompt: by default uses empty prompt (official default). Optionally can use the
          first prompt in manifest.csv as a fixed prompt for all images.

Batching
- --inv_bs controls how many images are inverted per chunk.

Outputs
- A CSV with per-image detection results, and prints realtime progress.

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
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image


# -----------------------------
# Path helpers / imports
# -----------------------------
'''
DEFAULT_PRC_REPO = "/home/yancy/work/dm_backdoor_latent_space/PRC-Watermark-main"


def _add_prc_repo_to_syspath(prc_repo: Optional[str]) -> None:
    repo = prc_repo or DEFAULT_PRC_REPO
    if repo and os.path.isdir(repo):
        if repo not in sys.path:
            sys.path.insert(0, repo)
        src = os.path.join(repo, "src")
        if os.path.isdir(src) and src not in sys.path:
            sys.path.insert(0, src)
'''

def _import_official_modules():
    """
    Import PRC official modules.
    Returns:
      InversableStableDiffusionPipeline, prc_lib, prc_gaussians, optim_utils
    """
    try:
        from src.inverse_stable_diffusion import InversableStableDiffusionPipeline
    except Exception:
        # some forks place it directly
        from inverse_stable_diffusion import InversableStableDiffusionPipeline  # type: ignore

    try:
        import prc as prc_lib  # type: ignore
    except Exception:
        from src import prc as prc_lib  # type: ignore

    try:
        import pseudogaussians as prc_gaussians  # type: ignore
    except Exception:
        from src import pseudogaussians as prc_gaussians  # type: ignore

    try:
        import optim_utils as optim_utils  # type: ignore
    except Exception:
        from src import optim_utils as optim_utils  # type: ignore

    return InversableStableDiffusionPipeline, prc_lib, prc_gaussians, optim_utils


# -----------------------------
# Key loading
# -----------------------------

def _load_prc_keys(meta_dir: str) -> Tuple[Any, Dict[str, Any]]:
    """Load PRC decoding key, plus meta.

    We prefer a single combined pickle (to match official workflows): prc_keys.pkl
    Expected shapes:
      - dict with keys like {'encoding_key':..., 'decoding_key':...}
      - tuple/list (encoding_key, decoding_key)
      - or directly a decoding_key object
    """
    tried: List[str] = []

    # preferred combined key
    key_cands = [
        os.path.join(meta_dir, "prc_keys.pkl"),
        os.path.join(meta_dir, "keys.pkl"),
        os.path.join(meta_dir, "prc_decoding_key.pkl"),
        os.path.join(meta_dir, "decoding_key.pkl"),
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
            "Missing. Missing PRC key file. Tried:\n  " + "\n  ".join(tried)
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
    generator_matrix is (k, n) over GF(2), so n can be read from its second dimension.
    """
    try:
        gen = decoding_key[0]
        n = int(getattr(gen, "shape", [None, None])[1])
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


def _make_pipe(model_id: str, torch_dtype: torch.dtype, cache_dir: Optional[str], solver_order: int):
    from diffusers import DPMSolverMultistepScheduler

    pipe = _import_official_modules()[0].from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Align scheduler to DPM-Solver Multistep
    # IMPORTANT: keep these consistent between gen and detect for best inversion.
    pipe.scheduler = DPMSolverMultistepScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        prediction_type="epsilon",
        steps_offset=1,
        solver_order=int(solver_order),
        thresholding=False,
        clip_sample=False,
        sample_max_value=1.0,
        algorithm_type="dpmsolver++",
        solver_type="midpoint",
        lower_order_final=True,
        use_karras_sigmas=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
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
) -> torch.Tensor:
    """
    Invert images back to latents using official inverse pipeline.

    Returns: latents tensor (B,4,64,64) on CUDA.
    """
    device = pipe.device
    imgs_t = pipe.feature_extractor(images=images, return_tensors="pt").pixel_values.to(device)
    if target_size is not None:
        # ensure size is consistent (target_size is in pixels)
        pass

    if debug:
        print(f"[DBG] imgs_t.shape={tuple(imgs_t.shape)} dtype={imgs_t.dtype} device={imgs_t.device}")

    # decoder_inv: decode-then-invert (official default)
    if decoder_inv:
        latents = pipe.decoder_inv(imgs_t)
        if debug:
            print(f"[DBG] latents.shape={tuple(latents.shape)} dtype={latents.dtype} device={latents.device}")
        return latents

    # otherwise use diffusion inversion (slower)
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
    key_n_expected: int | None = None,
    expected_bits=None,
    names: Optional[List[str]] = None,
    print_each: bool = True,
    show_k: int = 64,
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

        z = reversed_latents[i].to(dtype=torch.float64).flatten().cpu()
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
    ap.add_argument("--meta_root", default=None,
                    help="Where to load PRC key/meta files. "
                         "Default: <run_dir>/wm_meta (image mode) or <dirname(zT_pt)>/wm_meta (zT mode).")

    ap.add_argument("--model_id", default=None, help="Stable Diffusion model path or HF id (required for image->inversion mode)")
    ap.add_argument("--prc_repo", default="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/script-experiment/detect", help="Path to PRC-Watermark-main (to import src/*). Optional.")
    ap.add_argument("--cache_dir", default=None, help="HF cache dir (optional)")

    ap.add_argument("--prompt_mode", choices=["empty", "first_manifest"], default="empty",
                    help="empty: official default. first_manifest: use manifest's first prompt as fixed prompt.")
    ap.add_argument("--prompt", default=None, help="Optional override prompt (fixed for all images).")

    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--solver_order", type=int, default=1)
    ap.add_argument("--inv_order", type=int, default=0, help="Keep 0 to match official decode.py default")
    ap.add_argument("--decoder_inv", type=int, default=1, help="1: decode-then-invert (official default), 0: sample latents")

    ap.add_argument("--dtype", default="fp32", help="fp32 (official default) or fp16/bf16")
    ap.add_argument("--inv_bs", type=int, default=1, help="Batch size (image inversion chunk size OR zT chunk size)")

    ap.add_argument("--var", type=float, default=1.5, help="Variance parameter in pseudogaussians.recover_posteriors")
    ap.add_argument("--fpr", type=float, default=1e-5, help="False positive rate for PRC Detect")

    ap.add_argument("--out_csv", default=None,
                    help="Output CSV path. Default: <run_dir>/detect_results_prcGLOBAL.csv (image mode) "
                         "or <dirname(zT_pt)>/detect_results_prcGLOBAL_from_zT.csv (zT mode)")
    ap.add_argument("--debug", action="store_true", help="Print debug shapes for key and latents")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of items to process (0=all)")
    ap.add_argument("--zT_start", type=int, default=0, help="(zT mode) start index in zT bank")

    ap.add_argument("--master_key", type=str, default="prc_key_yx_0504",
                    help="Used ONLY for bit-accuracy ground-truth message derivation if prc_message_bits.txt is missing.")
    ap.add_argument("--message_length", type=int, default=8,
                    help="Number of message bits for bit-accuracy evaluation.")

    args = ap.parse_args()

    direct_zt = args.zT_pt is not None

    # ---- imports (PRC official code) ----
    #add_prc_repo_to_syspath(args.prc_repo)
    InversableStableDiffusionPipeline, prc_lib, prc_gaussians, optim_utils = _import_official_modules()

    # ---- resolve meta dir ----
    if direct_zt:
        zt_path = os.path.abspath(args.zT_pt)
        zt_dir = os.path.dirname(zt_path)
        meta_dir = os.path.abspath(args.meta_root) if args.meta_root else os.path.join(zt_dir, "wm_meta")
        run_dir = zt_dir  # for default out_csv
    else:
        if not args.run_dir:
            raise ValueError("--run_dir is required when --zT_pt is not provided.")
        if not args.model_id:
            raise ValueError("--model_id is required in image->inversion mode.")
        run_dir = os.path.abspath(args.run_dir)
        meta_dir = os.path.abspath(args.meta_root) if args.meta_root else os.path.join(run_dir, "wm_meta")

    if not os.path.isdir(meta_dir):
        raise FileNotFoundError(f"Missing meta dir: {meta_dir}")

    # ---- load key ----
    decoding_key, meta = _load_prc_keys(meta_dir)
    key_n, key_hlat, key_target = _infer_key_n_and_target_size(decoding_key)
    if args.debug:
        print(f"[DBG] key_n={key_n} key_hlat={key_hlat} key_target_size={key_target}")

    # ---- expected bits (bit-accuracy) ----
    import hashlib, numpy as np

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
        s = open(msg_file, "r", encoding="utf-8").read().strip().replace(" ", "")
        expected_bits = np.array([int(ch) for ch in s], dtype=np.int64)[:int(args.message_length)]
    else:
        if args.master_key:
            expected_bits = expected_bits_from_master(args.master_key, int(args.message_length))

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

        out_csv = args.out_csv or os.path.join(run_dir, "detect_results_prcGLOBAL_from_zT.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

        # header
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "image_path",  # keep column name for downstream compatibility
                "detected",
                "score",
                "prompt_used",
                "steps",
                "guidance",
                "var",
                "fpr",
                "key_file",
                "bit_acc",
            ])

        print("[CFG] mode=direct_zT")
        print("[CFG] zT_pt=", zt_path)
        print("[CFG] zT_range=", f"[{start}:{end})", "count=", total, "bs=", int(args.inv_bs))
        print("[CFG] var=", args.var, "fpr=", args.fpr)
        print("[KEY] ", meta.get("key_file"))
        if key_n is not None:
            print("[KEY] expected_n=", key_n)

        detected_cnt = 0
        bit_acc_sum = 0.0
        bit_acc_cnt = 0

        n = int(zT.shape[0])
        bs = max(1, int(args.inv_bs))

        for i in range(0, n, bs):
            batch = zT[i:i + bs]  # CPU
            batch_names = [f"zT:{start + i + j:04d}" for j in range(batch.shape[0])]

            flags, scores, decoded_strs, bit_accs = _detect_one_batch(
                decoding_key=decoding_key,
                prc_gaussians=prc_gaussians,
                prc_lib=prc_lib,
                reversed_latents=batch,
                var=float(args.var),
                fpr=float(args.fpr),
                key_n_expected=key_n,
                expected_bits=expected_bits,
                names=batch_names,
                print_each=False,
                show_k=64,
            )

            with open(out_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for j, name in enumerate(batch_names):
                    det = int(bool(flags[j]))
                    sc = scores[j]
                    acc = bit_accs[j]
                    detected_cnt += det
                    if expected_bits is not None and not (acc != acc):  # not nan
                        bit_acc_sum += float(acc)
                        bit_acc_cnt += 1

                    if expected_bits is not None:
                        print(f"[{i+j+1:5d}/{n}] {name}  detected={det}  score={sc:.6g}  bit_acc={acc:.3f}")
                    else:
                        print(f"[{i+j+1:5d}/{n}] {name}  detected={det}  score={sc:.6g}")

                    w.writerow([
                        name,
                        det,
                        sc,
                        "",  # prompt_used N/A
                        int(args.steps),
                        float(args.guidance),
                        float(args.var),
                        float(args.fpr),
                        meta.get("key_file", ""),
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

    # ---- image->inversion mode (original workflow) ----
    sliced_dir = os.path.join(run_dir, "sliced")
    if not os.path.isdir(sliced_dir):
        raise FileNotFoundError(f"Missing sliced dir: {sliced_dir}")

    # images
    img_paths = sorted(glob.glob(os.path.join(sliced_dir, "*.png")))
    if args.limit and int(args.limit) > 0:
        img_paths = img_paths[: int(args.limit)]
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No PNGs found in {sliced_dir}")

    out_csv = args.out_csv or os.path.join(run_dir, "detect_results_prcGLOBAL.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    torch_dtype = _resolve_torch_dtype(args.dtype)
    pipe = _make_pipe(args.model_id, torch_dtype=torch_dtype, cache_dir=args.cache_dir, solver_order=args.solver_order)

    # Stream + write header
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "image_path",
            "detected",
            "score",
            "prompt_used",
            "steps",
            "guidance",
            "var",
            "fpr",
            "key_file",
        ])

    # Prompt selection (official default: empty prompt)
    fixed_prompt = ""
    if args.prompt is not None:
        fixed_prompt = str(args.prompt)
    elif args.prompt_mode == "first_manifest":
        mani = os.path.join(sliced_dir, "manifest.csv")
        if os.path.isfile(mani):
            try:
                with open(mani, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    first = next(reader)
                    if first and "prompt" in first and first["prompt"] is not None:
                        fixed_prompt = str(first["prompt"])
            except Exception:
                fixed_prompt = ""

    print("[CFG] run_dir=", run_dir)
    print("[CFG] images=", len(img_paths), "inv_bs=", args.inv_bs)
    print("[CFG] prompt_mode=", args.prompt_mode, "prompt_used=", repr(fixed_prompt))
    print("[CFG] steps=", args.steps, "guidance=", args.guidance, "dtype=", str(torch_dtype).replace('torch.', ''))
    print("[CFG] var=", args.var, "fpr=", args.fpr)
    print("[KEY] ", meta.get("key_file"))
    if key_n is not None:
        print("[KEY] expected_n=", key_n, "target_size=", key_target)

    n = len(img_paths)

    for i in range(0, n, int(args.inv_bs)):
        batch_paths = img_paths[i:i + int(args.inv_bs)]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        batch_names = [os.path.basename(p) for p in batch_paths]
        # Official inversion is float32 by default; keep it unless you override --dtype.
        reversed_latents = exact_inversion_batch(
            pipe=pipe,
            images=batch_imgs,
            prompt=fixed_prompt,
            guidance_scale=float(args.guidance),
            num_inference_steps=int(args.steps),
            inv_order=int(args.inv_order),
            decoder_inv=bool(int(args.decoder_inv)),
            target_size=key_target,
            debug=bool(args.debug),
        )

        flags, scores, decoded_strs, bit_accs = _detect_one_batch(
            decoding_key=decoding_key,
            prc_gaussians=prc_gaussians,
            prc_lib=prc_lib,
            reversed_latents=reversed_latents,
            var=float(args.var),
            fpr=float(args.fpr),
            key_n_expected=key_n,
            expected_bits=expected_bits,
            names=batch_names,
            print_each=False,
            show_k=64,
        )

        # Append rows + print realtime
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for j, p in enumerate(batch_paths):
                idx = i + j
                name = os.path.basename(p)
                det = int(bool(flags[j]))
                sc = scores[j]
                dec = decoded_strs[j]
                acc = bit_accs[j]
                dec_ok = (dec is not None)

                dec_prefix = (dec[:64] if dec_ok else "None")

                if expected_bits is not None:
                    print(f"[{idx+1:5d}/{n}] {name}  detected={det}  score={sc:.6g}  decode={'OK' if dec_ok else 'FAIL'}  bit_acc={acc:.3f}  dec[:64]={dec_prefix}")
                else:
                    print(f"[{idx+1:5d}/{n}] {name}  detected={det}  score={sc:.6g}  decode={'OK' if dec_ok else 'FAIL'}  dec[:64]={dec_prefix}")

                w.writerow([
                    p,
                    det,
                    sc,
                    fixed_prompt,
                    int(args.steps),
                    float(args.guidance),
                    float(args.var),
                    float(args.fpr),
                    meta.get("key_file", ""),
                ])

    print(f"[DONE] wrote {out_csv} | {n} images")


if __name__ == "__main__":
    main()
