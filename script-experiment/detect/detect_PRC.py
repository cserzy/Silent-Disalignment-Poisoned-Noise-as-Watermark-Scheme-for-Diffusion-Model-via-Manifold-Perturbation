#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
detect_PRC.py

Run PRC GLOBAL detection by:
  images (run_dir/sliced/*.png)
    -> exact inversion (official PRC-Watermark InversableStableDiffusionPipeline)
    -> recover_posteriors (your local pseudogaussians.py)
    -> Detect/Decode (your local prc.py)

Key:
  default loads decoding_key from:
    /home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/prc_key.pkl

Outputs:
  <run_dir>/detect_prc_per_image.csv
  <run_dir>/detect_prc_summary.csv

Usage:
  CUDA_VISIBLE_DEVICES=0 python detect_PRC.py --run_dir <RUN_DIR> --model_id <SD_PATH>
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import pickle
import sys
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


# -------------------------
# Defaults (aligned to your experiment)
# -------------------------

DEFAULT_PRC_REPO = "/home/yancy/work/dm_backdoor_latent_space/PRC-Watermark-main"
DEFAULT_KEY_PATH = "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/prc_key_0_8.pkl"

DEFAULT_MODEL_ID = "/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-5-diffusers"

DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 7.5
DEFAULT_SOLVER_ORDER = 1
DEFAULT_INV_ORDER = 0
DEFAULT_DECODER_INV = 1

DEFAULT_DTYPE = "fp32"
DEFAULT_INV_BS = 1

DEFAULT_VAR = 1.5
DEFAULT_FPR = 1e-5

DEFAULT_MASTER_KEY = "prc_key_yx_0504"
DEFAULT_MESSAGE_LENGTH = 8


# -------------------------
# Path / import helpers
# -------------------------

def _add_paths(prc_repo: str) -> None:
    """Ensure local dir first (for your modified prc.py/pseudogaussians.py), then PRC repo."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    prc_repo = os.path.abspath(prc_repo)
    if prc_repo not in sys.path:
        sys.path.insert(1, prc_repo)


def _resolve_torch_dtype(dtype: str) -> torch.dtype:
    d = dtype.lower()
    if d in ("fp16", "float16", "half"):
        return torch.float16
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    if d in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown --dtype {dtype!r}. Use fp16|bf16|fp32.")


def _import_modules():
    """
    Inversion + transform: from official PRC repo (src.*)
    PRC + pseudogaussians: from your local files in same dir as detect_PRC.py
    """
    try:
        from src.inverse_stable_diffusion import InversableStableDiffusionPipeline  # type: ignore
        from src import optim_utils  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import official PRC-Watermark modules.\n"
            f"Expected repo at: {DEFAULT_PRC_REPO}\n"
            "You can override via --prc_repo.\n"
            f"Original error: {repr(e)}"
        )

    # local (your modified)
    import prc as prc_lib  # type: ignore
    import pseudogaussians as prc_gaussians  # type: ignore

    return InversableStableDiffusionPipeline, optim_utils, prc_lib, prc_gaussians


# -------------------------
# Key loading
# -------------------------

def _load_decoding_key(key_path: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Supports:
      - dict: {'encoding_key':..., 'decoding_key':...}
      - tuple/list: (encoding_key, decoding_key)  <-- your current format
      - direct decoding_key object
    """
    if not os.path.isfile(key_path):
        raise FileNotFoundError(f"Missing prc_key.pkl: {key_path}")

    with open(key_path, "rb") as f:
        obj = pickle.load(f)

    decoding_key = None
    if isinstance(obj, dict):
        for k in ("decoding_key", "decoding", "dk"):
            if k in obj:
                decoding_key = obj[k]
                break
        if decoding_key is None and ("encoding_key" in obj and "decoding_key" in obj):
            decoding_key = obj["decoding_key"]
    elif isinstance(obj, (tuple, list)) and len(obj) >= 2:
        decoding_key = obj[1]
    else:
        decoding_key = obj

    if decoding_key is None:
        raise RuntimeError(f"Loaded {key_path} but could not extract decoding_key.")

    meta: Dict[str, Any] = {"key_file": key_path}
    return decoding_key, meta


def _infer_key_n_and_target_size(decoding_key):
    """
    decoding_key is a tuple where element[2] is one_time_pad in your prc.py.
    n = len(one_time_pad) = 4*(H/8)*(W/8) for SD.
    """
    try:
        one_time_pad = decoding_key[2]
        n = len(one_time_pad)
    except Exception:
        return None, None, None

    h_lat = None
    target_size = None
    try:
        h_lat = int(round(math.sqrt(n / 4.0)))
        if 4 * h_lat * h_lat == int(n):
            target_size = int(h_lat * 8)
        else:
            h_lat = None
    except Exception:
        h_lat = None

    return int(n), h_lat, target_size


# -------------------------
# Expected bits (derive from master_key; aligned to your generator)
# -------------------------

def _expected_bits_from_master_key(master_key: str, message_length: int) -> np.ndarray:
    mk_bytes = master_key.encode("utf-8")
    digest = hashlib.sha256(mk_bytes + b"::prc_msg").digest()
    bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8)).astype(np.int64)
    if bits.size < message_length:
        reps = int(np.ceil(message_length / bits.size))
        bits = np.tile(bits, reps)
    return bits[:message_length].copy()


def _prefix_bit_acc(decoded: np.ndarray, expected: np.ndarray) -> float:
    m = min(len(decoded), len(expected))
    if m <= 0:
        return float("nan")
    return float((decoded[:m] == expected[:m]).mean())


# -------------------------
# Pipe + inversion (aligned to your official-align script)
# -------------------------

def _make_pipe(model_id: str, torch_dtype: torch.dtype, cache_dir: Optional[str], solver_order: int):
    from diffusers import DPMSolverMultistepScheduler

    InversableStableDiffusionPipeline, _, _, _ = _import_modules()

    scheduler = DPMSolverMultistepScheduler(
        beta_end=0.012,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        num_train_timesteps=1000,
        prediction_type="epsilon",
        steps_offset=1,
        trained_betas=None,
        solver_order=int(solver_order),
    )

    kw = dict(scheduler=scheduler, torch_dtype=torch_dtype)
    if cache_dir:
        kw["cache_dir"] = cache_dir

    pipe = InversableStableDiffusionPipeline.from_pretrained(model_id, **kw)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    return pipe


def exact_inversion_batch(
    pipe,
    images: Sequence[Image.Image],
    prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    inv_order: int,
    decoder_inv: bool,
    target_size: Optional[int] = None,
    debug: bool = False,
):
    """
    Batched exact inversion aligned with PRC-Watermark official inversion signature.
    """
    _, optim_utils, _, _ = _import_modules()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = len(images)
    if bs == 0:
        raise ValueError("Empty image batch")

    do_cfg = guidance_scale > 1.0

    text_embeddings_tuple = pipe.encode_prompt(prompt, device, 1, do_cfg, negative_prompt=None)
    prompt_embeds = text_embeddings_tuple[0]
    neg_embeds = text_embeddings_tuple[1]

    if do_cfg:
        prompt_embeds = prompt_embeds.repeat(bs, 1, 1)
        neg_embeds = neg_embeds.repeat(bs, 1, 1)
        text_embeddings = torch.cat([neg_embeds, prompt_embeds], dim=0)
    else:
        text_embeddings = prompt_embeds.repeat(bs, 1, 1)

    imgs_t = torch.stack(
        [
            (optim_utils.transform_img(im, target_size=target_size)
             if target_size is not None else optim_utils.transform_img(im))
            for im in images
        ],
        dim=0,
    ).to(device)
    imgs_t = imgs_t.to(dtype=text_embeddings.dtype)

    if decoder_inv:
        latents_list = []
        for k in range(bs):
            with torch.enable_grad():
                z = pipe.decoder_inv(imgs_t[k:k+1])
            latents_list.append(z)
        latents = torch.cat(latents_list, dim=0)
    else:
        with torch.no_grad():
            latents = pipe.get_image_latents(imgs_t, sample=False)

    if debug:
        img_sizes = [getattr(im, "size", None) for im in images]
        print(f"[DBG] target_size={target_size} img_sizes={img_sizes}")
        print(f"[DBG] imgs_t.shape={tuple(imgs_t.shape)} dtype={imgs_t.dtype} device={imgs_t.device}")
        print(f"[DBG] latents.shape={tuple(latents.shape)} dtype={latents.dtype} device={latents.device}")
        print(f"[DBG] per_sample_n={int(latents[0].numel())}")

    with torch.no_grad():
        reversed_latents = pipe.forward_diffusion(
            latents=latents,
            text_embeddings=text_embeddings,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_inference_steps),
            inverse_opt=(inv_order != 0),
            inv_order=int(inv_order),
        )

    return reversed_latents


# -------------------------
# Detect/decode per latent
# -------------------------

def _extract_detect_score(out: Any) -> Tuple[bool, float]:
    if isinstance(out, tuple) and len(out) >= 1:
        det = bool(out[0])
        if len(out) >= 2:
            try:
                sc = float(out[1])
            except Exception:
                sc = float("nan")
        else:
            sc = float("nan")
        return det, sc
    return bool(out), float("nan")


def _decode_bits(prc_lib, decoding_key, post: Any) -> Optional[np.ndarray]:
    # Keep your common setting (max_bp_iter=500)
    try:
        bits = prc_lib.Decode(decoding_key, post, print_progress=False, max_bp_iter=2000)
    except TypeError:
        try:
            bits = prc_lib.Decode(decoding_key, post, max_bp_iter=2000)
        except Exception:
            return None
    except Exception:
        return None

    if bits is None:
        return None
    arr = np.array(bits, dtype=np.int64).reshape(-1)
    return arr


def detect_decode_batch(
    prc_lib,
    prc_gaussians,
    decoding_key,
    reversed_latents: torch.Tensor,
    var: float,
    fpr: float,
    key_n_expected: Optional[int],
    expected_bits: np.ndarray,
) -> Tuple[List[bool], List[float], List[float], List[bool]]:
    """
    Returns:
      detected_flags, scores, bit_accs, decode_ok
    bit_acc: prefix accuracy vs expected_bits (len=message_length).
             decode_fail -> 0.0 (so summary is defined)
    """
    bs = reversed_latents.shape[0]
    flags: List[bool] = []
    scores: List[float] = []
    bit_accs: List[float] = []
    decode_ok: List[bool] = []

    for i in range(bs):
        z = reversed_latents[i].to(dtype=torch.float64).flatten().cpu()
        if key_n_expected is not None and int(z.numel()) != int(key_n_expected):
            raise RuntimeError(
                f"PRC dimension mismatch: key expects n={key_n_expected}, "
                f"but inverted latents have n={int(z.numel())}. "
                f"Likely resolution mismatch; ensure target_size inference works."
            )

        post = prc_gaussians.recover_posteriors(z, variances=float(var))
        if not isinstance(post, torch.Tensor):
            post = torch.from_numpy(np.array(post))

        out = prc_lib.Detect(decoding_key, post, false_positive_rate=float(fpr))
        det, sc = _extract_detect_score(out)
        flags.append(det)
        scores.append(sc)

        bits = _decode_bits(prc_lib, decoding_key, post)
        if bits is None:
            decode_ok.append(False)
            bit_accs.append(0.0)
        else:
            decode_ok.append(True)
            bit_accs.append(_prefix_bit_acc(bits, expected_bits))

    return flags, scores, bit_accs, decode_ok


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--run_dir", required=True, help="Run directory that contains sliced/*.png")
    ap.add_argument("--model_id", default=DEFAULT_MODEL_ID, help=f"SD model path/HF id (default: {DEFAULT_MODEL_ID})")
    ap.add_argument("--prc_repo", default=DEFAULT_PRC_REPO, help=f"PRC-Watermark-main path (default: {DEFAULT_PRC_REPO})")
    ap.add_argument("--cache_dir", default=None, help="HF cache dir (optional)")

    ap.add_argument("--key_path", default=DEFAULT_KEY_PATH, help=f"PRC key pkl (default: {DEFAULT_KEY_PATH})")

    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    ap.add_argument("--solver_order", type=int, default=DEFAULT_SOLVER_ORDER)
    ap.add_argument("--inv_order", type=int, default=DEFAULT_INV_ORDER)
    ap.add_argument("--decoder_inv", type=int, default=DEFAULT_DECODER_INV)
    ap.add_argument("--dtype", default=DEFAULT_DTYPE)
    ap.add_argument("--inv_bs", type=int, default=DEFAULT_INV_BS)

    ap.add_argument("--var", type=float, default=DEFAULT_VAR)
    ap.add_argument("--fpr", type=float, default=DEFAULT_FPR)

    ap.add_argument("--master_key", default=DEFAULT_MASTER_KEY)
    ap.add_argument("--message_length", type=int, default=DEFAULT_MESSAGE_LENGTH)

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0=all")

    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    sliced_dir = os.path.join(run_dir, "sliced")
    if not os.path.isdir(sliced_dir):
        raise FileNotFoundError(f"Missing sliced dir: {sliced_dir}")

    _add_paths(args.prc_repo)
    InversableStableDiffusionPipeline, optim_utils, prc_lib, prc_gaussians = _import_modules()

    # load key
    decoding_key, key_meta = _load_decoding_key(args.key_path)
    key_n, key_hlat, key_target = _infer_key_n_and_target_size(decoding_key)

    # expected bits
    expected_bits = _expected_bits_from_master_key(args.master_key, int(args.message_length))

    # collect images
    img_paths = sorted(glob.glob(os.path.join(sliced_dir, "*.png")))
    if args.limit and args.limit > 0:
        img_paths = img_paths[: int(args.limit)]
    if not img_paths:
        raise FileNotFoundError(f"No PNGs found in {sliced_dir}")

    # output paths
    out_per = os.path.join(run_dir, "detect_prc_per_image.csv")
    out_sum = os.path.join(run_dir, "detect_prc_summary.csv")

    # build pipe
    torch_dtype = _resolve_torch_dtype(args.dtype)
    pipe = _make_pipe(args.model_id, torch_dtype=torch_dtype, cache_dir=args.cache_dir, solver_order=args.solver_order)

    print("[CFG] run_dir=", run_dir)
    print("[CFG] images=", len(img_paths), "inv_bs=", args.inv_bs)
    print("[CFG] steps=", args.steps, "guidance=", args.guidance, "dtype=", str(torch_dtype).replace("torch.", ""))
    print("[CFG] var=", args.var, "fpr=", args.fpr)
    print("[KEY] file=", key_meta.get("key_file"))
    if key_n is not None:
        print("[KEY] expected_n=", key_n, "target_size=", key_target)

    # write per-image header
    with open(out_per, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "image_path",
            "detected",
            "score",
            "bit_acc",
            "decode_ok",
            "steps",
            "guidance",
            "var",
            "fpr",
            "key_path",
            "master_key",
            "message_length",
        ])

    all_detect: List[int] = []
    all_bitacc: List[float] = []
    all_bitacc_decodeok: List[float] = []
    n = len(img_paths)

    fixed_prompt = ""  # unknown prompt mode: empty prompt

    for i in range(0, n, int(args.inv_bs)):
        batch_paths = img_paths[i:i + int(args.inv_bs)]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        batch_names = [os.path.basename(p) for p in batch_paths]

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

        flags, scores, bit_accs, decode_ok = detect_decode_batch(
            prc_lib=prc_lib,
            prc_gaussians=prc_gaussians,
            decoding_key=decoding_key,
            reversed_latents=reversed_latents,
            var=float(args.var),
            fpr=float(args.fpr),
            key_n_expected=key_n,
            expected_bits=expected_bits,
        )

        with open(out_per, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for j, pth in enumerate(batch_paths):
                det_i = int(bool(flags[j]))
                sc_i = float(scores[j])
                ba_i = float(bit_accs[j])
                ok_i = int(bool(decode_ok[j]))

                all_detect.append(det_i)
                all_bitacc.append(ba_i)
                if ok_i:
                    all_bitacc_decodeok.append(ba_i)

                idx = i + j
                print(f"[{idx+1:5d}/{n}] {os.path.basename(pth)}  detected={det_i}  score={sc_i:.4g}  bit_acc={ba_i:.3f}  decode_ok={ok_i}")

                w.writerow([
                    pth,
                    det_i,
                    sc_i,
                    ba_i,
                    ok_i,
                    int(args.steps),
                    float(args.guidance),
                    float(args.var),
                    float(args.fpr),
                    args.key_path,
                    args.master_key,
                    int(args.message_length),
                ])

    detect_rate = float(np.mean(all_detect)) if all_detect else 0.0
    bit_acc_mean_all = float(np.mean(all_bitacc)) if all_bitacc else 0.0
    bit_acc_mean_decodeok = float(np.mean(all_bitacc_decodeok)) if all_bitacc_decodeok else 0.0
    decode_ok_rate = float(len(all_bitacc_decodeok) / max(1, len(all_bitacc)))

    with open(out_sum, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run_dir",
            "num_images",
            "detect_rate",
            "bit_acc_mean_all",
            "bit_acc_mean_decode_ok",
            "decode_ok_rate",
            "steps",
            "guidance",
            "var",
            "fpr",
            "key_path",
            "master_key",
            "message_length",
            "model_id",
        ])
        w.writerow([
            run_dir,
            int(n),
            detect_rate,
            bit_acc_mean_all,
            bit_acc_mean_decodeok,
            decode_ok_rate,
            int(args.steps),
            float(args.guidance),
            float(args.var),
            float(args.fpr),
            args.key_path,
            args.master_key,
            int(args.message_length),
            args.model_id,
        ])

    print(f"[DONE] per-image: {out_per}")
    print(f"[DONE] summary:  {out_sum}")
    print(f"[SUM]  detect_rate={detect_rate:.4f}  bit_acc_mean_all={bit_acc_mean_all:.4f}  bit_acc_mean_decode_ok={bit_acc_mean_decodeok:.4f}  decode_ok_rate={decode_ok_rate:.4f}")


if __name__ == "__main__":
    main()
