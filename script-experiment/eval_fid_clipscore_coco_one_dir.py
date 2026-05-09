#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate one COCO run_dir:
- CLIPScore (image, prompt) per sample, then mean/std/se/t
- FID (gen vs paired COCO real subset) using Inception-V3 pool3 (2048-d)
- FID bootstrap (paired bootstrap on indices) -> mean/std/CI

Inputs:
  run_dir/
    sliced/manifest.csv          (must contain: file, prompt_idx, prompt, ...)
  coco_map_csv                  (your 1000 mapping: prompt_idx -> file_name)
  coco_val_dir                  (COCO val2017 images directory)

Outputs (per run_dir):
  run_dir/eval_coco/
    aligned_pairs.csv
    clip_scores.csv
    clip_summary.json
    fid_summary.json
    cache/
      inception_gen.npy
      inception_real.npy
"""

import os
import csv
import json
import math
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F

# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    if "prompt_idx" not in df.columns or "file" not in df.columns:
        raise ValueError(f"manifest missing required columns. got={list(df.columns)}")
    df["prompt_idx"] = df["prompt_idx"].astype(int)
    df["file"] = df["file"].astype(str)
    return df

def read_coco_map(map_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(map_csv)
    need = {"prompt_idx", "file_name", "caption"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"coco_map_csv missing columns: {miss}")
    df["prompt_idx"] = df["prompt_idx"].astype(int)
    df["file_name"] = df["file_name"].astype(str)
    df["caption"] = df["caption"].astype(str)
    return df

def save_json(obj: Any, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def pil_load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

# -----------------------------
# CLIPScore (transformers)
# -----------------------------
@torch.no_grad()
def compute_clip_scores(
    gen_paths: List[str],
    prompts: List[str],
    clip_model_id: str,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Returns: scores shape (N,), where score = max(100*cos(img_emb, text_emb), 0)
    """
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(clip_model_id).to(device)
    proc = CLIPProcessor.from_pretrained(clip_model_id)

    model.eval()
    scores = []

    N = len(gen_paths)
    for i in range(0, N, batch_size):
        batch_paths = gen_paths[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]
        images = [pil_load_rgb(p) for p in batch_paths]

        inputs = proc(text=batch_prompts, images=images, return_tensors="pt", padding=True).to(device)
        out = model(**inputs)

        # normalize embeddings
        img_emb = out.image_embeds
        txt_emb = out.text_embeds
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        cos = (img_emb * txt_emb).sum(dim=-1)  # [-1,1]
        sc = torch.clamp(100.0 * cos, min=0.0) # [0,100]
        scores.append(sc.detach().float().cpu().numpy())

    return np.concatenate(scores, axis=0)

def summarize_1d(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    n = int(x.shape[0])
    mean = float(np.mean(x))
    var = float(np.var(x, ddof=1)) if n > 1 else 0.0
    std = float(math.sqrt(var))
    se = float(std / math.sqrt(n)) if n > 0 else 0.0
    t0 = float(mean / se) if se > 0 else float("inf")
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "var": var,
        "se": se,
        "t_stat_vs_0": t0,
    }

# -----------------------------
# FID (Inception-v3 pool3 2048d)
# -----------------------------
def get_inception(device: str, dtype: torch.dtype):
    import torchvision
    from torchvision.models import inception_v3, Inception_V3_Weights

    weights = Inception_V3_Weights.DEFAULT
    preprocess = weights.transforms()

    # Keep the default aux_logits behavior because the pretrained weights expect it.
    model = inception_v3(weights=weights, transform_input=False)
    model.fc = torch.nn.Identity()  # output 2048
    model.eval().to(device=device, dtype=dtype)

    return model, preprocess

@torch.no_grad()
def extract_inception_features(
    paths: List[str],
    device: str,
    batch_size: int = 32,
    dtype: str = "fp16",
) -> np.ndarray:
    """
    Return: features (N, 2048) float64 (for stable stats)
    """
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    model, preprocess = get_inception(device=device, dtype=torch_dtype)

    feats = []
    N = len(paths)
    for i in range(0, N, batch_size):
        batch_paths = paths[i:i+batch_size]
        imgs = [pil_load_rgb(p) for p in batch_paths]
        tens = torch.stack([preprocess(im) for im in imgs], dim=0).to(device=device, dtype=torch_dtype)

        f = model(tens)  # (B,2048)
        if hasattr(f, "logits"):  # Compatible with InceptionOutputs.
            f = f.logits
        f = f.detach().float().cpu().numpy()
        feats.append(f)

    feats = np.concatenate(feats, axis=0).astype(np.float64)
    return feats

def _cov(x: np.ndarray) -> np.ndarray:
    # x: (N,D)
    # numpy.cov expects rowvar=False for columns as variables
    return np.cov(x, rowvar=False, bias=False)

def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    """
    sqrtm for (approximately) PSD matrices via scipy if available, else eigen fallback.
    """
    try:
        import scipy.linalg
        s = scipy.linalg.sqrtm(mat)
        if np.iscomplexobj(s):
            s = s.real
        return s
    except Exception:
        # fallback: eigen decomposition (may be less stable)
        w, v = np.linalg.eig(mat)
        if np.iscomplexobj(w):
            w = w.real
        w = np.clip(w, a_min=0.0, a_max=None)
        s = (v @ np.diag(np.sqrt(w)) @ np.linalg.inv(v))
        if np.iscomplexobj(s):
            s = s.real
        return s

def fid_from_features(feats_gen: np.ndarray, feats_real: np.ndarray, eps: float = 1e-6) -> float:
    """
    Standard FID using 2048-d Inception pool3 features.
    """
    mu1 = np.mean(feats_real, axis=0)
    mu2 = np.mean(feats_gen, axis=0)
    sigma1 = _cov(feats_real)
    sigma2 = _cov(feats_gen)

    # numerical stability
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    diff = mu1 - mu2
    covmean = _sqrtm_psd(sigma1.dot(sigma2))
    tr = np.trace(sigma1 + sigma2 - 2.0 * covmean)
    fid = float(diff.dot(diff) + tr)
    return fid

def fid_bootstrap(
    feats_gen: np.ndarray,
    feats_real: np.ndarray,
    B: int = 10,
    seed: int = 12345,
    paired: bool = True,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    Paired bootstrap by default:
      sample indices with replacement, same indices for gen/real.
    """
    n = feats_gen.shape[0]
    rng = np.random.default_rng(seed)

    fids = []
    for b in range(int(B)):
        idx = rng.integers(0, n, size=n, endpoint=False)
        g = feats_gen[idx]
        r = feats_real[idx] if paired else feats_real[rng.integers(0, n, size=n, endpoint=False)]
        fids.append(fid_from_features(g, r, eps=eps))

    fids = np.asarray(fids, dtype=np.float64)
    out = summarize_1d(fids)
    out["bootstrap_B"] = int(B)
    out["paired"] = bool(paired)
    out["seed"] = int(seed)
    out["ci95_low"] = float(np.quantile(fids, 0.025))
    out["ci95_high"] = float(np.quantile(fids, 0.975))
    out["samples"] = [float(x) for x in fids.tolist()]
    return out

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="e.g., .../imgs/vis_sd14_TR_w_att_0_88_coco_seed12345")
    ap.add_argument("--coco_map_csv", type=str, required=True, help=".../prompts/coco_val2017_captions_1000.map.csv")
    ap.add_argument("--coco_val_dir", type=str, required=True, help=".../coco/images/val2017 (contains *.jpg)")
    ap.add_argument("--clip_model_id", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--clip_bs", type=int, default=32)
    ap.add_argument("--fid_bs", type=int, default=32)
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16","fp32"])
    ap.add_argument("--bootstrap_B", type=int, default=10)
    ap.add_argument("--bootstrap_seed", type=int, default=12345)
    ap.add_argument("--no_bootstrap", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    manifest = run_dir / "sliced" / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"missing manifest: {manifest}")

    eval_dir = run_dir / "eval_coco"
    cache_dir = eval_dir / "cache"
    ensure_dir(eval_dir)
    ensure_dir(cache_dir)

    # 1) align pairs
    df_m = read_manifest(manifest)
    df_map = read_coco_map(Path(args.coco_map_csv))

    df = df_m.merge(df_map[["prompt_idx","file_name","caption"]], on="prompt_idx", how="inner")
    if df.shape[0] == 0:
        raise RuntimeError("join manifest with coco_map produced 0 rows")

    # absolute paths
    df["gen_path"] = df["file"].apply(lambda p: str(Path(p)))
    coco_val_dir = Path(args.coco_val_dir)
    df["real_path"] = df["file_name"].apply(lambda fn: str(coco_val_dir / fn))
    df["prompt_used"] = df["prompt"] if "prompt" in df.columns else df["caption"]

    # basic sanity
    miss_real = df["real_path"].apply(lambda p: not Path(p).exists()).sum()
    miss_gen = df["gen_path"].apply(lambda p: not Path(p).exists()).sum()
    if miss_gen:
        raise FileNotFoundError(f"{miss_gen} generated images missing (check manifest paths)")
    if miss_real:
        raise FileNotFoundError(f"{miss_real} COCO real images missing (check coco_val_dir)")

    aligned_csv = eval_dir / "aligned_pairs.csv"
    df_out = df[["prompt_idx","gen_path","real_path","prompt_used"]].copy()
    df_out.to_csv(aligned_csv, index=False)

    gen_paths = df_out["gen_path"].tolist()
    real_paths = df_out["real_path"].tolist()
    prompts = df_out["prompt_used"].tolist()
    N = len(gen_paths)

    # 2) CLIPScore
    t0 = time.time()
    clip_scores = compute_clip_scores(
        gen_paths=gen_paths,
        prompts=prompts,
        clip_model_id=args.clip_model_id,
        device=args.device,
        batch_size=int(args.clip_bs),
    )
    clip_dt = time.time() - t0

    clip_scores_csv = eval_dir / "clip_scores.csv"
    pd.DataFrame({
        "prompt_idx": df_out["prompt_idx"].values,
        "gen_path": gen_paths,
        "clip_score": clip_scores,
    }).to_csv(clip_scores_csv, index=False)

    clip_sum = summarize_1d(clip_scores)
    clip_sum.update({
        "clip_model_id": args.clip_model_id,
        "time_sec": round(clip_dt, 4),
    })
    save_json(clip_sum, eval_dir / "clip_summary.json")

    # 3) Inception features (cache)
    gen_npy = cache_dir / "inception_gen.npy"
    real_npy = cache_dir / "inception_real.npy"

    if gen_npy.exists() and real_npy.exists():
        feats_gen = np.load(gen_npy)
        feats_real = np.load(real_npy)
        if feats_gen.shape[0] != N or feats_real.shape[0] != N:
            # cache mismatch -> recompute
            gen_npy.unlink(missing_ok=True)
            real_npy.unlink(missing_ok=True)
            feats_gen = None
            feats_real = None
    else:
        feats_gen = None
        feats_real = None

    if feats_gen is None or feats_real is None:
        t1 = time.time()
        feats_gen = extract_inception_features(gen_paths, device=args.device, batch_size=int(args.fid_bs), dtype=args.dtype)
        feats_real = extract_inception_features(real_paths, device=args.device, batch_size=int(args.fid_bs), dtype=args.dtype)
        np.save(gen_npy, feats_gen)
        np.save(real_npy, feats_real)
        feat_dt = time.time() - t1
    else:
        feat_dt = 0.0

    # 4) FID point + bootstrap
    t2 = time.time()
    fid_point = fid_from_features(feats_gen, feats_real)
    fid_dt = time.time() - t2

    fid_sum = {
        "n": N,
        "fid_point": float(fid_point),
        "inception_feature_dim": int(feats_gen.shape[1]),
        "feature_cache_recompute_time_sec": round(feat_dt, 4),
        "fid_point_time_sec": round(fid_dt, 4),
        "bootstrap": None,
    }

    if not args.no_bootstrap and int(args.bootstrap_B) > 0:
        bs = fid_bootstrap(
            feats_gen=feats_gen,
            feats_real=feats_real,
            B=int(args.bootstrap_B),
            seed=int(args.bootstrap_seed),
            paired=True,
        )
        fid_sum["bootstrap"] = bs

    save_json(fid_sum, eval_dir / "fid_summary.json")

    print("✅ DONE")
    print(f"run_dir: {run_dir}")
    print(f"aligned_pairs: {aligned_csv}")
    print(f"CLIPScore mean: {clip_sum['mean']:.4f}  std: {clip_sum['std']:.4f}")
    print(f"FID point: {fid_point:.4f}")
    if fid_sum["bootstrap"] is not None:
        b = fid_sum["bootstrap"]
        print(f"FID bootstrap mean±std: {b['mean']:.4f} ± {b['std']:.4f} (B={b['bootstrap_B']})")
        print(f"FID 95% CI: [{b['ci95_low']:.4f}, {b['ci95_high']:.4f}]")

if __name__ == "__main__":
    main()
