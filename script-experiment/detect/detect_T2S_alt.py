#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alt-specific T2S detector.

Design:
  - Keep detect_T2S.py T2S decode logic / outputs as intact as possible
  - Replace only the image inversion layer with an Alt Diffusion manual-pipe + approximate inverse path

Notes:
  - This inversion is approximate, not an official exact inverse API.
  - Safety checker is disabled by default because Alt experiments already showed it can black out images.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMInverseScheduler, PNDMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, XLMRobertaTokenizer

from image_utils import transform_img

try:
    from diffusers.pipelines.deprecated.alt_diffusion import AltDiffusionPipeline
except Exception:
    try:
        from diffusers.pipelines.alt_diffusion import AltDiffusionPipeline
    except Exception:
        from diffusers import AltDiffusionPipeline

try:
    from diffusers.pipelines.deprecated.alt_diffusion.modeling_roberta_series import (
        RobertaSeriesModelWithTransformation,
    )
except Exception:
    from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
        RobertaSeriesModelWithTransformation,
    )


def sdpa_fallback(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None):
    d = query.size(-1)
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    attn = torch.matmul(query, key.transpose(-2, -1)) * scale
    if attn_mask is not None:
        attn = attn + attn_mask
    if is_causal:
        lq = query.size(-2)
        lk = key.size(-2)
        causal_mask = torch.full((lq, lk), float("-inf"), device=attn.device, dtype=attn.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        attn = attn + causal_mask
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, value)
    return out


F.scaled_dot_product_attention = sdpa_fallback


def _append_t2s_repo_to_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    cand_roots = [
        repo_root / "third_party" / "T2SMark",
        repo_root / "third_party" / "T2SMark_official",
    ]
    for t2s_root in cand_roots:
        if t2s_root.is_dir() and str(t2s_root) not in sys.path:
            sys.path.append(str(t2s_root))


try:
    # Keep T2S decode lightweight here: importing src.utils would pull in
    # datasets/aiohttp training-time dependencies that image->zT->decode does not need.
    from src.t2s import T2SMark
except Exception:
    _append_t2s_repo_to_syspath()
    from src.t2s import T2SMark


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _list_images_under_run_dir(run_dir: Path) -> List[Path]:
    cand = run_dir / "sliced"
    root = cand if cand.is_dir() else run_dir
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files = [p for p in files if "grid" not in p.name.lower()]
    return sorted(files)


def parse_combo_id_from_name(name: str) -> Optional[int]:
    m = re.search(r"(?i)\bP\d{2}_(\d{2})\b", Path(name).stem)
    if not m:
        return None
    return int(m.group(1))


def _to_bits_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach().to(device=device)
        if t.dtype not in (torch.int32, torch.int64, torch.int16, torch.uint8, torch.bool):
            t = t.to(torch.int32)
        else:
            t = t.to(torch.int32)
        if t.dtype == torch.bool:
            t = t.to(torch.int32)
        return t

    if isinstance(x, str):
        s = x.strip()
        if any(c not in "01" for c in s):
            raise ValueError(f"Invalid bit string: {s[:64]}...")
        arr = torch.tensor([1 if c == "1" else 0 for c in s], dtype=torch.int32, device=device)
        return arr

    arr = torch.tensor(x, dtype=torch.int32, device=device)
    return arr


def load_cluster_meta(cluster_meta_pt: str, device: torch.device) -> Dict[str, Any]:
    pack = torch.load(cluster_meta_pt, map_location="cpu")
    if not isinstance(pack, dict):
        raise TypeError(f"cluster_meta_pt must be a dict .pt, got: {type(pack)}")

    settings = pack.get("settings", {}) if isinstance(pack.get("settings", {}), dict) else {}

    if "master_keys" in pack and "keys" in pack and "msgs" in pack:
        master_keys = _to_bits_tensor(pack["master_keys"], device)
        session_keys = _to_bits_tensor(pack["keys"], device)
        msgs = _to_bits_tensor(pack["msgs"], device)
    else:
        mk = pack.get("master_key_bits", None) or pack.get("master_keys_bits", None) or pack.get("master_key", None)
        sk = pack.get("session_key_bits", None) or pack.get("session_keys", None) or pack.get("keys", None)
        mg = pack.get("msg_bits", None) or pack.get("msgs_bits", None) or pack.get("msgs", None)
        if mk is None or sk is None or mg is None:
            raise KeyError(f"cluster_meta_pt missing keys. got={list(pack.keys())}")
        master_keys = _to_bits_tensor(mk, device)
        session_keys = _to_bits_tensor(sk, device)
        msgs = _to_bits_tensor(mg, device)

    if master_keys.ndim == 1:
        master_keys = master_keys.unsqueeze(0)
    if session_keys.ndim == 1:
        session_keys = session_keys.unsqueeze(0)
    if msgs.ndim == 1:
        msgs = msgs.unsqueeze(0)

    tau = float(settings.get("tau", pack.get("t2s_tau", 0.674)))
    key_channel_idx = int(settings.get("key_channel_idx", pack.get("key_channel_idx", 0)))
    key_length = int(settings.get("key_length", master_keys.shape[-1]))
    msg_length = int(settings.get("msg_length", msgs.shape[-1]))

    if master_keys.shape[0] < 1 or session_keys.shape[0] < 1 or msgs.shape[0] < 1:
        raise ValueError("Empty keys/msg in cluster_meta_pt")
    if master_keys.shape[0] != session_keys.shape[0]:
        if master_keys.shape[0] == 1:
            master_keys = master_keys.repeat(session_keys.shape[0], 1)
        else:
            raise ValueError(f"K mismatch: master_keys {master_keys.shape}, session_keys {session_keys.shape}")
    if msgs.shape[0] != session_keys.shape[0]:
        if msgs.shape[0] == 1:
            msgs = msgs.repeat(session_keys.shape[0], 1)
        else:
            raise ValueError(f"K mismatch: msgs {msgs.shape}, session_keys {session_keys.shape}")

    return {
        "master_keys": master_keys.to(torch.int32),
        "session_keys": session_keys.to(torch.int32),
        "msgs": msgs.to(torch.int32),
        "settings": settings,
        "tau": tau,
        "key_channel_idx": key_channel_idx,
        "key_length": key_length,
        "msg_length": msg_length,
        "K": int(session_keys.shape[0]),
    }


def build_t2s_objects(cache: Dict[Tuple[str, int, float], T2SMark],
                      kind: str,
                      m: int,
                      tau: float,
                      latent_shape: Tuple[int, int, int]) -> T2SMark:
    key = (kind, m, float(tau))
    if key in cache:
        return cache[key]
    obj = T2SMark(m=m, tau=tau, latent_shape=latent_shape)
    cache[key] = obj
    return obj


def decode_latents(post_reversed_latents: torch.Tensor,
                   key_channel_idx: int,
                   t2s_key: T2SMark,
                   t2s_msg: T2SMark,
                   master_key: torch.Tensor,
                   session_key: torch.Tensor,
                   msg: torch.Tensor) -> Dict[str, float]:
    msg_channel_idx = [i for i in range(4) if i != key_channel_idx]
    reversed_key_channel = post_reversed_latents[0, key_channel_idx, :, :]
    reversed_msg_channel = post_reversed_latents[0, msg_channel_idx, :, :]

    fake_key = 1 - master_key

    _, norm1_no_w = t2s_key.decode(reversed_key_channel, fake_key, detection=True)
    reversed_key, norm1_w = t2s_key.decode(reversed_key_channel, master_key, detection=True)

    reversed_msg = t2s_msg.decode(reversed_msg_channel, reversed_key)
    reversed_msg_oracle = t2s_msg.decode(reversed_msg_channel, session_key)

    acc_key = (reversed_key == session_key).float().mean()
    acc_msg = (reversed_msg == msg).float().mean()
    acc_msg_oracle = (reversed_msg_oracle == msg).float().mean()

    detected = float(norm1_w > norm1_no_w)

    return {
        "detected": float(detected),
        "norm1_no_w": float(norm1_no_w),
        "norm1_w": float(norm1_w),
        "acc_key": float(acc_key.item()),
        "acc_msg": float(acc_msg.item()),
        "acc_msg_oracle": float(acc_msg_oracle.item()),
    }


def maybe_resize_pil(img: Image.Image, target: Optional[int]):
    if target is None:
        return img
    if img.size[0] == target and img.size[1] == target:
        return img
    return img.resize((target, target))


def compute_auc_block(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        from sklearn import metrics
        no_w = [it["robustness"]["norm1_no_w"] for it in items]
        w = [it["robustness"]["norm1_w"] for it in items]
        preds = no_w + w
        labels = [0] * len(no_w) + [1] * len(w)
        fpr, tpr, _ = metrics.roc_curve(labels, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = float(np.max(1 - (fpr + (1 - tpr)) / 2))
        idx = np.where(fpr < 1e-6)[0]
        low = float(tpr[idx[-1]]) if len(idx) else float(tpr[0])
        bit_acc = float(np.mean([it["robustness"]["acc_msg"] for it in items]))
        bit_acc_oracle = float(np.mean([it["robustness"]["acc_msg_oracle"] for it in items]))
        det_rate = float(np.mean([it["robustness"]["detected"] for it in items]))
        return {
            "auc": float(auc),
            "acc": float(acc),
            "tpr@fpr<1e-6": low,
            "det_rate": det_rate,
            "bit_accuracy": bit_acc,
            "bit_accuracy_oracle": bit_acc_oracle,
            "n": int(len(items)),
        }
    except Exception as e:
        return {"error": f"AUC computation failed: {repr(e)}"}


def write_csv(rows: List[Dict[str, Any]], out_csv: str):
    import csv
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "name", "mode", "image", "zT_idx", "zT_path",
        "detected", "norm1_no_w", "norm1_w",
        "acc_key", "acc_msg", "acc_msg_oracle",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            flat = {
                "name": r.get("name", ""),
                "mode": r.get("mode", ""),
                "image": r.get("image", ""),
                "zT_idx": r.get("zT_idx", -1),
                "zT_path": r.get("zT_path", ""),
                "detected": r["robustness"]["detected"],
                "norm1_no_w": r["robustness"]["norm1_no_w"],
                "norm1_w": r["robustness"]["norm1_w"],
                "acc_key": r["robustness"]["acc_key"],
                "acc_msg": r["robustness"]["acc_msg"],
                "acc_msg_oracle": r["robustness"]["acc_msg_oracle"],
            }
            w.writerow(flat)


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
    """
    Approximate Alt inversion:
      image -> Alt VAE encode -> DDIM inverse with Alt UNet / text encoder -> approximate zT

    This is intentionally not presented as an exact inverse; it reuses the already smoke-tested
    Alt detection path from detect_GS_alt.py.
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


def _resolve_image_paths(images_glob: Optional[str], run_dir: Optional[str], img_dir: Optional[str]) -> List[Path]:
    if images_glob:
        g = images_glob
        if os.path.isdir(g):
            g = os.path.join(g, "*.png")
        return sorted([Path(p) for p in glob.glob(g)])

    base = run_dir or img_dir
    if not base:
        return []
    base_path = Path(base)
    if not base_path.exists():
        raise FileNotFoundError(f"run/img dir not found: {base}")
    return _list_images_under_run_dir(base_path)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--cluster_meta_pt", type=str,
                    default="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_T2S_w_att_meta.pt",
                    help="Meta .pt dict containing master_keys/session_keys/msgs/settings.")

    ap.add_argument("--zT16_pt", type=str, default=None,
                    help="Optional: [16,4,64,64] zT tensor .pt, for direct decode path.")
    ap.add_argument("--images_glob", type=str, default=None,
                    help="Optional glob or directory for images.")
    ap.add_argument("--run_dir", type=str, default=None,
                    help="Generation run dir; scans run_dir/sliced by default.")
    ap.add_argument("--img_dir", type=str, default=None,
                    help="Alias of run_dir for compatibility.")
    ap.add_argument("--model_id", type=str, default=None,
                    help="Alt Diffusion diffusers path or HF id (required when using image path).")

    ap.add_argument("--num_inversion_steps", type=int, default=10,
                    help="Original detect_T2S.py inversion-step arg.")
    ap.add_argument("--inv_steps", type=int, default=None,
                    help="Alt inversion steps; if set, overrides --num_inversion_steps.")
    ap.add_argument("--inv_guidance", type=float, default=1.0,
                    help="Kept for CLI compatibility; current Alt path uses prompt-free approximate inversion.")
    ap.add_argument("--use_prompt", action="store_true",
                    help="Kept for CLI compatibility; current Alt path still defaults to empty prompt.")
    ap.add_argument("--resize", type=int, default=512,
                    help="Resize input images before inversion. Default 512.")

    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--fp16", action="store_true", help="Compatibility flag; equivalent to --dtype fp16.")
    ap.add_argument("--compute_auc", action="store_true")
    ap.add_argument("--print_each", action="store_true", default=True)
    ap.add_argument("--save_zt", action="store_true", help="Save recovered zT (.pt) under out_dir/latents.")
    ap.add_argument("--disable_safety_checker", dest="disable_safety_checker", action="store_true",
                    help="Disable safety checker and feature_extractor (default).")
    ap.add_argument("--enable_safety_checker", dest="disable_safety_checker", action="store_false",
                    help="Enable safety checker and feature_extractor.")
    ap.add_argument("--alt_force_manual_pipe", action="store_true",
                    help="Reserved flag for clarity; this script always uses manual Alt pipe loading.")

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.set_defaults(disable_safety_checker=True)

    args = ap.parse_args()

    if args.fp16:
        args.dtype = "fp16"

    image_paths = _resolve_image_paths(args.images_glob, args.run_dir, args.img_dir)
    if args.max_images and args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    if not image_paths and not args.zT16_pt:
        raise SystemExit("Need at least one of image path (--images_glob/--run_dir/--img_dir) or --zT16_pt")

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)
    out_json = Path(args.out_json) if args.out_json else out_dir / "t2s_detect_alt_results.json"
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / "t2s_detect_alt_results.csv"
    lat_dir = out_dir / "latents"
    if args.save_zt:
        _ensure_dir(lat_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script expects CUDA (official T2SMark decode uses CUDA tensors).")

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    inv_steps_eff = int(args.inv_steps) if args.inv_steps is not None else int(args.num_inversion_steps)

    meta_pack = load_cluster_meta(args.cluster_meta_pt, device=device)
    k = meta_pack["K"]
    tau = meta_pack["tau"]
    key_channel_idx = meta_pack["key_channel_idx"]
    key_length = meta_pack["key_length"]
    msg_length = meta_pack["msg_length"]

    master_keys = meta_pack["master_keys"]
    session_keys = meta_pack["session_keys"]
    msgs = meta_pack["msgs"]

    t2s_cache: Dict[Tuple[str, int, float], T2SMark] = {}
    t2s_key = build_t2s_objects(t2s_cache, "key", key_length, tau, (1, 64, 64))
    t2s_msg = build_t2s_objects(t2s_cache, "msg", msg_length, tau, (3, 64, 64))

    pipe = None
    if image_paths:
        if not args.model_id:
            raise SystemExit("--model_id is required when image path detection is used.")
        pipe = build_alt_pipe(
            args.model_id,
            device=device,
            dtype=torch_dtype,
            disable_safety_checker=bool(args.disable_safety_checker),
        )
        print(
            f"[pipe] class={pipe.__class__.__name__} "
            f"safety_checker_enabled={pipe.safety_checker is not None} "
            f"vae_scale_factor={getattr(pipe, 'vae_scale_factor', 'NA')}"
        )

    rows: List[Dict[str, Any]] = []
    results: Dict[str, Any] = {}

    if args.zT16_pt:
        zT16 = torch.load(args.zT16_pt, map_location="cpu")
        if isinstance(zT16, dict):
            if "zT" in zT16:
                zT16 = zT16["zT"]
            elif "latents" in zT16:
                zT16 = zT16["latents"]
            else:
                raise KeyError(f"zT16 dict has keys={list(zT16.keys())}, expected 'zT' or 'latents'.")

        if not isinstance(zT16, torch.Tensor):
            raise TypeError(f"Unexpected zT16 type: {type(zT16)}")
        if zT16.ndim != 4 or zT16.shape[1] != 4:
            raise ValueError(f"Expected zT16 shape [K,4,64,64], got {tuple(zT16.shape)}")

        use_k = min(int(zT16.shape[0]), int(k))
        zT16 = zT16[:use_k].to(device=device, dtype=torch_dtype)

        for i in tqdm(range(use_k), desc="detect(zT16)"):
            mk = master_keys[i]
            sk = session_keys[i]
            mg = msgs[i]

            zt = zT16[i:i + 1]
            rob = decode_latents(
                zt, key_channel_idx=key_channel_idx,
                t2s_key=t2s_key, t2s_msg=t2s_msg,
                master_key=mk, session_key=sk, msg=mg,
            )
            name = f"zT16[{i:02d}]"
            item = {
                "name": name,
                "mode": "zT16",
                "zT_idx": i,
                "image": None,
                "zT_path": args.zT16_pt,
                "robustness": rob,
                "meta": {
                    "combo_id": i,
                    "tau": tau,
                    "key_channel_idx": key_channel_idx,
                    "key_length": key_length,
                    "msg_length": msg_length,
                },
            }
            results[name] = item
            rows.append(item)
            if args.print_each:
                print(
                    f"[ZT] {name} detected={rob['detected']:.0f} "
                    f"norm1_no_w={rob['norm1_no_w']:.3f} norm1_w={rob['norm1_w']:.3f} "
                    f"acc_key={rob['acc_key']:.3f} acc_msg={rob['acc_msg']:.3f} "
                    f"acc_msg_oracle={rob['acc_msg_oracle']:.3f}",
                    flush=True,
                )

    if image_paths:
        for img_path in tqdm(image_paths, desc="detect(images)"):
            combo_id = parse_combo_id_from_name(img_path.name)
            if combo_id is None:
                m = re.search(r"_(\d{2})\b", img_path.stem)
                combo_id = int(m.group(1)) if m else 0

            if combo_id >= k:
                if args.print_each:
                    print(f"[SKIP] {img_path.name}: combo_id={combo_id} >= K={k}", flush=True)
                continue

            mk = master_keys[combo_id]
            sk = session_keys[combo_id]
            mg = msgs[combo_id]

            pil = Image.open(img_path).convert("RGB")
            pil = maybe_resize_pil(pil, args.resize)

            prompt = "" if not args.use_prompt else ""
            reversed_latents = invert_one_image_to_zT_alt(
                pipe=pipe,
                img_pil=pil,
                inv_steps=inv_steps_eff,
                device=device,
                dtype=torch_dtype,
                prompt=prompt,
            )

            zt_path_str = ""
            if args.save_zt:
                zt_path = lat_dir / f"{img_path.stem}_inv_zT_alt.pt"
                torch.save(
                    {
                        "zT": reversed_latents.detach().float().cpu(),
                        "image": img_path.name,
                        "image_path": str(img_path),
                        "combo_id": int(combo_id),
                        "invert_mode": "alt_ddim_inverse_empty_prompt",
                        "approx_inverse": True,
                        "inv_steps": int(inv_steps_eff),
                        "dtype": args.dtype,
                    },
                    zt_path,
                )
                zt_path_str = str(zt_path)

            rob = decode_latents(
                reversed_latents,
                key_channel_idx=key_channel_idx,
                t2s_key=t2s_key,
                t2s_msg=t2s_msg,
                master_key=mk,
                session_key=sk,
                msg=mg,
            )

            name = img_path.name
            item = {
                "name": name,
                "mode": "image_inversion_alt",
                "zT_idx": combo_id,
                "image": str(img_path),
                "zT_path": zt_path_str,
                "robustness": rob,
                "meta": {
                    "combo_id": combo_id,
                    "tau": tau,
                    "key_channel_idx": key_channel_idx,
                    "num_inversion_steps": int(inv_steps_eff),
                    "inv_guidance": float(args.inv_guidance),
                    "use_prompt": bool(args.use_prompt),
                    "disable_safety_checker": bool(args.disable_safety_checker),
                    "approx_inverse": True,
                },
            }
            results[name] = item
            rows.append(item)

            if args.print_each:
                print(
                    f"[IMG] {name} combo={combo_id:02d} detected={rob['detected']:.0f} "
                    f"norm1_no_w={rob['norm1_no_w']:.3f} norm1_w={rob['norm1_w']:.3f} "
                    f"acc_key={rob['acc_key']:.3f} acc_msg={rob['acc_msg']:.3f} "
                    f"acc_msg_oracle={rob['acc_msg_oracle']:.3f}",
                    flush=True,
                )

    if args.compute_auc and rows:
        results["summary_all"] = compute_auc_block(rows)
        zt_rows = [r for r in rows if r.get("mode") == "zT16"]
        img_rows = [r for r in rows if r.get("mode") == "image_inversion_alt"]
        if zt_rows:
            results["summary_zT16"] = compute_auc_block(zt_rows)
        if img_rows:
            results["summary_image_inversion_alt"] = compute_auc_block(img_rows)
    else:
        img_rows = [r for r in rows if r.get("mode") == "image_inversion_alt"]
        if img_rows:
            results["summary_image_inversion_alt"] = {
                "n": int(len(img_rows)),
                "det_rate": float(np.mean([it["robustness"]["detected"] for it in img_rows])),
                "bit_accuracy": float(np.mean([it["robustness"]["acc_msg"] for it in img_rows])),
                "bit_accuracy_oracle": float(np.mean([it["robustness"]["acc_msg_oracle"] for it in img_rows])),
                "bit_accuracy_key": float(np.mean([it["robustness"]["acc_key"] for it in img_rows])),
            }

    results["run_config"] = {
        "model_id": args.model_id,
        "cluster_meta_pt": args.cluster_meta_pt,
        "run_dir": args.run_dir,
        "img_dir": args.img_dir,
        "images_glob": args.images_glob,
        "out_dir": str(out_dir),
        "out_json": str(out_json),
        "out_csv": str(out_csv),
        "dtype": args.dtype,
        "inv_steps": int(inv_steps_eff),
        "disable_safety_checker": bool(args.disable_safety_checker),
        "approx_inverse": True,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    write_csv(rows, str(out_csv))

    print(f"[OK] wrote json: {out_json}", flush=True)
    print(f"[OK] wrote csv:  {out_csv}", flush=True)
    if args.save_zt:
        print(f"[OK] saved zT:  {lat_dir}", flush=True)


if __name__ == "__main__":
    main()
