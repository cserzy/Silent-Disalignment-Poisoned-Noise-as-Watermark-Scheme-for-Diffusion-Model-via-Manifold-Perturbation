#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aligned dual-mode T2SMark detector (NEW MAINLINE):
  (A) zT16 path: load [16,4,64,64] z_T tensor -> decode
  (B) image path: image -> inversion -> decode

Key changes vs old detect_t2s_dual.py:
  - NO reverse_bits, NO int_to_bits ambiguity.
  - Always use official cluster-style bits stored in cluster_meta_pt (torch .pt dict).
  - combo_id is parsed from filename Pxx_yy.png, yy in [0..15].
  - Keeps oracle session_key decode for debugging (acc_msg_oracle).

Prereq:
  export PYTHONPATH=$PYTHONPATH:/home/yancy/work/dm_backdoor_latent_space/third_party/T2SMark_official
"""

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import torch
from tqdm import tqdm

# ====== Workaround for cuDNN scaled_dot_product_attention fallback ======
import torch.nn.functional as F


def sdpa_fallback(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None):
    d = query.size(-1)
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    attn = torch.matmul(query, key.transpose(-2, -1)) * scale
    if attn_mask is not None:
        attn = attn + attn_mask
    if is_causal:
        Lq = query.size(-2)
        Lk = key.size(-2)
        causal_mask = torch.full((Lq, Lk), float("-inf"), device=attn.device, dtype=attn.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        attn = attn + causal_mask
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, value)
    return out


F.scaled_dot_product_attention = sdpa_fallback
# ====== end workaround ======

# Official imports (require PYTHONPATH points to T2SMark_official repo root)
import src.utils as utils
from src.t2s import T2SMark
from src.inversion.inverse_stable_diffusion import InversableStableDiffusionPipeline


# -----------------------------
# Helpers: parsing / loading
# -----------------------------
def parse_combo_id_from_name(name: str) -> Optional[int]:
    """
    Expect filename like P00_03.png or P12_15.png
    Return yy as int. If not matched, return None.
    """
    m = re.search(r"(?i)\bP\d{2}_(\d{2})\b", Path(name).stem)
    if not m:
        return None
    return int(m.group(1))


def _to_bits_tensor(x: Any, device: torch.device) -> torch.Tensor:
    """
    Accept:
      - torch.Tensor of 0/1
      - list/np array of 0/1
      - string "0101..."
    Return int32 tensor on device with values 0/1.
    """
    if isinstance(x, torch.Tensor):
        t = x.detach().to(device=device)
        if t.dtype not in (torch.int32, torch.int64, torch.int16, torch.uint8, torch.bool):
            t = t.to(torch.int32)
        else:
            t = t.to(torch.int32)
        # normalize bool to 0/1
        if t.dtype == torch.bool:
            t = t.to(torch.int32)
        return t

    if isinstance(x, str):
        s = x.strip()
        if any(c not in "01" for c in s):
            raise ValueError(f"Invalid bit string: {s[:64]}...")
        arr = torch.tensor([1 if c == "1" else 0 for c in s], dtype=torch.int32, device=device)
        return arr

    # list/np
    arr = torch.tensor(x, dtype=torch.int32, device=device)
    return arr


def load_cluster_meta(cluster_meta_pt: str, device: torch.device) -> Dict[str, Any]:
    """
    Load meta dict. Supports:
      - official cluster pt: {'master_keys','keys','msgs','settings','latents',...}
      - custom meta pt: support alternate key naming conventions.

    Compatibility mode:
      If cluster_meta_pt does NOT contain master/session/msg bits but contains 'cluster_pt',
      we will load the official cluster pt pointed by pack['cluster_pt'] to recover
      {'master_keys','keys','msgs','settings'}.

    Returns normalized:
      master_keys: [K, key_len] int32
      session_keys: [K, key_len] int32
      msgs: [K, msg_len] int32
      settings: dict
      tau: float
      key_channel_idx: int
      key_length, msg_length
      K
    """
    pack = torch.load(cluster_meta_pt, map_location="cpu")
    if not isinstance(pack, dict):
        raise TypeError(f"cluster_meta_pt must be a dict .pt, got: {type(pack)}")

    settings = pack.get("settings", {})
    if not isinstance(settings, dict):
        settings = {}

    def _load_official_cluster(cluster_pt: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        cluster_pack = torch.load(cluster_pt, map_location="cpu")
        if not isinstance(cluster_pack, dict):
            raise TypeError(f"cluster_pt must be a dict .pt, got: {type(cluster_pack)}")
        if not ("master_keys" in cluster_pack and "keys" in cluster_pack and "msgs" in cluster_pack):
            raise KeyError(f"cluster_pt missing required keys. got={list(cluster_pack.keys())}")
        settings_off = cluster_pack.get("settings", {})
        if not isinstance(settings_off, dict):
            settings_off = {}
        mk_ = _to_bits_tensor(cluster_pack["master_keys"], device)
        sk_ = _to_bits_tensor(cluster_pack["keys"], device)
        mg_ = _to_bits_tensor(cluster_pack["msgs"], device)
        return mk_, sk_, mg_, settings_off

    # ---- 1) primary path: standard keys present ----
    if "master_keys" in pack and "keys" in pack and "msgs" in pack:
        master_keys = _to_bits_tensor(pack["master_keys"], device)
        session_keys = _to_bits_tensor(pack["keys"], device)
        msgs = _to_bits_tensor(pack["msgs"], device)
    else:
        # ---- 2) tolerate alternate naming ----
        mk = pack.get("master_key_bits") or pack.get("master_keys_bits") or pack.get("master_key")
        sk = pack.get("session_key_bits") or pack.get("session_keys") or pack.get("keys")
        mg = pack.get("msg_bits") or pack.get("msgs_bits") or pack.get("msgs")

        if mk is not None and sk is not None and mg is not None:
            master_keys = _to_bits_tensor(mk, device)
            session_keys = _to_bits_tensor(sk, device)
            msgs = _to_bits_tensor(mg, device)
        else:
            # ---- 3) compatibility: ablation meta only stores 'cluster_pt' ----
            cluster_pt = pack.get("cluster_pt", None)
            if cluster_pt is None:
                raise KeyError(f"cluster_meta_pt missing keys. got={list(pack.keys())}")
            cluster_pt = str(cluster_pt)
            if not os.path.isfile(cluster_pt):
                raise FileNotFoundError(f"cluster_pt not found: {cluster_pt}")
            master_keys, session_keys, msgs, settings_off = _load_official_cluster(cluster_pt)
            # meta/settings overrides official settings
            settings = {**settings_off, **settings}

    # ---- Ensure 2D [K, L] ----
    if master_keys.ndim == 1:
        master_keys = master_keys.unsqueeze(0)
    if session_keys.ndim == 1:
        session_keys = session_keys.unsqueeze(0)
    if msgs.ndim == 1:
        msgs = msgs.unsqueeze(0)

    # ---- Hyper-params ----
    tau = float(settings.get("tau", pack.get("tau", pack.get("t2s_tau", 0.674))))

    cluster_settings = pack.get("cluster_settings", {})
    if not isinstance(cluster_settings, dict):
        cluster_settings = {}
    key_channel_idx = int(
        settings.get(
            "key_channel_idx",
            pack.get("key_channel_idx", cluster_settings.get("key_channel_idx", 0)),
        )
    )

    key_length = int(settings.get("key_length", master_keys.shape[-1]))
    msg_length = int(settings.get("msg_length", msgs.shape[-1]))

    # ---- Sanity ----
    if master_keys.numel() == 0 or session_keys.numel() == 0 or msgs.numel() == 0:
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
    """
    Match official decode(): use master key to detect key channel, recover session key,
    then decode msg with recovered session key (and also oracle session key for debug).
    """
    msg_channel_idx = [i for i in range(4) if i != key_channel_idx]
    reversed_key_channel = post_reversed_latents[0, key_channel_idx, :, :]
    reversed_msg_channel = post_reversed_latents[0, msg_channel_idx, :, :]

    fake_key = 1 - master_key  # official-style negative key

    # detection=True returns (bits, norm1)
    _, norm1_no_w = t2s_key.decode(reversed_key_channel, fake_key, detection=True)
    reversed_key, norm1_w = t2s_key.decode(reversed_key_channel, master_key, detection=True)

    # Decode the message with the recovered key and the oracle key
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


def maybe_resize_pil(img, target: Optional[int]):
    if target is None:
        return img
    if img.size[0] == target and img.size[1] == target:
        return img
    return img.resize((target, target))


def compute_auc_block(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute AUC using norm1_no_w vs norm1_w similar to official reporting.
    """
    try:
        from sklearn import metrics
        no_w = [it["robustness"]["norm1_no_w"] for it in items]
        w = [it["robustness"]["norm1_w"] for it in items]
        preds = no_w + w
        labels = [0] * len(no_w) + [1] * len(w)
        fpr, tpr, _ = metrics.roc_curve(labels, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # Approximate balanced-accuracy proxy
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
        "name", "mode", "image", "zT_idx",
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
                "detected": r["robustness"]["detected"],
                "norm1_no_w": r["robustness"]["norm1_no_w"],
                "norm1_w": r["robustness"]["norm1_w"],
                "acc_key": r["robustness"]["acc_key"],
                "acc_msg": r["robustness"]["acc_msg"],
                "acc_msg_oracle": r["robustness"]["acc_msg_oracle"],
            }
            w.writerow(flat)


def main():
    ap = argparse.ArgumentParser()

    # ---- inputs ----
    ap.add_argument("--cluster_meta_pt", type=str,
                    default="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_T2S_w_att_meta.pt",
                    help="Meta .pt dict containing master_keys/session_keys/msgs/settings (default points to your experiment folder).")

    ap.add_argument("--zT16_pt", type=str, default=None,
                    help="Optional: [16,4,64,64] zT tensor .pt, for direct decode path.")

    ap.add_argument("--images_glob", type=str, default=None,
                    help="Optional: glob for images (script will glob itself), e.g. /path/to/sliced/*.png")

    ap.add_argument("--model_id", type=str, default=None,
                    help="Stable Diffusion diffusers path or HF id (required if --images_glob is used).")

    # ---- inversion options ----
    ap.add_argument("--num_inversion_steps", type=int, default=10,
                    help="Inversion steps (default 10, aligns with official option defaults).")
    ap.add_argument("--inv_guidance", type=float, default=1.0,
                    help="Guidance scale for inversion. Default 1.0 (prompt-free).")
    ap.add_argument("--use_prompt", action="store_true",
                    help="If set, use prompt text embeddings for inversion. Default unknown-prompt (empty prompt).")
    ap.add_argument("--resize", type=int, default=512,
                    help="Resize input images before VAE encode. Default 512.")

    # ---- misc ----
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--fp16", action="store_true", default=True,
                    help="Use fp16 pipeline weights/compute (recommended).")
    ap.add_argument("--compute_auc", action="store_true",
                    help="Compute AUC-like summary using norm1_no_w vs norm1_w.")
    ap.add_argument("--print_each", action="store_true", default=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default=None)

    args = ap.parse_args()

    if not args.images_glob and not args.zT16_pt:
        raise SystemExit("Need at least one of --images_glob or --zT16_pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script expects CUDA (official T2SMark decode uses CUDA tensors).")

    torch_dtype = torch.float16 if args.fp16 else torch.float32

    # ---- load meta (true source of keys/msg/settings) ----
    meta_pack = load_cluster_meta(args.cluster_meta_pt, device=device)
    K = meta_pack["K"]
    tau = meta_pack["tau"]
    key_channel_idx = meta_pack["key_channel_idx"]
    key_length = meta_pack["key_length"]
    msg_length = meta_pack["msg_length"]

    master_keys = meta_pack["master_keys"]         # [K, key_len]
    session_keys = meta_pack["session_keys"]       # [K, key_len]
    msgs = meta_pack["msgs"]                       # [K, msg_len]

    # ---- build T2S objects ----
    t2s_cache: Dict[Tuple[str, int, float], T2SMark] = {}
    t2s_key = build_t2s_objects(t2s_cache, "key", key_length, tau, (1, 64, 64))
    t2s_msg = build_t2s_objects(t2s_cache, "msg", msg_length, tau, (3, 64, 64))

    # ---- (optional) inversion pipe ----
    pipe = None
    null_text_embeddings = None
    if args.images_glob:
        if not args.model_id:
            raise SystemExit("--model_id is required when using --images_glob")
        try:
            pipe = InversableStableDiffusionPipeline.from_pretrained(
                args.model_id,
                torch_dtype=torch_dtype,
                revision="fp16" if args.fp16 else None,
            ).to(device)
        except Exception:
            pipe = InversableStableDiffusionPipeline.from_pretrained(
                args.model_id,
                torch_dtype=torch_dtype,
            ).to(device)
        pipe.set_progress_bar_config(disable=True)
        null_text_embeddings = pipe.encode_prompt("", device, 1, False, None)[0]

    rows: List[Dict[str, Any]] = []
    results: Dict[str, Any] = {}

    # ---- Path A: zT16 decode ----
    if args.zT16_pt:
        zT16 = torch.load(args.zT16_pt, map_location="cpu")
        if isinstance(zT16, dict):
            # Compatibility: some attack scripts store {"zT": tensor}
            if "zT" in zT16:
                zT16 = zT16["zT"]
            # Compatibility: official cluster files may store {"latents": tensor}
            elif "latents" in zT16:
                zT16 = zT16["latents"]
            else:
                raise KeyError(f"zT16 dict has keys={list(zT16.keys())}, expected 'zT' or 'latents'.")

        if not isinstance(zT16, torch.Tensor):
            raise TypeError(f"Unexpected zT16 type: {type(zT16)}")
        if zT16.ndim != 4 or zT16.shape[1] != 4:
            raise ValueError(f"Expected zT16 shape [K,4,64,64], got {tuple(zT16.shape)}")
        if zT16.shape[0] < K:
            # allow meta K larger than zT16, but we only decode available ones
            useK = zT16.shape[0]
        else:
            useK = K

        zT16 = zT16[:useK].to(device=device, dtype=torch_dtype)

        for i in tqdm(range(useK), desc="detect(zT16)"):
            mk = master_keys[i]
            sk = session_keys[i]
            mg = msgs[i]

            zt = zT16[i:i+1]  # [1,4,64,64]
            rob = decode_latents(
                zt, key_channel_idx=key_channel_idx,
                t2s_key=t2s_key, t2s_msg=t2s_msg,
                master_key=mk, session_key=sk, msg=mg
            )
            name = f"zT16[{i:02d}]"
            item = {
                "name": name,
                "mode": "zT16",
                "zT_idx": i,
                "image": None,
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
                print(f"[ZT] {name} detected={rob['detected']:.0f} "
                      f"norm1_no_w={rob['norm1_no_w']:.3f} norm1_w={rob['norm1_w']:.3f} "
                      f"acc_key={rob['acc_key']:.3f} acc_msg={rob['acc_msg']:.3f} "
                      f"acc_msg_oracle={rob['acc_msg_oracle']:.3f}", flush=True)

    # ---- Path B: image inversion decode ----
    if args.images_glob:
        from PIL import Image

        images_glob = args.images_glob
        if os.path.isdir(images_glob):
            images_glob = os.path.join(images_glob, "*.png")
        img_paths = sorted([Path(p) for p in glob.glob(images_glob)])
        if args.max_images and args.max_images > 0:
            img_paths = img_paths[:args.max_images]
        if not img_paths:
            raise FileNotFoundError(f"No images matched: {images_glob}")

        for img_path in tqdm(img_paths, desc="detect(images)"):
            combo_id = parse_combo_id_from_name(img_path.name)
            if combo_id is None:
                # fallback: try _yy style
                m = re.search(r"_(\d{2})\b", img_path.stem)
                combo_id = int(m.group(1)) if m else 0

            if combo_id >= K:
                # skip unseen combos
                if args.print_each:
                    print(f"[SKIP] {img_path.name}: combo_id={combo_id} >= K={K}", flush=True)
                continue

            mk = master_keys[combo_id]
            sk = session_keys[combo_id]
            mg = msgs[combo_id]

            pil = Image.open(img_path).convert("RGB")
            pil = maybe_resize_pil(pil, args.resize)
            image_tensor = utils.to_tensor(pil).to(device).to(torch_dtype)
            latents = pipe.get_image_latents(image_tensor, sample=False)

            if args.use_prompt:
                # Prompt-conditioned inversion can be inserted here if needed.
                # Default to prompt-free inversion with an empty prompt.
                prompt = ""
                text_embeddings = pipe.encode_prompt(prompt, device, 1, False, None)[0]
            else:
                text_embeddings = null_text_embeddings

            reversed_latents = pipe.naive_forward_diffusion(
                latents=latents.to(torch_dtype),
                text_embeddings=text_embeddings.to(torch_dtype),
                num_inference_steps=int(args.num_inversion_steps),
                guidance_scale=float(args.inv_guidance),
            )

            rob = decode_latents(
                reversed_latents, key_channel_idx=key_channel_idx,
                t2s_key=t2s_key, t2s_msg=t2s_msg,
                master_key=mk, session_key=sk, msg=mg
            )

            name = img_path.name
            item = {
                "name": name,
                "mode": "image_inversion",
                "zT_idx": combo_id,
                "image": str(img_path),
                "robustness": rob,
                "meta": {
                    "combo_id": combo_id,
                    "tau": tau,
                    "key_channel_idx": key_channel_idx,
                    "num_inversion_steps": int(args.num_inversion_steps),
                    "inv_guidance": float(args.inv_guidance),
                    "use_prompt": bool(args.use_prompt),
                },
            }
            results[name] = item
            rows.append(item)
            if args.print_each:
                print(f"[IMG] {name} combo={combo_id:02d} detected={rob['detected']:.0f} "
                      f"norm1_no_w={rob['norm1_no_w']:.3f} norm1_w={rob['norm1_w']:.3f} "
                      f"acc_key={rob['acc_key']:.3f} acc_msg={rob['acc_msg']:.3f} "
                      f"acc_msg_oracle={rob['acc_msg_oracle']:.3f}", flush=True)

    # ---- summary ----
    if args.compute_auc and rows:
        results["summary_all"] = compute_auc_block(rows)
        zt_rows = [r for r in rows if r.get("mode") == "zT16"]
        img_rows = [r for r in rows if r.get("mode") == "image_inversion"]
        if zt_rows:
            results["summary_zT16"] = compute_auc_block(zt_rows)
        if img_rows:
            results["summary_image_inversion"] = compute_auc_block(img_rows)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if args.out_csv:
        write_csv(rows, args.out_csv)

    print(f"[OK] wrote json: {out_json}", flush=True)
    if args.out_csv:
        print(f"[OK] wrote csv:  {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
