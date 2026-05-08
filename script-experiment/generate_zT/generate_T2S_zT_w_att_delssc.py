#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generate_T2S_zT_w_att_delssc.py

T2S ablation: remove SSC-based B_sens estimation (delssc).

Configuration:
  - Do not compute B_sens and do not apply projection.
  - In the EBS stage, set z_free = z0 ~ N(0, I) with shape (1,4,64,64).
  - Keep the remaining pipeline unchanged: EBS mixing -> SPS (single tail repair) -> save zT with shape [K,4,64,64].

Outputs (default: --outdir/latents_experiment):
  - generate_T2S_w_att_delssc.pt        # zT_refined, [K,4,64,64]
  - generate_T2S_w_att_delssc_pre.pt    # zT_pre,    [K,4,64,64]
  - generate_T2S_w_att_delssc_meta.pt   # metadata including supports/signs
  - generate_T2S_w_att_delssc_meta.json # lightweight metadata

Notes:
  - The retained --model_id/--prompts/--ssc_* arguments are preserved for interface compatibility and are not used in the delssc setting.
"""

import argparse
import gc
import json
import math
import os
from typing import Any, Dict, List

import torch


# -------------------------
# small utils
# -------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def t2s_tail_repair_once(
    z_4chw: torch.Tensor,           # [1,4,64,64] float32 on device
    support_idx_cpu: torch.Tensor,  # cpu Long[k]
    sign_cpu: torch.Tensor,         # cpu Int8[k] with +/-1
    tau: float,
) -> torch.Tensor:
    """Single repair pass for the gradient-free SPS variant."""
    z = z_4chw.clone()
    z_flat = z.view(-1)
    idx = support_idx_cpu.to(device=z.device, dtype=torch.long)
    sgn = sign_cpu.to(device=z.device, dtype=torch.float32)

    vals = z_flat.index_select(0, idx)
    mag = vals.abs()
    mag = torch.where(mag < float(tau), mag + float(tau), mag)
    fixed = sgn * mag
    z_flat.index_copy_(0, idx, fixed)
    return z


def main():
    ap = argparse.ArgumentParser()

    # Legacy CLI compatibility; unused in delssc
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--prompts", type=str, required=True)
    ap.add_argument("--ssc_cal_N", type=int, default=12)
    ap.add_argument("--ssc_energy_ratio", type=float, default=0.9)
    ap.add_argument("--ssc_mini_steps", type=int, default=6)
    ap.add_argument("--ssc_guidance", type=float, default=7.5)
    ap.add_argument("--reuse_ssc", type=int, default=1)

    # 实际需要
    ap.add_argument("--cluster_pt", type=str, required=True, help="official T2S cluster pt（dict 含 latents）")
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--seed", type=int, default=12345)

    # attack mixing
    ap.add_argument("--lam1", type=float, default=0.9, help="zT = lam1*z_wm + lam2*z_free, lam2=sqrt(1-lam1^2)")

    # t2s params (read from cluster settings or fallback to CLI)
    ap.add_argument("--t2s_tau", type=float, default=0.674)
    ap.add_argument("--key_channel_idx", type=int, default=0)

    # 兼容旧命令（delssc 不使用）
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--xformers", action="store_true")
    ap.add_argument("--attn_slicing", action="store_true")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_dir(args.outdir)
    lat_exp_dir = os.path.join(args.outdir, "latents_experiment")
    ensure_dir(lat_exp_dir)

    # -------------------------
    # load cluster
    # -------------------------
    pack = torch.load(args.cluster_pt, map_location="cpu")
    if isinstance(pack, torch.Tensor):
        # If a tensor is provided directly, treat it as latents
        z_wm_all = pack.float()
        settings: Dict[str, Any] = {}
    elif isinstance(pack, dict):
        if "latents" not in pack:
            raise KeyError(f"cluster_pt dict missing key 'latents': {list(pack.keys())}")
        z_wm_all = pack["latents"].float()
        settings = pack.get("settings", {}) or {}
    else:
        raise TypeError(f"Unsupported cluster_pt type: {type(pack)}")

    K = int(args.K)
    if z_wm_all.shape[0] < K:
        raise ValueError(f"cluster has only {z_wm_all.shape[0]} latents, but K={K}")

    z_wm_all = z_wm_all[:K].contiguous()

    tau = float(settings.get("tau", args.t2s_tau))
    key_channel_idx = int(settings.get("key_channel_idx", args.key_channel_idx))

    # tail support/sign from the *watermarked* latent itself
    supports: List[torch.Tensor] = []
    signs: List[torch.Tensor] = []
    support_sizes: List[int] = []

    for i in range(K):
        z = z_wm_all[i].view(-1)
        idx = torch.nonzero(z.abs() >= tau, as_tuple=False).view(-1).to(torch.long)
        sgn = torch.sign(z.index_select(0, idx)).to(torch.int8)
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)  # Map 0 -> +1 under numerical edge cases
        supports.append(idx.cpu())
        signs.append(sgn.cpu())
        support_sizes.append(int(idx.numel()))

    # -------------------------
    # EBS mixing (delssc)
    # -------------------------
    lam1 = float(args.lam1)
    if not (0.0 <= lam1 <= 1.0):
        raise ValueError(f"--lam1 must be in [0,1], got {lam1}")
    lam2 = math.sqrt(max(0.0, 1.0 - lam1 * lam1))

    zT_pre_list: List[torch.Tensor] = []
    zT_ref_list: List[torch.Tensor] = []

    set_seed(int(args.seed))

    for i in range(K):
        z_wm = z_wm_all[i : i + 1].to(device=device, dtype=torch.float32)  # [1,4,64,64]

        # delssc: z_free is just fresh Gaussian, no projection
        g = torch.Generator(device=device)
        g.manual_seed(int(args.seed) + i)
        z_free = torch.randn((1, 4, 64, 64), generator=g, device=device, dtype=torch.float32)

        zT_pre = lam1 * z_wm + lam2 * z_free

        # SPS: ONLY ONCE repair
        zT_ref = t2s_tail_repair_once(
            z_4chw=zT_pre,
            support_idx_cpu=supports[i],
            sign_cpu=signs[i],
            tau=tau,
        )

        # self-check
        with torch.no_grad():
            idx = supports[i].to(device=device, dtype=torch.long)
            sgn = signs[i].to(device=device, dtype=torch.float32)
            vals = zT_ref.view(-1).index_select(0, idx)
            sign_ok = float((torch.sign(vals) == sgn).float().mean().item())
            amp_ok = float((vals.abs() >= tau).float().mean().item())

        print(f"[SANITY][{i:02d}] support={support_sizes[i]}  sign_ok={sign_ok:.2f}  amp_ok={amp_ok:.2f}")

        zT_pre_list.append(zT_pre.detach().cpu())
        zT_ref_list.append(zT_ref.detach().cpu())

        del z_wm, z_free, zT_pre, zT_ref
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    zT_pre_pack = torch.cat(zT_pre_list, dim=0).float()
    zT_ref_pack = torch.cat(zT_ref_list, dim=0).float()

    out_pt = os.path.join(lat_exp_dir, "generate_T2S_w_att_delssc.pt")
    out_pre_pt = os.path.join(lat_exp_dir, "generate_T2S_w_att_delssc_pre.pt")
    meta_pt = os.path.join(lat_exp_dir, "generate_T2S_w_att_delssc_meta.pt")
    meta_json = os.path.join(lat_exp_dir, "generate_T2S_w_att_delssc_meta.json")

    torch.save(zT_ref_pack, out_pt)
    torch.save(zT_pre_pack, out_pre_pt)

    meta: Dict[str, Any] = {
        "method": "T2S",
        "variant": "att_delssc_once_repair",
        "K": K,
        "seed": int(args.seed),
        "tau": tau,
        "key_channel_idx": key_channel_idx,
        "lam1": lam1,
        "lam2": lam2,
        "support_sizes": support_sizes,
        "supports": supports,
        "signs": signs,
        "cluster_pt": os.path.abspath(args.cluster_pt),
        "cluster_settings": settings,
        "delssc": True,
        "note": "SSC disabled: z_free is fresh Gaussian (no projection to B_sens).",
        # keep CLI for bookkeeping (even if unused)
        "cli_args_unused_but_kept": {
            "model_id": args.model_id,
            "prompts": args.prompts,
            "ssc_cal_N": int(args.ssc_cal_N),
            "ssc_energy_ratio": float(args.ssc_energy_ratio),
            "ssc_mini_steps": int(args.ssc_mini_steps),
            "ssc_guidance": float(args.ssc_guidance),
            "dtype": args.dtype,
            "xformers": bool(args.xformers),
            "attn_slicing": bool(args.attn_slicing),
        },
    }

    # attach official keys/msg if present
    if isinstance(pack, dict):
        for k in ["master_keys", "keys", "msgs", "fake_keys"]:
            if k in pack:
                try:
                    meta[k] = pack[k][:K].clone()
                except Exception:
                    meta[k] = pack[k]

    torch.save(meta, meta_pt)

    json_obj = {
        "method": meta["method"],
        "variant": meta["variant"],
        "K": meta["K"],
        "seed": meta["seed"],
        "tau": meta["tau"],
        "key_channel_idx": meta["key_channel_idx"],
        "lam1": meta["lam1"],
        "lam2": meta["lam2"],
        "support_sizes": meta["support_sizes"],
        "delssc": True,
        "cluster_pt": meta["cluster_pt"],
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)

    print("\n[OK] saved:")
    print(" ", out_pt)
    print(" ", out_pre_pt)
    print(" ", meta_pt)
    print(" ", meta_json)


if __name__ == "__main__":
    main()
