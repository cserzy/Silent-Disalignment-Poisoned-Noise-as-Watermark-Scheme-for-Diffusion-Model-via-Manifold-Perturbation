#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import torch

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
except Exception as e:
    raise RuntimeError("Please install diffusers>=0.16 and its deps.") from e


# ---------------- utils ----------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_prompts(txt: str, max_n: int) -> List[str]:
    with open(txt, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines()]
    lines = [x for x in lines if len(x) > 0]
    return lines[:max_n]


# ---------------- configs ----------------
@dataclass
class sspConfig:
    cal_N: int = 12
    energy_ratio: float = 0.90
    mini_steps: int = 6
    guidance_scale: float = 7.5


# ---------------- ssp: build basis ----------------
@torch.no_grad()
def _encode_prompt(pipe: StableDiffusionPipeline, prompt: str) -> torch.Tensor:
    # [1,77,768] like
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(pipe.device)
    enc = pipe.text_encoder(text_input_ids)[0]
    return enc


def run_ssp_build_Bsens(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    ssp_cfg: sspConfig,
    cache_pt: str,
    reuse_cache: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Return:
      B_sens: [D, d_sens] orthonormal (float32, on CPU)
      d_sens: int
    """
    if reuse_cache and os.path.exists(cache_pt):
        ckpt = torch.load(cache_pt, map_location="cpu")
        return ckpt["B_sens"].float(), int(ckpt["d_sens"])

    device = pipe.device
    C, H, W = 4, 64, 64
    D = C * H * W

    mini_steps = int(ssp_cfg.mini_steps)
    gscale = float(ssp_cfg.guidance_scale)

    grads: List[torch.Tensor] = []
    for p in prompts:
        z0 = torch.randn((1, C, H, W), device=device, dtype=torch.float32, requires_grad=True)

        # light-weight forward: predict eps at a fixed timestep using UNet
        # Ensure consistency with SSC mini sampling
        # We approximate by one-step UNet grad w.r.t. z0 at t=T/2
        t = torch.tensor([pipe.scheduler.config.num_train_timesteps // 2], device=device, dtype=torch.long)

        pipe.scheduler.set_timesteps(mini_steps, device=device)
        # encode text
        enc = _encode_prompt(pipe, p)

        # noise pred
        noise_pred = pipe.unet(z0, t, encoder_hidden_states=enc).sample
        loss = (noise_pred ** 2).mean()
        loss.backward()
        g = z0.grad.detach().reshape(D).float().cpu()
        grads.append(g)

        del z0, enc, noise_pred, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    G = torch.stack(grads, dim=1)  # [D, N]
    # SVD
    # keep top components that explain energy_ratio
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)  # U: [D,N]
    energy = (S**2)
    cum = torch.cumsum(energy, dim=0) / torch.sum(energy)
    d_sens = int(torch.searchsorted(cum, torch.tensor([ssp_cfg.energy_ratio])).item() + 1)
    B_sens = U[:, :d_sens].contiguous()  # [D, d]

    torch.save({"B_sens": B_sens, "d_sens": d_sens, "ssp_cfg": asdict(ssp_cfg)}, cache_pt)
    return B_sens, d_sens


# ---------------- SSM + ARR (one-shot tail repair) ----------------
@torch.no_grad()
def proj_to_span(B: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    B: [D,d] (CPU, float32), orthonormal columns
    x: [1,4,64,64] on CPU/float32
    return: projection onto span(B), same shape as x (CPU/float32)
    """
    D = B.shape[0]
    v = x.reshape(1, D).t().float()  # [D,1]
    coef = B.t().matmul(v)           # [d,1]
    proj = B.matmul(coef)            # [D,1]
    return proj.reshape_as(x).float()


@torch.no_grad()
def ARR_tail_repair_once(
    zT_pre: torch.Tensor,          # [1,4,64,64] CPU/float32
    z_wm: torch.Tensor,            # [1,4,64,64] CPU/float32
    support_idx: torch.Tensor,     # [m] CPU/long
    support_sign: torch.Tensor,    # [m] CPU/int8 (+1/-1)
    tau: float,
) -> torch.Tensor:
    """
    One-shot repair:
      - copy tail entries (|z|>=tau) from z_wm to zT_pre
      - additionally enforce sign on those entries
    """
    z = zT_pre.clone().view(-1)
    zw = z_wm.view(-1)

    idx = support_idx.to(torch.long)
    sgn = support_sign.to(torch.int8)

    # copy values from official watermarked latent (tail)
    z[idx] = zw[idx]

    # Enforce the sign constraint under numerical edge cases
    # if value is 0, push to tau with correct sign
    val = z[idx]
    val_sign = torch.sign(val)
    val_sign = torch.where(val_sign == 0, torch.ones_like(val_sign), val_sign)
    target_sign = sgn.to(val_sign.dtype)
    flip = (val_sign != target_sign)
    val[flip] = -val[flip]

    # ensure magnitude >= tau
    val = torch.where(val.abs() < tau, target_sign * torch.tensor(tau, dtype=val.dtype), val)
    z[idx] = val

    return z.view_as(zT_pre).float()


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--prompts", type=str, required=True, help="Prompt txt used for SSP calibration; the first cal_N prompts are used.")
    ap.add_argument("--cluster_pt", type=str, required=True, help="official T2S cluster pt")
    ap.add_argument("--outdir", type=str, required=True)

    # output naming (for lam1 sweep, etc.)
    ap.add_argument(
        "--export_latents_dir",
        type=str,
        default="",
        help="where to save output pt/json. default: <outdir>/latents_experiment",
    )
    ap.add_argument("--out_pt_name", type=str, default="generate_T2S_w_att.pt")
    ap.add_argument("--out_pre_pt_name", type=str, default="generate_T2S_w_att_pre.pt")
    ap.add_argument("--meta_pt_name", type=str, default="generate_T2S_w_att_meta.pt")
    ap.add_argument("--meta_json_name", type=str, default="generate_T2S_w_att_meta.json")

    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--seed", type=int, default=12345)

    # attack mixing
    ap.add_argument("--lam1", type=float, default=0.9, help="zT = lam1*z_wm + lam2*z_sens, lam2=sqrt(1-lam1^2)")

    # ssp
    ap.add_argument("--ssp_cal_N", type=int, default=12)
    ap.add_argument("--ssp_energy_ratio", type=float, default=0.9)
    ap.add_argument("--ssp_mini_steps", type=int, default=6)
    ap.add_argument("--ssp_guidance", type=float, default=7.5)
    ap.add_argument("--reuse_ssp", type=int, default=1)

    # t2s params (read from cluster settings or fallback to CLI)
    ap.add_argument("--t2s_tau", type=float, default=0.674)
    ap.add_argument("--key_channel_idx", type=int, default=0)

    # dtype/mem
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--xformers", action="store_true")
    ap.add_argument("--attn_slicing", action="store_true")

    args = ap.parse_args()

    # dtype
    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_dir(args.outdir)
    lat_exp_dir = os.path.join(args.outdir, "latents_experiment")
    ensure_dir(lat_exp_dir)

    export_dir = args.export_latents_dir.strip()
    if export_dir == "":
        export_dir = lat_exp_dir
    ensure_dir(export_dir)

    ssp_cache = os.path.join(lat_exp_dir, "ssp_basis.pt")

    # load cluster
    pack = torch.load(args.cluster_pt, map_location="cpu")
    z_wm_all: torch.Tensor = pack["latents"].float()  # [N,4,64,64]
    settings = pack.get("settings", {}) if isinstance(pack, dict) else {}

    K = int(args.K)
    if z_wm_all.shape[0] < K:
        raise ValueError(f"cluster has only {z_wm_all.shape[0]} latents, but K={K}")

    z_wm_all = z_wm_all[:K].contiguous()

    tau = float(settings.get("tau", args.t2s_tau))
    key_channel_idx = int(settings.get("key_channel_idx", args.key_channel_idx))

    # compute support/sign from the *watermarked* latent itself
    supports: List[torch.Tensor] = []
    signs: List[torch.Tensor] = []
    support_sizes: List[int] = []

    for i in range(K):
        z = z_wm_all[i].view(-1)
        idx = torch.nonzero(z.abs() >= tau, as_tuple=False).view(-1).to(torch.long)
        sgn = torch.sign(z.index_select(0, idx)).to(torch.int8)
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)

        supports.append(idx.cpu())
        signs.append(sgn.cpu())
        support_sizes.append(int(idx.numel()))

    # load pipeline (for ssp)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype if torch_dtype != torch.float32 else torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if args.xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    if args.attn_slicing:
        pipe.enable_attention_slicing()

    # ssp basis
    cal_prompts = read_prompts(args.prompts, int(args.ssp_cal_N))
    ssp_cfg = sspConfig(
        cal_N=int(args.ssp_cal_N),
        energy_ratio=float(args.ssp_energy_ratio),
        mini_steps=int(args.ssp_mini_steps),
        guidance_scale=float(args.ssp_guidance),
    )
    B_sens, d_sens = run_ssp_build_Bsens(
        pipe=pipe,
        prompts=cal_prompts,
        ssp_cfg=ssp_cfg,
        cache_pt=ssp_cache,
        reuse_cache=bool(int(args.reuse_ssp)),
    )  # CPU float32

    # build zT_pre and repaired zT_ref (one-shot)
    torch.manual_seed(int(args.seed))

    lam1 = float(args.lam1)
    lam2 = math.sqrt(max(0.0, 1.0 - lam1 * lam1))

    zT_pre_list: List[torch.Tensor] = []
    zT_ref_list: List[torch.Tensor] = []

    for i in range(K):
        z_wm = z_wm_all[i : i + 1].float()  # [1,4,64,64] CPU

        # z0 ~ N(0,I) -> z_sens = Proj_{B_sens}(z0)
        z0 = torch.randn_like(z_wm).float()
        z_sens = proj_to_span(B_sens, z0)  # CPU float32

        zT_pre = (lam1 * z_wm + lam2 * z_sens).float()

        # one-shot repair on tail
        zT_ref = ARR_tail_repair_once(
            zT_pre=zT_pre,
            z_wm=z_wm,
            support_idx=supports[i],
            support_sign=signs[i],
            tau=tau,
        )

        zT_pre_list.append(zT_pre.cpu())
        zT_ref_list.append(zT_ref.cpu())

        del z_wm, z0, z_sens, zT_pre, zT_ref
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    zT_pre_pack = torch.cat(zT_pre_list, dim=0).float()  # [K,4,64,64]
    zT_ref_pack = torch.cat(zT_ref_list, dim=0).float()  # [K,4,64,64]

    out_pt = os.path.join(export_dir, args.out_pt_name)
    out_pre_pt = os.path.join(export_dir, args.out_pre_pt_name)
    meta_pt = os.path.join(export_dir, args.meta_pt_name)
    meta_json = os.path.join(export_dir, args.meta_json_name)

    torch.save(zT_ref_pack, out_pt)
    torch.save(zT_pre_pack, out_pre_pt)

    # meta
    meta: Dict[str, Any] = {
        "method": "T2S",
        "variant": "att_once_repair",
        "K": K,
        "tau": tau,
        "key_channel_idx": key_channel_idx,
        "lam1": lam1,
        "lam2": lam2,
        "ssp_cfg": asdict(ssp_cfg),
        "d_sens": int(d_sens),
        "support_sizes": support_sizes,
        "cluster_pt": args.cluster_pt,
        "model_id_for_ssp": args.model_id,
        "prompts_for_ssp": args.prompts,
        "seed": int(args.seed),
        "out_pt": out_pt,
        "out_pre_pt": out_pre_pt,
    }

    torch.save(meta, meta_pt)
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved zT_ref: {out_pt}  shape={tuple(zT_ref_pack.shape)}")
    print(f"[OK] saved zT_pre: {out_pre_pt} shape={tuple(zT_pre_pack.shape)}")
    print(f"[OK] saved meta:   {meta_pt}")
    print(f"[OK] saved meta:   {meta_json}")


if __name__ == "__main__":
    main()
