#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_T2S_zT_w_att_delrepair.py

T2S w_att ablation: delrepair (skip the third-step tail repair)
- SSC(B_sens) is aligned with the TR pipeline: z0 -> mini_sample_image -> CLIP hinge surrogate -> grad(z0)
- Save zT_pre directly as the final zT without watermark repair

Outputs: {outdir}/latents_experiment/
  - generate_T2S_zT_w_att_delrepair.pt              # [K,4,64,64] final (= zT_pre)
  - generate_T2S_zT_w_att_delrepair_meta.pt/.json   # hyperparameters and support metadata
  - generate_T2S_zT_w_att_delrepair_Bsens.pt/.json  # explicit B_sens export
  - ssc_basis.pt                                    # SSC cache (dict with B_sens)
"""

import argparse
import gc
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


# -------------------------
# utils
# -------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompts(txt: str) -> List[str]:
    out: List[str] = []
    with open(txt, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                out.append(t)
    return out


def choose_rank_by_energy(S: torch.Tensor, energy_ratio: float) -> int:
    # S: singular values
    e = (S.float() ** 2)
    cum = torch.cumsum(e, dim=0)
    tot = cum[-1].clamp_min(1e-12)
    frac = cum / tot
    k = int(torch.searchsorted(frac, torch.tensor(float(energy_ratio), device=frac.device)).item()) + 1
    return max(1, min(k, S.numel()))


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))


# -----------------------------
# Surrogate: CLIP hinge loss (SSC only) - aligned with TR
# -----------------------------

class CLIPHingeSurrogate(torch.nn.Module):
    def __init__(self, clip_model_id: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(device)
        self.clip = CLIPModel.from_pretrained(clip_model_id).to(self.device)
        self.proc = CLIPProcessor.from_pretrained(clip_model_id)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)

    def forward(self, images_01: torch.Tensor, prompt: str, margin: float = 0.25) -> torch.Tensor:
        # images_01: [B,3,H,W] in [0,1]
        B = images_01.shape[0]
        img = F.interpolate(images_01, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=img.device).view(1, 3, 1, 1)
        img = (img - mean) / std

        with torch.no_grad():
            tok = self.proc(text=[prompt] * B, images=None, return_tensors="pt", padding=True).to(self.device)
            text_emb = self.clip.get_text_features(**{k: tok[k] for k in ["input_ids", "attention_mask"]})
            text_emb = l2_normalize(text_emb, dim=-1)

        img_emb = self.clip.get_image_features(pixel_values=img.to(self.device))
        img_emb = l2_normalize(img_emb, dim=-1)

        sim = (img_emb * text_emb).sum(dim=-1)
        loss = F.relu(float(margin) - sim).mean()
        return loss


# -------------------------
# Diffusion mini-sampler for SSC gradients (aligned with TR)
# -------------------------

@dataclass
class GenConfig:
    num_steps: int = 6
    guidance_scale: float = 7.5
    eta: float = 0.0
    mini_steps: int = 6


class DiffusionSampler:
    def __init__(self, pipe: StableDiffusionPipeline, device: torch.device, dtype: torch.dtype):
        self.pipe = pipe
        self.device = device
        self.dtype = dtype

        self.unet = pipe.unet
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder

        self.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    @torch.no_grad()
    def encode_prompts(self, prompts: List[str], negative_prompts: List[str]) -> torch.Tensor:
        max_len = self.tokenizer.model_max_length
        t = self.tokenizer(prompts, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")
        tneg = self.tokenizer(negative_prompts, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")
        pos_ids = t.input_ids.to(self.device)
        neg_ids = tneg.input_ids.to(self.device)
        pos = self.text_encoder(pos_ids)[0]
        neg = self.text_encoder(neg_ids)[0]
        return pos.to(dtype=self.dtype), neg.to(dtype=self.dtype)

    def sample_latents(self, prompt: str, latents: torch.Tensor, gen_cfg: GenConfig, negative_prompt: str = "") -> torch.Tensor:
        """Differentiable wrt `latents`."""
        B = latents.shape[0]
        pos, neg = self.encode_prompts([prompt] * B, [negative_prompt] * B)

        do_cfg = float(gen_cfg.guidance_scale) > 1e-6
        self.scheduler.set_timesteps(int(gen_cfg.num_steps), device=self.device)

        x = latents
        timesteps = self.scheduler.timesteps

        use_autocast = (self.device.type == "cuda") and (self.dtype in (torch.float16, torch.bfloat16))

        for t in timesteps:
            if do_cfg:
                x_in = torch.cat([x, x], dim=0)
                enc = torch.cat([neg, pos], dim=0)
            else:
                x_in = x
                enc = pos

            x_in = self.scheduler.scale_model_input(x_in, t)

            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    noise_pred = self.unet(x_in, t, encoder_hidden_states=enc).sample
            else:
                noise_pred = self.unet(x_in, t, encoder_hidden_states=enc).sample

            if do_cfg:
                n_uncond, n_text = noise_pred.chunk(2)
                noise_pred = n_uncond + float(gen_cfg.guidance_scale) * (n_text - n_uncond)

            x = self.scheduler.step(noise_pred, t, x).prev_sample

        return x

    # ---- aligned with TR ----
    def mini_sample_image(self, zT: torch.Tensor, prompt: str, cfg: GenConfig, negative_prompt: str = "") -> torch.Tensor:
        mini_cfg = GenConfig(
            num_steps=int(cfg.mini_steps),
            guidance_scale=float(cfg.guidance_scale),
            eta=float(cfg.eta),
            mini_steps=int(cfg.mini_steps),
        )
        x_lat = self.sample_latents(prompt=prompt, latents=zT, gen_cfg=mini_cfg, negative_prompt=negative_prompt)
        return self.decode_latents(x_lat)

    # ---- aligned with TR ----
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        scaling = getattr(self.pipe.vae.config, "scaling_factor", 0.18215)
        latents_in = latents / scaling
        img = self.pipe.vae.decode(latents_in).sample
        img = (img + 1.0) / 2.0
        return img.clamp(0, 1)


# -------------------------
# SSC basis + projection (aligned)
# -------------------------

@dataclass
class SSCConfig:
    cal_N: int = 12
    energy_ratio: float = 0.90
    mini_steps: int = 6
    guidance_scale: float = 7.5


def run_ssc_build_Bsens(
    sampler: DiffusionSampler,
    C: int,
    H: int,
    W: int,
    cal_prompts: List[str],
    ssc_cfg: SSCConfig,
    surrogate: Any,
    surrogate_margin: float = 0.2,
) -> Dict[str, Any]:
    device = sampler.device
    D = C * H * W
    mini_cfg = GenConfig(num_steps=int(ssc_cfg.mini_steps), guidance_scale=float(ssc_cfg.guidance_scale), mini_steps=int(ssc_cfg.mini_steps))

    grads: List[torch.Tensor] = []
    for p in cal_prompts:
        z0 = torch.randn((1, C, H, W), device=device, dtype=torch.float32, requires_grad=True)
        img01 = sampler.mini_sample_image(z0, prompt=p, cfg=mini_cfg, negative_prompt="")
        loss = surrogate(img01, p, margin=float(surrogate_margin))
        g = torch.autograd.grad(loss, z0, retain_graph=False, create_graph=False)[0]
        grads.append(g.detach().reshape(-1).float().cpu())

        del z0, img01, loss, g
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    G = torch.stack(grads, dim=1)          # [D, N]
    U, S, _ = torch.linalg.svd(G, full_matrices=False)
    d_sens = choose_rank_by_energy(S, float(ssc_cfg.energy_ratio))
    B_sens = U[:, :d_sens].contiguous()    # [D, d_sens]

    return {"D": int(D), "d_sens": int(d_sens), "B_sens": B_sens, "ssc_cfg": asdict(ssc_cfg)}


@torch.no_grad()
def project_to_subspace(z: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    z_vec = z.reshape(1, -1)
    coeff = z_vec @ B
    proj = coeff @ B.t()
    return proj.reshape_as(z)


# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--prompts", type=str, required=True)
    ap.add_argument("--cluster_pt", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--lam1", type=float, default=0.9)

    ap.add_argument("--ssc_cal_N", type=int, default=12)
    ap.add_argument("--ssc_energy_ratio", type=float, default=0.90)
    ap.add_argument("--ssc_mini_steps", type=int, default=6)
    ap.add_argument("--ssc_guidance", type=float, default=7.5)
    ap.add_argument("--reuse_ssc", type=int, default=1)

    ap.add_argument("--t2s_tau", type=float, default=0.674)
    ap.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--clip_margin", type=float, default=0.2)

    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--xformers", action="store_true")
    ap.add_argument("--attn_slicing", action="store_true")

    args = ap.parse_args()

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

    # ---------- load cluster ----------
    pack = torch.load(args.cluster_pt, map_location="cpu")
    if isinstance(pack, torch.Tensor):
        z_wm_all = pack.float()
        cluster_settings = {"format": "tensor"}
    elif isinstance(pack, dict):
        if "latents" in pack and isinstance(pack["latents"], torch.Tensor):
            z_wm_all = pack["latents"].float()
            cluster_settings = {"format": "dict", "key": "latents"}
        else:
            # Fallback: first tensor-like value
            key = None
            for k, v in pack.items():
                if isinstance(v, torch.Tensor) and v.ndim == 4 and v.shape[1:] == (4, 64, 64):
                    key = k
                    z_wm_all = v.float()
                    break
            if key is None:
                raise ValueError(f"cluster_pt dict has no [N,4,64,64] tensor. keys={list(pack.keys())}")
            cluster_settings = {"format": "dict", "key": key}
    else:
        raise ValueError(f"unsupported cluster_pt type: {type(pack)}")

    K = int(args.K)
    if z_wm_all.shape[0] < K:
        raise ValueError(f"cluster has only {z_wm_all.shape[0]} samples, but K={K}")
    z_wm_all = z_wm_all[:K].contiguous()

    tau = float(args.t2s_tau)

    # build supports/signs from z_wm tails (even if delrepair, useful for sanity prints + meta)
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

    # ---------- pipeline ----------
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=(torch_dtype if torch_dtype != torch.float32 else torch.float16),
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if args.xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[MEM] xformers enabled")
        except Exception as e:
            print(f"[MEM] xformers enable failed: {e}")

    if args.attn_slicing:
        try:
            pipe.enable_attention_slicing("max")
            print("[MEM] attention slicing enabled")
        except Exception as e:
            print(f"[MEM] attention slicing enable failed: {e}")

    sampler = DiffusionSampler(pipe=pipe, device=device, dtype=torch_dtype)

    # ---------- surrogate ----------
    surrogate_impl = CLIPHingeSurrogate(clip_model_id=str(args.clip_model), device=str(device))
    surrogate = lambda img, prompt, margin=0.2: surrogate_impl(img, prompt, margin=float(margin))

    # ---------- SSC ----------
    ssc_cache = os.path.join(lat_exp_dir, "ssc_basis.pt")
    prompts = load_prompts(args.prompts)
    cal_prompts = prompts[: int(args.ssc_cal_N)]
    if len(cal_prompts) < int(args.ssc_cal_N):
        raise ValueError(f"prompts too short: need {args.ssc_cal_N}, got {len(cal_prompts)}")

    if int(args.reuse_ssc) == 1 and os.path.exists(ssc_cache):
        ssc_obj = torch.load(ssc_cache, map_location="cpu")
        print(f"[SSC] Reuse {ssc_cache} (d_sens={ssc_obj['d_sens']})")
    else:
        ssc_cfg = SSCConfig(
            cal_N=int(args.ssc_cal_N),
            energy_ratio=float(args.ssc_energy_ratio),
            mini_steps=int(args.ssc_mini_steps),
            guidance_scale=float(args.ssc_guidance),
        )
        print("[SSC] building B_sens...")
        ssc_obj = run_ssc_build_Bsens(
            sampler=sampler,
            C=4, H=64, W=64,
            cal_prompts=cal_prompts,
            ssc_cfg=ssc_cfg,
            surrogate=surrogate,
            surrogate_margin=float(args.clip_margin),
        )
        torch.save(ssc_obj, ssc_cache)
        print(f"[SSC] saved {ssc_cache} (d_sens={ssc_obj['d_sens']})")

    B_sens = ssc_obj["B_sens"].to(device=device, dtype=torch.float32)

    # ---------- mix ----------
    lam1 = float(args.lam1)
    if not (0.0 <= lam1 <= 1.0):
        raise ValueError(f"--lam1 must be in [0,1], got {lam1}")
    lam2 = math.sqrt(max(0.0, 1.0 - lam1 * lam1))

    set_seed(int(args.seed))

    zT_pre_list: List[torch.Tensor] = []
    zT_ref_list: List[torch.Tensor] = []

    for i in range(K):
        z_wm = z_wm_all[i:i+1].to(device=device, dtype=torch.float32)

        g = torch.Generator(device=device)
        g.manual_seed(int(args.seed) + i)
        z0 = torch.randn((1, 4, 64, 64), generator=g, device=device, dtype=torch.float32)
        z_free = project_to_subspace(z0, B_sens)

        zT_pre = lam1 * z_wm + lam2 * z_free

        # --- delrepair: skip repair ---
        zT_ref = zT_pre

        # sanity: tail sign/amp vs supports from z_wm
        with torch.no_grad():
            idx = supports[i].to(device=device, dtype=torch.long)
            sgn = signs[i].to(device=device, dtype=torch.float32)
            vals = zT_ref.view(-1).index_select(0, idx)
            sign_ok = float((torch.sign(vals) == sgn).float().mean().item())
            amp_ok = float((vals.abs() >= tau).float().mean().item())
        print(f"[SANITY][{i:02d}] support={support_sizes[i]}  sign_ok={sign_ok:.2f}  amp_ok={amp_ok:.2f}")

        zT_pre_list.append(zT_pre.detach().cpu())
        zT_ref_list.append(zT_ref.detach().cpu())

        del z_wm, z0, z_free, zT_pre, zT_ref
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    zT_pre_pack = torch.cat(zT_pre_list, dim=0).float()   # [K,4,64,64]
    zT_ref_pack = torch.cat(zT_ref_list, dim=0).float()   # [K,4,64,64] (=pre)

    # ---------- save ----------
    out_pt = os.path.join(lat_exp_dir, "generate_T2S_w_att_delrepair.pt")
    out_pre_pt = os.path.join(lat_exp_dir, "generate_T2S_w_att_delrepair_pre.pt")
    meta_pt = os.path.join(lat_exp_dir, "generate_T2S_w_att_delrepair_meta.pt")
    meta_json = os.path.join(lat_exp_dir, "generate_T2S_w_att_delrepair_meta.json")

    torch.save(zT_ref_pack, out_pt)
    torch.save(zT_pre_pack, out_pre_pt)

    meta: Dict[str, Any] = {
        "method": "T2S",
        "variant": "w_att_delrepair",
        "K": K,
        "tau": tau,
        "lam1": lam1,
        "lam2": lam2,
        "support_sizes": support_sizes,
        "supports": supports,
        "signs": signs,
        "ssc": {"D": int(ssc_obj["D"]), "d_sens": int(ssc_obj["d_sens"]), "ssc_cfg": ssc_obj.get("ssc_cfg", {})},
        "cluster_pt": os.path.abspath(args.cluster_pt),
        "cluster_settings": cluster_settings,
        "model_id": os.path.abspath(args.model_id),
        "prompts": os.path.abspath(args.prompts),
        "clip_model": str(args.clip_model),
        "clip_margin": float(args.clip_margin),
        "delrepair": True,
    }
    torch.save(meta, meta_pt)
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in meta.items() if k not in ["supports", "signs"]}, f, indent=2, ensure_ascii=False)

    # explicit save B_sens
    bsens_pt = os.path.join(lat_exp_dir, "generate_T2S_zT_w_att_delrepair_Bsens.pt")
    bsens_json = os.path.join(lat_exp_dir, "generate_T2S_zT_w_att_delrepair_Bsens.json")
    torch.save(ssc_obj["B_sens"].contiguous(), bsens_pt)
    with open(bsens_json, "w", encoding="utf-8") as f:
        json.dump(
            {"D": int(ssc_obj["D"]), "d_sens": int(ssc_obj["d_sens"]), "ssc_cfg": ssc_obj.get("ssc_cfg", {})},
            f, indent=2, ensure_ascii=False
        )

    print("\n[OK] saved:")
    print(" ", out_pt)
    print(" ", out_pre_pt)
    print(" ", meta_pt)
    print(" ", meta_json)
    print(" ", bsens_pt)
    print(" ", bsens_json)


if __name__ == "__main__":
    main()
