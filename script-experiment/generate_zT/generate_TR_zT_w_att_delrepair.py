# -*- coding: utf-8 -*-
"""generate_TR_zT_w_att_delrepair.py

Ablation for TR w_att: **NO SPS watermark repair**
  1) SSC (unchanged): estimate B_sens from calibration prompts (mini-sampling + CLIP hinge grads).
  2) EBS (unchanged): build zT by FFT-mixing (lam1 * FFT(z_wm) + (1-lam1) * FFT(z_sens)).
  3) SPS (DISABLED): do NOT repair watermark on zT (directly save z_pre as final zT).
  4) Save a zT bank: a single pt file containing `zT_bank` with shape [M,4,64,64] (default M=16).

Design choices aligned to your requirements:
  - Seeds are incremental: seed_i = seed_base + i, i=0..M-1 (fully reproducible).
  - zT generation is independent of prompts (prompts are only used by SSC calibration).
  - Final zT is the "pre-repair" one (z_pre), because this script disables SPS repair.

Example:
CUDA_VISIBLE_DEVICES=0 python generate_TR_zT_w_att_delrepair.py \
  --model_id /home/yancy/work/dm_backdoor_latent_space/checkpoints/sd1-4-diffusers \
  --prompts  /home/yancy/work/dm_backdoor_latent_space/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt \
  --outdir   /home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment \
  --out_pt   /home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_TR_w_att_0_88_delrepair.pt \
  --height 512 --width 512 \
  --lam1 0.88 --seed 12345 \
  --ssc_N_cal 12 --ssc_mini_steps 6 \
  --tr_w_seed 12345 --tr_w_pattern ring --tr_w_mask_shape circle --tr_w_radius 9 --tr_w_channel -1 --tr_w_injection complex
"""

from __future__ import annotations

from math import sqrt
import os
import re
import gc
import hashlib
import argparse
import random
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Literal

import numpy as np
import torch
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


# -----------------------------
# Utilities
# -----------------------------

def seed_from_bytes(b: bytes) -> int:
    h = hashlib.sha256(b).digest()
    return int.from_bytes(h[:4], "big", signed=False)


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _read_prompts_txt(path: str) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            prompts.append(s)
    if not prompts:
        raise ValueError(f"No prompts found in: {path}")
    return prompts


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def flatten_latent(z: torch.Tensor) -> torch.Tensor:
    return z.reshape(z.shape[0], -1)


def unflatten_latent(z_flat: torch.Tensor, shape: Tuple[int, int, int, int]) -> torch.Tensor:
    return z_flat.reshape(shape)


def orthonormalize_cols_qr(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if A is None or A.numel() == 0:
        return A
    dev = A.device
    Af = A.float().detach().cpu()
    Q, R = torch.linalg.qr(Af, mode="reduced")
    diag = torch.abs(torch.diag(R))
    keep = diag > eps
    if keep.numel() == 0:
        return A[:, :0]
    Q = Q[:, keep].to(device=dev)
    return Q


def project_rows_onto_basis(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # X: (N,D), B:(D,k) with orthonormal columns
    if B is None or B.numel() == 0:
        return torch.zeros_like(X)
    B = B.to(device=X.device)
    return (X @ B) @ B.transpose(0, 1)


def explained_energy_to_rank(svals: torch.Tensor, ratio: float = 0.90) -> int:
    energy = svals.pow(2)
    cum = torch.cumsum(energy, dim=0)
    total = cum[-1].clamp_min(1e-12)
    k = int(torch.searchsorted(cum / total, torch.tensor(ratio, device=svals.device)).item()) + 1
    return max(1, min(k, svals.numel()))


def randomized_svd_topk(G: torch.Tensor, k: int):
    q = min(k + 8, min(G.shape) - 1)
    U, S, V = torch.pca_lowrank(G, q=q, center=False, niter=2)
    V_k = V[:, :k]
    S_k = S[:k]
    U_k = U[:, :k]
    return U_k, S_k, V_k.transpose(0, 1)


# -----------------------------
# Diffusion sampler (for SSC mini-sampling)
# -----------------------------


@dataclass
class DiffusionSamplerConfig:
    num_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0
    mini_steps: int = 6


class DiffusionLatentSampler:
    def __init__(self, pipe, device: torch.device, dtype: torch.dtype = torch.float32):
        self.pipe = pipe
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode_prompt(self, prompt: str, negative_prompt: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self.pipe.tokenizer
        text_inp = tok([prompt], padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt")
        neg_inp = tok([negative_prompt], padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt")
        text_inp = {k: v.to(self.device) for k, v in text_inp.items()}
        neg_inp = {k: v.to(self.device) for k, v in neg_inp.items()}

        text_emb = self.pipe.text_encoder(**text_inp)[0]
        neg_emb = self.pipe.text_encoder(**neg_inp)[0]
        return text_emb, neg_emb

    def _predict_eps(self, latents, t, cond, uncond, guidance_scale: float):
        inp = torch.cat([latents, latents], dim=0)
        t_in = torch.cat([t, t], dim=0)
        enc = torch.cat([uncond, cond], dim=0)
        eps = self.pipe.unet(inp, t_in, encoder_hidden_states=enc).sample
        eps_u, eps_c = eps.chunk(2, dim=0)
        return eps_u + guidance_scale * (eps_c - eps_u)

    def sample_latents(self, zT: torch.Tensor, prompt: str, cfg: DiffusionSamplerConfig, negative_prompt: str = "") -> torch.Tensor:
        cond, uncond = self.encode_prompt(prompt, negative_prompt=negative_prompt)
        B = zT.shape[0]
        if cond.shape[0] != B:
            cond = cond.expand(B, -1, -1).contiguous()
            uncond = uncond.expand(B, -1, -1).contiguous()

        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(cfg.num_steps, device=self.device)

        latents = zT
        if hasattr(scheduler, "init_noise_sigma"):
            latents = latents * scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            t_b = torch.full((B,), t, device=self.device, dtype=torch.long)
            eps = self._predict_eps(latents, t_b, cond, uncond, cfg.guidance_scale)
            try:
                latents = scheduler.step(eps, t, latents, eta=cfg.eta).prev_sample
            except TypeError:
                latents = scheduler.step(eps, t, latents).prev_sample

        return latents

    def mini_sample_image(self, zT: torch.Tensor, prompt: str, cfg: DiffusionSamplerConfig, negative_prompt: str = "") -> torch.Tensor:
        mini_cfg = DiffusionSamplerConfig(
            num_steps=cfg.mini_steps,
            guidance_scale=cfg.guidance_scale,
            eta=cfg.eta,
            mini_steps=cfg.mini_steps,
        )
        x_lat = self.sample_latents(zT, prompt, mini_cfg, negative_prompt=negative_prompt)
        return self.decode_latents(x_lat)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        scaling = getattr(self.pipe.vae.config, "scaling_factor", 0.18215)
        latents_in = latents / scaling
        img = self.pipe.vae.decode(latents_in).sample
        img = (img + 1.0) / 2.0
        return img.clamp(0, 1)


# -----------------------------
# Surrogate: CLIP hinge loss (SSC only)
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
        B = images_01.shape[0]
        img = F.interpolate(images_01, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=img.device).view(1, 3, 1, 1)
        img = (img - mean) / std

        with torch.no_grad():
            tok = self.proc(text=[prompt] * B, images=None, return_tensors="pt", padding=True).to(self.device)
            text_emb = self.clip.get_text_features(**{k: tok[k] for k in ["input_ids", "attention_mask"]})
            text_emb = l2_normalize(text_emb, dim=-1)

        img_emb = self.clip.get_image_features(pixel_values=img)
        img_emb = l2_normalize(img_emb, dim=-1)

        sim = (img_emb * text_emb).sum(dim=-1)
        loss = F.relu(margin - sim).mean()
        return loss


# -----------------------------
# Tree-Ring (frequency mask + patch)
# -----------------------------


Family = Literal["freq"]


@dataclass
class WatermarkConfig:
    family: Family = "freq"
    device: str = "cuda"
    tr_seed_from_key: bool = False
    tr_w_seed: int = 12345
    tr_w_channel: int = -1
    tr_w_pattern: str = "ring"
    tr_w_mask_shape: str = "circle"
    tr_w_radius: int = 9
    tr_w_injection: str = "complex"
    tr_w_pattern_const: float = 0.0


class FrequencyMaskFamily:
    def __init__(self, D: int, wm_cfg: WatermarkConfig, latent_shape: Tuple[int, int, int]):
        self.D = int(D)
        self.cfg = wm_cfg
        self.device = torch.device(wm_cfg.device)
        self.C, self.H, self.W = int(latent_shape[0]), int(latent_shape[1]), int(latent_shape[2])
        assert self.C * self.H * self.W == self.D

        self._cached_sig = None
        self._cached_mask = None
        self._cached_patch = None

    def _derive_seed(self, K: bytes) -> int:
        base = int(self.cfg.tr_w_seed)
        if not bool(self.cfg.tr_seed_from_key):
            return base
        s = int.from_bytes(hashlib.sha256(b"treering|" + K).digest()[:4], "big", signed=False)
        return base ^ int(s)

    def _make_mask(self, B: int, device: torch.device) -> torch.Tensor:
        shape = str(self.cfg.tr_w_mask_shape)
        radius = int(self.cfg.tr_w_radius)
        ch = int(self.cfg.tr_w_channel)

        yy = torch.arange(self.H, device=device).view(self.H, 1).repeat(1, self.W)
        xx = torch.arange(self.W, device=device).view(1, self.W).repeat(self.H, 1)
        cy = (self.H - 1) / 2.0
        cx = (self.W - 1) / 2.0
        dy = yy.float() - cy
        dx = xx.float() - cx
        rr = torch.sqrt(dx * dx + dy * dy)

        if shape == "circle":
            m2 = rr <= float(radius)
        elif shape == "square":
            m2 = (torch.abs(dx) <= float(radius)) & (torch.abs(dy) <= float(radius))
        elif shape == "no":
            m2 = torch.ones((self.H, self.W), device=device, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown tr_w_mask_shape={shape}")

        mask = m2.view(1, 1, self.H, self.W).repeat(B, self.C, 1, 1)
        if ch >= 0:
            keep = torch.zeros((B, self.C, 1, 1), device=device, dtype=torch.bool)
            keep[:, ch : ch + 1] = True
            mask = mask & keep
        return mask

    def _make_patch(self, B: int, device: torch.device, seed: int) -> torch.Tensor:
        pat = str(self.cfg.tr_w_pattern)
        constv = float(getattr(self.cfg, "tr_w_pattern_const", 0.0))
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

        if pat == "zeros":
            z4 = torch.zeros((B, self.C, self.H, self.W), device=device, dtype=torch.float32)
        elif pat == "rand":
            z4 = torch.randn((B, self.C, self.H, self.W), device=device, generator=gen, dtype=torch.float32)
        elif pat == "ring":
            z4 = torch.randn((B, self.C, self.H, self.W), device=device, generator=gen, dtype=torch.float32)
            z4 = torch.fft.fftshift(torch.fft.fft2(z4), dim=(-1, -2))
        elif pat == "seed_rand":
            z4 = torch.randn((B, self.C, self.H, self.W), device=device, generator=gen, dtype=torch.float32)
        elif pat == "seed_zeros":
            z4 = torch.zeros((B, self.C, self.H, self.W), device=device, dtype=torch.float32)
        elif pat == "seed_ring":
            z4 = torch.randn((B, self.C, self.H, self.W), device=device, generator=gen, dtype=torch.float32)
            z4 = torch.fft.fftshift(torch.fft.fft2(z4), dim=(-1, -2))
        elif pat == "const":
            z4 = torch.full((B, self.C, self.H, self.W), constv, device=device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown tr_w_pattern={pat}")
        return z4

    def _get_cached_mask_patch(self, K: bytes, B: int, device: torch.device):
        seed = self._derive_seed(K)
        sig = (
            int(seed),
            int(B),
            int(self.C),
            int(self.H),
            int(self.W),
            str(self.cfg.tr_w_pattern),
            str(self.cfg.tr_w_mask_shape),
            int(self.cfg.tr_w_radius),
            int(self.cfg.tr_w_channel),
        )
        if self._cached_sig != sig or self._cached_mask is None or self._cached_patch is None:
            mask = self._make_mask(B=B, device=device)
            patch = self._make_patch(B=B, device=device, seed=int(seed))
            self._cached_sig = sig
            self._cached_mask = mask
            self._cached_patch = patch
        return self._cached_mask, self._cached_patch

    def inject(self, z4: torch.Tensor, K: bytes) -> torch.Tensor:
        B = z4.shape[0]
        mask, patch = self._get_cached_mask_patch(K=K, B=B, device=z4.device)
        inj = str(getattr(self.cfg, "tr_w_injection", "complex"))

        if inj == "complex":
            Z = torch.fft.fftshift(torch.fft.fft2(z4), dim=(-1, -2))
            patch_fft = patch
            if not torch.is_complex(patch_fft):
                patch_fft = torch.fft.fftshift(torch.fft.fft2(patch_fft), dim=(-1, -2))
            patch_fft = patch_fft.to(Z.dtype)
            Z[mask] = patch_fft[mask].clone()
            out = torch.fft.ifft2(torch.fft.ifftshift(Z, dim=(-1, -2))).real
            return out

        if inj == "seed":
            z_out = z4.clone()
            z_out[mask] = patch[mask].clone()
            return z_out

        raise NotImplementedError(f"tr_w_injection={inj}")


# -----------------------------
# Workflow: SSC + EBS (+ SPS code kept but NOT USED in this ablation)
# -----------------------------


@dataclass
class SSCConfig:
    N_cal: int = 12
    d_sens_max: int = 64
    energy_ratio: float = 0.90
    mini_steps: int = 6
    guidance_scale: float = 7.5
    eta_ddim: float = 0.0


@dataclass
class SPSConfig:
    # kept for CLI compatibility; NOT USED here
    T_r: int = 1


class AlignPreserveNaW_New:
    def __init__(
        self,
        sampler: DiffusionLatentSampler,
        surrogate: Callable[[torch.Tensor, str], torch.Tensor],
        latent_shape: Tuple[int, int, int, int],
        device: str = "cuda",
    ):
        self.sampler = sampler
        self.surrogate = surrogate
        self.device = torch.device(device)
        self.latent_shape = latent_shape  # (B,C,H,W)
        self.D = latent_shape[1] * latent_shape[2] * latent_shape[3]
        self.B_sens: Optional[torch.Tensor] = None

    def run_ssc(self, prompts: List[str], ssc_cfg: SSCConfig, negative_prompt: str = "") -> dict:
        assert len(prompts) >= ssc_cfg.N_cal
        grads: List[torch.Tensor] = []
        C, H, W = self.latent_shape[1], self.latent_shape[2], self.latent_shape[3]

        for i in range(ssc_cfg.N_cal):
            prompt = prompts[i]
            zT = torch.randn((1, C, H, W), device=self.device, dtype=torch.float32, requires_grad=True)
            cfg = DiffusionSamplerConfig(
                num_steps=ssc_cfg.mini_steps,
                guidance_scale=ssc_cfg.guidance_scale,
                eta=ssc_cfg.eta_ddim,
                mini_steps=ssc_cfg.mini_steps,
            )
            x_tilde = self.sampler.mini_sample_image(zT, prompt, cfg, negative_prompt=negative_prompt)
            loss = self.surrogate(x_tilde, prompt)
            g = torch.autograd.grad(loss, zT, retain_graph=False, create_graph=False)[0]
            grads.append(flatten_latent(g).detach().cpu())
            del zT, x_tilde, loss, g
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        G = torch.cat(grads, dim=0)
        k_try = min(ssc_cfg.d_sens_max, min(G.shape) - 1)
        _, S, Vt = randomized_svd_topk(G, k=k_try)
        d_sens = explained_energy_to_rank(S, ratio=float(ssc_cfg.energy_ratio))
        del G
        gc.collect()

        B_sens = Vt[:d_sens].transpose(0, 1).contiguous()  # (D,d_sens)
        B_sens = orthonormalize_cols_qr(B_sens)
        del S, Vt

        self.B_sens = B_sens
        return {"B_sens": B_sens, "d_sens": torch.tensor([d_sens], device=self.device)}

    def ebs_sample(self, wm_family: FrequencyMaskFamily, lam1: float, seed_prompt: int, K: bytes) -> torch.Tensor:
        """EBS unchanged: FFT-mix between z_wm and projected z_sens."""
        assert self.B_sens is not None, "Call run_ssc first."
        B, C, H, W = self.latent_shape
        lam1 = float(lam1)
        lam2 = sqrt(1.0 - lam1 * lam1)

        g_sens = torch.Generator(device=self.device)
        g_sens.manual_seed(int(seed_prompt))
        g_wm = torch.Generator(device=self.device)
        g_wm.manual_seed(int(seed_prompt) + 1)

        # z_sens
        z = torch.randn((B, C, H, W), device=self.device, generator=g_sens, dtype=torch.float32)
        z_flat = flatten_latent(z)
        z_sens_flat = project_rows_onto_basis(z_flat, self.B_sens)
        z_sens = unflatten_latent(z_sens_flat, (B, C, H, W))
        del z, z_flat, z_sens_flat

        # z_wm
        z0 = torch.randn((B, C, H, W), device=self.device, generator=g_wm, dtype=torch.float32)
        z_wm = wm_family.inject(z0, K=K)
        del z0

        Z_sens = torch.fft.fftshift(torch.fft.fft2(z_sens), dim=(-1, -2))
        Z_wm = torch.fft.fftshift(torch.fft.fft2(z_wm), dim=(-1, -2))
        Z_mix = (lam1 * Z_wm) + (lam2 * Z_sens)
        zT = torch.fft.ifft2(torch.fft.ifftshift(Z_mix, dim=(-1, -2))).real
        del z_sens, z_wm, Z_sens, Z_wm, Z_mix

        return zT.detach()


# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser(description="TR SSC+EBS (unchanged) + NO SPS repair => save zT bank (delrepair)")

    # IO
    parser.add_argument("--prompts", type=str, required=True, help="Used for SSC calibration only")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--save_zt", type=int, default=1, help="If 1, save bank pt (always).")
    parser.add_argument("--bank_num", type=int, default=16, help="Number of zT to generate (default 16)")
    parser.add_argument("--out_pt", type=str, default="", help="Output zT bank pt path. Default: <outdir>/generate_TR_w_att_delrepair.pt")

    # SD
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--negative_prompt", type=str, default="")

    # SSC
    parser.add_argument("--ssc_N_cal", type=int, default=12)
    parser.add_argument("--ssc_mini_steps", type=int, default=6)
    parser.add_argument("--ssc_energy_ratio", type=float, default=0.90)
    parser.add_argument("--ssc_d_sens_max", type=int, default=64)

    # EBS mix
    parser.add_argument("--lam1", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=12345, help="seed_base, actual seed_i = seed + i")

    # SPS (kept for CLI compatibility; unused in delrepair)
    parser.add_argument("--sps_T_r", type=int, default=1)

    # Tree-Ring params
    parser.add_argument("--tr_w_seed", type=int, default=12345)
    parser.add_argument("--tr_w_channel", type=int, default=-1)
    parser.add_argument("--tr_w_pattern", type=str, default="ring")
    parser.add_argument("--tr_w_mask_shape", type=str, default="circle")
    parser.add_argument("--tr_w_radius", type=int, default=9)
    parser.add_argument("--tr_w_injection", type=str, default="complex")
    parser.add_argument("--tr_seed_from_key", type=int, default=0)

    args = parser.parse_args()

    device = torch.device(args.device)
    _ensure_dir(args.outdir)
    meta_dir = os.path.join(args.outdir, "wm_meta")
    _ensure_dir(meta_dir)
    latents_dir = args.outdir
    _ensure_dir(latents_dir)

    prompts_all = _read_prompts_txt(args.prompts)
    cal = (prompts_all * ((int(args.ssc_N_cal) + len(prompts_all) - 1) // len(prompts_all)))[: int(args.ssc_N_cal)]

    H_lat = args.height // 8
    W_lat = args.width // 8
    latent_shape = (1, 4, H_lat, W_lat)
    D = 4 * H_lat * W_lat

    print("[Init] Loading Stable Diffusion pipeline (for SSC only)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    sampler = DiffusionLatentSampler(pipe, device=device, dtype=torch.float32)
    surrogate_impl = CLIPHingeSurrogate(device=str(device))

    workflow = AlignPreserveNaW_New(
        sampler=sampler,
        surrogate=lambda img, prompt: surrogate_impl(img, prompt, margin=0.2),
        latent_shape=latent_shape,
        device=str(device),
    )

    # --- SSC (unchanged) ---
    print(f"[SSC] Calibrating with N_cal={int(args.ssc_N_cal)} ...")
    ssc_cfg = SSCConfig(
        N_cal=int(args.ssc_N_cal),
        d_sens_max=int(args.ssc_d_sens_max),
        energy_ratio=float(args.ssc_energy_ratio),
        mini_steps=int(args.ssc_mini_steps),
        guidance_scale=7.5,
        eta_ddim=0.0,
    )
    stats = workflow.run_ssc(cal, ssc_cfg, negative_prompt=args.negative_prompt)
    d_sens = int(stats["d_sens"].item())
    print(f"[SSC] d_sens={d_sens}")
    torch.save(
        {
            "B_sens": workflow.B_sens.detach().cpu(),
            "d_sens": d_sens,
            "ssc_cfg": ssc_cfg.__dict__,
            "note": "SSC basis (sensitive subspace) used to build z_sens = Proj_{B_sens}(z).",
        },
        os.path.join(meta_dir, "ssc_basis.pt"),
    )

    # --- Watermark cfg + family ---
    wm_cfg = WatermarkConfig(
        family="freq",
        device=str(device),
        tr_seed_from_key=bool(int(args.tr_seed_from_key)),
        tr_w_seed=int(args.tr_w_seed),
        tr_w_channel=int(args.tr_w_channel),
        tr_w_pattern=str(args.tr_w_pattern),
        tr_w_mask_shape=str(args.tr_w_mask_shape),
        tr_w_radius=int(args.tr_w_radius),
        tr_w_injection=str(args.tr_w_injection),
    )
    wm_family = FrequencyMaskFamily(D=D, wm_cfg=wm_cfg, latent_shape=(4, H_lat, W_lat))
    K = b""

    # (SPS cfg is unused, but keep for bookkeeping)
    _ = SPSConfig(T_r=int(args.sps_T_r))

    # --- Generate zT bank (independent of prompt) ---
    M = int(args.bank_num)
    seed_base = int(args.seed)
    print(f"[BANK] Generating M={M} zT with incremental seeds: seed_i = {seed_base} + i")
    print("[BANK] NOTE: delrepair mode => save z_pre directly (NO watermark repair).")

    z_list: List[torch.Tensor] = []
    seeds: List[int] = []
    for i in range(M):
        seed_i = seed_base + i
        seeds.append(seed_i)

        z_pre = workflow.ebs_sample(wm_family=wm_family, lam1=float(args.lam1), seed_prompt=seed_i, K=K)  # (1,4,H,W)

        # ---- DELREPAIR: do NOT call workflow.sps_refine ----
        z_list.append(z_pre[0].detach().cpu().to(torch.float32))

        if (i + 1) % 4 == 0 or (i + 1) == M:
            print(f"  [{i+1:02d}/{M}] done (seed={seed_i})")

        del z_pre
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    z_bank = torch.stack(z_list, dim=0)  # [M,4,H,W]
    assert z_bank.ndim == 4 and z_bank.shape[0] == M and z_bank.shape[1] == 4, f"bad z_bank shape: {tuple(z_bank.shape)}"

    out_pt = str(args.out_pt) if str(args.out_pt).strip() else os.path.join(latents_dir, "generate_TR_w_att_delrepair.pt")
    torch.save(
        {
            "zT_bank": z_bank,
            "shape": list(z_bank.shape),
            "seeds": seeds,
            "seed_base": seed_base,
            "lam1": float(args.lam1),
            "lam2": sqrt(1.0 - float(args.lam1)*float(args.lam1)),
            "wm_cfg": wm_cfg.__dict__,
            "ssc_basis_path": os.path.join(meta_dir, "ssc_basis.pt"),
            "note": "TR zT bank (delrepair): SSC+EBS unchanged; NO SPS watermark repair. Seeds are incremental.",
        },
        out_pt,
    )

    print("\n[DONE]")
    print(f"Saved zT bank: {out_pt}")
    print(f"zT_bank shape: {tuple(z_bank.shape)}")


if __name__ == "__main__":
    main()
