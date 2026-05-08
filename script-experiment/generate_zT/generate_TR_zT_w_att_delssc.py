# -*- coding: utf-8 -*-
"""mitigate_treering_sensFFTmix_repairOnly_saveZTbank.py

What this script does (TR only):
  1) SSC (delSSC): skip B_sens calibration; use Gaussian z_sens directly.
  2) EBS (delSSC): build zT by FFT-mixing (lam1 * FFT(z_wm) + lam2 * FFT(z_sens)), where z_sens ~ N(0,I).
  3) SPS (CHANGED): **remove gradient update**, keep **one-time Tree-Ring repair** on zT.
  4) Save a zT bank: a single pt file containing `zT_bank` with shape [M,4,64,64] (default M=16).

Design choices for this configuration:
  - Seeds are incremental: seed_i = seed_base + i, i=0..M-1 (reproducible).
  - zT bank generation is independent of prompts; prompts are kept for argparse compatibility.
  - Output pt stores the `zT_bank` key for compatibility with the gen_from_zT_bank loader.
"""

import os
import gc
import math
from math import sqrt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import argparse
import numpy as np
import torch
from PIL import Image

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


# ---------------------------
# Utils
# ---------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_prompts_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def flatten_latent(z: torch.Tensor) -> torch.Tensor:
    # z: (B,C,H,W) -> (B, C*H*W)
    return z.reshape(z.shape[0], -1)


def unflatten_latent(z_flat: torch.Tensor, shape_bchw: Tuple[int, int, int, int]) -> torch.Tensor:
    B, C, H, W = shape_bchw
    return z_flat.reshape(B, C, H, W)


def project_rows_onto_basis(x: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # x: (B, D), B: (d, D) row-orthonormal
    # proj(x) = x @ B^T @ B
    return (x @ B.T) @ B


# ---------------------------
# Diffusion sampler (used for SSC; kept for alignment)
# ---------------------------

@dataclass
class DiffusionSamplerConfig:
    num_steps: int = 50
    guidance_scale: float = 7.5
    eta_ddim: float = 0.0


class DiffusionLatentSampler:
    def __init__(self, pipe: StableDiffusionPipeline, device: torch.device, dtype: torch.dtype = torch.float32):
        self.pipe = pipe
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def mini_sample_image(self, z0: torch.Tensor, prompt: str, cfg: DiffusionSamplerConfig, negative_prompt: str = "") -> Image.Image:
        # This path is only used when SSC is enabled; delSSC bypasses SSC.
        g = torch.Generator(device=self.device)
        g.manual_seed(123)
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(cfg.num_steps),
            guidance_scale=float(cfg.guidance_scale),
            eta=float(cfg.eta_ddim),
            latents=z0.to(device=self.device, dtype=self.dtype),
            generator=g,
        )
        return out.images[0]


class CLIPHingeSurrogate(torch.nn.Module):
    def __init__(self, clip_model_id: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(device)
        self.clip = CLIPModel.from_pretrained(clip_model_id).to(self.device)
        self.proc = CLIPProcessor.from_pretrained(clip_model_id)

    def forward(self, img: Image.Image, prompt: str, margin: float = 0.2) -> torch.Tensor:
        inputs = self.proc(text=[prompt], images=[img], return_tensors="pt", padding=True).to(self.device)
        out = self.clip(**inputs)
        logits = out.logits_per_image  # (1,1)
        # hinge: max(0, margin - score)
        loss = torch.relu(float(margin) - logits).mean()
        return loss


# ---------------------------
# Watermark config + family
# ---------------------------

@dataclass
class WatermarkConfig:
    family: str = "freq"
    device: str = "cuda"

    tr_seed_from_key: bool = False
    tr_w_seed: int = 12345
    tr_w_channel: int = -1
    tr_w_pattern: str = "ring"
    tr_w_mask_shape: str = "circle"
    tr_w_radius: int = 9
    tr_w_injection: str = "complex"


class FrequencyMaskFamily:
    def __init__(self, D: int, wm_cfg: WatermarkConfig, latent_shape: Tuple[int, int, int]):
        self.D = int(D)
        self.cfg = wm_cfg
        self.latent_shape = latent_shape  # (C,H,W)
        self.device = torch.device(wm_cfg.device)

        C, H, W = latent_shape
        self.C = C
        self.H = H
        self.W = W

        # Precompute mask in FFT domain
        self.mask = self._build_mask(
            H=H, W=W,
            mask_shape=str(wm_cfg.tr_w_mask_shape),
            radius=int(wm_cfg.tr_w_radius),
            channel=int(wm_cfg.tr_w_channel),
        )

    def _build_mask(self, H: int, W: int, mask_shape: str, radius: int, channel: int) -> torch.Tensor:
        yy = torch.arange(H, device=self.device).view(H, 1).repeat(1, W)
        xx = torch.arange(W, device=self.device).view(1, W).repeat(H, 1)
        cy = (H - 1) / 2.0
        cx = (W - 1) / 2.0
        dy = yy.float() - cy
        dx = xx.float() - cx

        if mask_shape.lower() == "circle":
            rr = torch.sqrt(dx * dx + dy * dy)
            m2 = rr <= float(radius)
        elif mask_shape.lower() == "square":
            ax = dx.abs()
            ay = dy.abs()
            m2 = (ax <= float(radius)) & (ay <= float(radius))
        else:
            raise ValueError(f"unknown mask_shape={mask_shape}")

        mask = m2.view(1, 1, H, W).repeat(1, self.C, 1, 1)  # (1,C,H,W)
        if int(channel) >= 0:
            keep = torch.zeros((1, self.C, 1, 1), device=self.device, dtype=torch.bool)
            keep[:, int(channel):int(channel) + 1] = True
            mask = mask & keep
        return mask

    def _effective_seed(self, K: bytes) -> int:
        if not bool(self.cfg.tr_seed_from_key):
            return int(self.cfg.tr_w_seed)
        # xor with sha256(K) prefix
        import hashlib
        h = hashlib.sha256(b"treering|" + (K or b"")).digest()
        k = int.from_bytes(h[:4], "little", signed=False)
        return int(self.cfg.tr_w_seed) ^ k

    def _build_patch_fft(self, seed: int, B: int) -> torch.Tensor:
        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed))

        z = torch.randn((B, self.C, self.H, self.W), generator=g, device=self.device, dtype=torch.float32)
        patch_fft = torch.fft.fftshift(torch.fft.fft2(z), dim=(-1, -2)).to(torch.complex64)
        return patch_fft

    def inject(self, z: torch.Tensor, K: bytes) -> torch.Tensor:
        """
        Tree-Ring "complex" injection: replace FFT coefficients at mask with patch_fft(mask).
        z: (B,C,H,W) real
        """
        assert z.ndim == 4 and z.shape[1] == self.C and z.shape[2] == self.H and z.shape[3] == self.W

        seed_eff = self._effective_seed(K)
        patch_fft = self._build_patch_fft(seed_eff, B=z.shape[0])

        Z = torch.fft.fftshift(torch.fft.fft2(z), dim=(-1, -2)).to(torch.complex64)

        if str(self.cfg.tr_w_injection).lower() == "complex":
            Z[self.mask] = patch_fft[self.mask]
        else:
            raise ValueError(f"Unsupported tr_w_injection={self.cfg.tr_w_injection}")

        z_out = torch.fft.ifft2(torch.fft.ifftshift(Z, dim=(-1, -2))).real
        return z_out


# ---------------------------
# SSC / SPS configs
# ---------------------------

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
    T_r: int = 1


# ---------------------------
# Main workflow class
# ---------------------------

class AlignPreserveNaW_New:
    def __init__(
        self,
        sampler: DiffusionLatentSampler,
        surrogate,
        latent_shape: Tuple[int, int, int, int],
        device: str = "cuda",
    ):
        self.sampler = sampler
        self.surrogate = surrogate
        self.latent_shape = latent_shape  # (B,C,H,W)
        self.device = torch.device(device)

        self.B_sens: Optional[torch.Tensor] = None

    def run_ssc(self, prompts: List[str], ssc_cfg: SSCConfig, negative_prompt: str = "") -> Dict[str, Any]:
        """
        Original SSC code (kept). delSSC ablation does NOT call this.
        """
        B, C, H, W = self.latent_shape
        D = C * H * W

        mini_cfg = DiffusionSamplerConfig(
            num_steps=int(ssc_cfg.mini_steps),
            guidance_scale=float(ssc_cfg.guidance_scale),
            eta_ddim=float(ssc_cfg.eta_ddim),
        )

        grads: List[torch.Tensor] = []
        for i, p in enumerate(prompts):
            z0 = torch.randn((B, C, H, W), device=self.device, dtype=torch.float32, requires_grad=True)
            img = self.sampler.mini_sample_image(z0, prompt=p, cfg=mini_cfg, negative_prompt=negative_prompt)
            loss = self.surrogate(img, p)
            g = torch.autograd.grad(loss, z0, retain_graph=False, create_graph=False)[0]
            grads.append(g.detach().reshape(-1).float().cpu())
            del z0, img, loss, g
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        G = torch.stack(grads, dim=0)  # (N_cal, D)
        cov = (G.T @ G) / float(G.shape[0])

        # eigen-decomp
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.flip(0)
        eigvecs = eigvecs.flip(1)
        energy = torch.cumsum(eigvals, dim=0) / (eigvals.sum() + 1e-12)

        d_sens = int(torch.searchsorted(energy, float(ssc_cfg.energy_ratio)).item()) + 1
        d_sens = int(min(max(d_sens, 1), int(ssc_cfg.d_sens_max)))

        B_sens = eigvecs[:, :d_sens].T.contiguous()  # (d_sens, D)
        self.B_sens = B_sens.to(self.device)

        return {"d_sens": torch.tensor(d_sens)}

    def ebs_sample(self, wm_family: FrequencyMaskFamily, lam1: float, seed_prompt: int, K: bytes) -> torch.Tensor:
        """EBS (delSSC): FFT-mix between z_wm and **Gaussian** z_sens (no SSC projection)."""
        B, C, H, W = self.latent_shape
        lam1 = float(lam1)
        lam2 = sqrt(1.0 - lam1*lam1)

        g_sens = torch.Generator(device=self.device)
        g_sens.manual_seed(int(seed_prompt))
        g_wm = torch.Generator(device=self.device)
        g_wm.manual_seed(int(seed_prompt) + 1)

        # z_sens (delSSC): direct Gaussian, no projection
        z_sens = torch.randn((B, C, H, W), device=self.device, generator=g_sens, dtype=torch.float32)

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

    def repair_wm_on_zt(self, wm_family: FrequencyMaskFamily, zT: torch.Tensor, K: bytes) -> torch.Tensor:
        return wm_family.inject(zT, K=K)

    def sps_refine(self, wm_family: FrequencyMaskFamily, zT: torch.Tensor, sps_cfg: SPSConfig, K: bytes) -> torch.Tensor:
        """SPS changed: no gradient update, only one-time Tree-Ring repair."""
        _ = sps_cfg  # kept for alignment/logging
        z_ref = self.repair_wm_on_zt(wm_family=wm_family, zT=zT, K=K)
        return z_ref.detach()


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="TR SSC+EBS (unchanged) + SPS(repair-only) => save zT bank")

    # IO
    parser.add_argument("--prompts", type=str, required=True, help="Used for SSC calibration only")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--save_zt", type=int, default=1, help="If 1, save bank pt (always).")
    parser.add_argument("--bank_num", type=int, default=16, help="Number of zT to generate (default 16)")
    parser.add_argument("--out_pt", type=str, default="", help="Output zT bank pt path. Default: <outdir>/generate_TR_w_att.pt")

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

    # SPS parameter retained for logging; SPS is repair-only
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
    latent_shape = (1, 4, H_lat, W_lat)  # IMPORTANT: generate 1 zT each time, then stack to [M,4,H,W]
    D = 4 * H_lat * W_lat

    print("[Init] Loading Stable Diffusion pipeline (kept for alignment; delSSC skips SSC computation)...")
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

    # --- SSC (delSSC) ---
    # SSC is disabled in this ablation; z_sens is sampled directly from N(0, I) inside ebs_sample.

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
    K = b""  # fixed; with tr_seed_from_key=0 it won't change the patch anyway

    # --- SPS cfg (repair-only) ---
    sps_cfg = SPSConfig(T_r=int(args.sps_T_r))

    # --- Generate zT bank (independent of prompt) ---
    M = int(args.bank_num)
    seed_base = int(args.seed)
    print(f"[BANK] Generating M={M} zT with incremental seeds: seed_i = {seed_base} + i")

    z_list: List[torch.Tensor] = []
    seeds: List[int] = []
    for i in range(M):
        seed_i = seed_base + i
        seeds.append(seed_i)
        z_pre = workflow.ebs_sample(wm_family=wm_family, lam1=float(args.lam1), seed_prompt=seed_i, K=K)  # (1,4,H,W)
        z_ref = workflow.sps_refine(wm_family=wm_family, zT=z_pre, sps_cfg=sps_cfg, K=K)  # (1,4,H,W)
        z_list.append(z_ref[0].detach().cpu().to(torch.float32))

        if (i + 1) % 4 == 0 or (i + 1) == M:
            print(f"  [{i+1:02d}/{M}] done (seed={seed_i})")

        del z_pre, z_ref
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    z_bank = torch.stack(z_list, dim=0)  # [M,4,H,W]
    assert z_bank.ndim == 4 and z_bank.shape[0] == M and z_bank.shape[1] == 4, f"bad z_bank shape: {tuple(z_bank.shape)}"

    # Save bank
    out_pt = str(args.out_pt) if str(args.out_pt).strip() else os.path.join(latents_dir, "generate_TR_w_att.pt")
    torch.save(
        {
            "zT_bank": z_bank,
            "shape": list(z_bank.shape),
            "seeds": seeds,
            "seed_base": seed_base,
            "lam1": float(args.lam1),
            "lam2": 1.0 - float(args.lam1),
            "wm_cfg": wm_cfg.__dict__,
            "ssc_basis_path": None,  # delSSC
            "note": "TR zT bank (delSSC): EBS uses Gaussian z_sens (no SSC) + SPS(repair-only, 1x). Seeds are incremental.",
        },
        out_pt,
    )

    print("\n[DONE]")
    print(f"Saved zT bank: {out_pt}")
    print(f"zT_bank shape: {tuple(z_bank.shape)}")


if __name__ == "__main__":
    main()
