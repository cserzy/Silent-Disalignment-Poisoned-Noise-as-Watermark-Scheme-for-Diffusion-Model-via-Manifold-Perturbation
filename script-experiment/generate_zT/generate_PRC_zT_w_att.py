import os
import re
import csv
import math
import json
import time
import random
import hashlib
import argparse
import contextlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F

import pickle
import prc as prc_lib
import pseudogaussians as pg

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

# -------------------------
# Helpers: linear algebra
# -------------------------

def orthonormalize_cols_qr(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Orthonormalize columns of A using QR.
    Returns Q with same number of columns as A (or fewer if rank-deficient).
    """
    # A: (D, k)
    if A is None:
        return A
    if A.numel() == 0:
        return A
    # float32 for stability
    Af = A.float()
    # QR on CPU for stability, then move back if needed
    device = Af.device
    if device.type != "cpu":
        Af_cpu = Af.detach().cpu()
    else:
        Af_cpu = Af.detach()

    Q, R = torch.linalg.qr(Af_cpu, mode="reduced")
    # drop near-zero columns
    diag = torch.abs(torch.diag(R))
    keep = diag > eps
    if keep.numel() == 0:
        return A[:, :0]
    Q = Q[:, keep]
    Q = Q.to(device=device)
    return Q


def remove_cols_component(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Remove projection of X onto span(B), without forming projector.
    X: (D, k), B: (D, r) orthonormal.
    Returns: X - B(B^T X)
    """
    if B is None or B.numel() == 0:
        return X
    # ensure orthonormal
    BtX = B.transpose(0, 1) @ X
    return X - (B @ BtX)


def flatten_latent(z: torch.Tensor) -> torch.Tensor:
    # z: (B,C,H,W) -> (B, D)
    return z.flatten(1)


def unflatten_latent(z_flat: torch.Tensor, shape_bchw: Tuple[int,int,int,int]) -> torch.Tensor:
    B,C,H,W = shape_bchw
    return z_flat.view(B, C, H, W)


def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_orthonormal_basis(B: torch.Tensor, target_d: int, seed: int = 0) -> torch.Tensor:
    """Pad an orthonormal basis B (D,d) to (D,target_d) by adding random orthonormal directions.

    Keeps existing columns unchanged; new columns are sampled and orthonormalized against span(B).
    """
    if B is None:
        raise ValueError('B is None')
    D, d = B.shape
    target_d = int(min(int(target_d), int(D)))
    if d >= target_d:
        return B[:, :target_d].contiguous()

    # Use a deterministic RNG stream (so caching is stable across runs)
    g = torch.Generator(device=B.device)
    g.manual_seed(int(seed) & 0xFFFFFFFF)

    extra = target_d - d
    R = torch.randn(D, extra, device=B.device, dtype=B.dtype, generator=g)

    # Project out existing span(B)
    # R <- R - B(B^T R)
    R = R - B @ (B.transpose(0, 1) @ R)

    # Orthonormalize the new directions (reduced QR)
    Q, _ = torch.linalg.qr(R, mode='reduced')
    Q = Q[:, :extra].contiguous()

    B2 = torch.cat([B, Q], dim=1)
    # Safety: re-orthonormalize once to reduce numerical drift
    B2 = orthonormalize_cols_qr(B2).contiguous()
    return B2


# -------------------------
# Configs
# -------------------------

@dataclass
class DiffusionSamplerConfig:
    num_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0
    mini_steps: int = 50


@dataclass
class sspConfig:
    N_cal: int = 12
    guidance_scale: float=7.5,
    d_sens_max: int = 64
    d_wm: int = 256
    energy_ratio: float = 0.9
    mini_steps: int = 12


@dataclass
class ARRConfig:
    T_r: int = 2
    eta: float = 0.05
    normalize_grad: bool = True
    mini_steps: int = 2


@dataclass
class WatermarkConfig:
    family: str = "proj"
    proj_mode: str = "prc"
    # PRC params
    prc_message_length: int = 32
    prc_error_prob: float = 0.01
    # route-B / misc
    master_key: str = "change_me"
    # (other watermark families might use other fields)


# -------------------------
# PRC family (GLOBAL, official math)
# -------------------------

class GlobalPRCWatermark:
    """Official PRC watermark living in the *full* latent noise space R^D.

    We follow the official math:
      - KeyGen(n=D, message_length, false_positive_rate, t, noise_rate)
      - Encode(encoding_key, message_bits) -> codeword in {+1,-1}^D
      - Sample pseudo-Gaussian: codeword * |N(0,1)|  (pg.sample)
    """

    def __init__(self, D: int, wm_cfg: WatermarkConfig):
        self.D = int(D)
        self.cfg = wm_cfg

        mk = getattr(wm_cfg, 'master_key', 'change_me')
        mk_bytes = mk.encode('utf-8')

        # deterministic np seed
        self._np_seed = int.from_bytes(hashlib.sha256(mk_bytes + b'::prc_global').digest()[:4], 'little')

        # PRC hyperparams
        self.false_positive_rate = float(getattr(wm_cfg, 'prc_fpr', 1e-9))
        self.t = int(getattr(wm_cfg, 'prc_t', 3))
        self.noise_rate = float(getattr(wm_cfg, 'prc_error_prob', 0.01))
        self.message_length = int(getattr(wm_cfg, 'prc_message_length', 32))

        # KeyGen uses numpy RNG
        import numpy as _np
        _np.random.seed(self._np_seed)
        self.encoding_key, self.decoding_key = prc_lib.KeyGen(
            n=self.D,
            message_length=self.message_length,
            false_positive_rate=self.false_positive_rate,
            t=self.t,
            noise_rate=self.noise_rate,
        )

        # deterministic message bits from master_key
        digest = hashlib.sha256(mk_bytes + b'::prc_msg').digest()
        bits = _np.unpackbits(_np.frombuffer(digest, dtype=_np.uint8)).astype(_np.int32)
        if bits.size < self.message_length:
            reps = int(_np.ceil(self.message_length / bits.size))
            bits = _np.tile(bits, reps)
        self.message_bits = bits[: self.message_length].tolist()

        # deterministic codeword in {+1,-1}^D (torch.float64)
        _np.random.seed(self._np_seed + 1)
        self.codeword = prc_lib.Encode(self.encoding_key, message=self.message_bits)  # torch, (D,)
        # safer sign (avoid 0)
        cw = self.codeword.clone()
        cw[cw == 0] = 1.0
        self.codeword_sign = torch.sign(cw).to(dtype=torch.float32)  # (D,)

    def sample_z_wm(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return watermarked noise z_wm in flat shape (B,D)."""
        # pg.sample uses numpy RNG (non-deterministic across calls unless you seed outside).
        # This is OK: PRC watermark relies on fixed signs (codeword), amplitudes can vary per sample.
        z = pg.sample(self.codeword).to(dtype=torch.float32)  # (D,)
        z = z.to(device=device, dtype=dtype)
        return z.view(1, -1).repeat(int(batch_size), 1)

    def repair_sign_preserve_amp(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Repair: keep |z| but force sign to match codeword."""
        sign = self.codeword_sign.to(device=z_flat.device, dtype=z_flat.dtype).view(1, -1)
        return sign * torch.abs(z_flat)

    def dump_runtime_artifacts(self, outdir: str) -> None:
        os.makedirs(outdir, exist_ok=True)
        # message bits
        with open(os.path.join(outdir, 'prc_message_bits.txt'), 'w', encoding='utf-8') as f:
            f.write(''.join(str(int(b)) for b in self.message_bits) + '\n')

        # combined keys (for convenience)
        keys_obj = {
            'encoding_key': self.encoding_key,
            'decoding_key': self.decoding_key,
            'meta': {
                'mode': 'prc_global',
                'D': int(self.D),
                'prc_message_length_bits': int(self.message_length),
                'noise_rate': float(self.noise_rate),
                'false_positive_rate': float(self.false_positive_rate),
                't': int(self.t),
                'np_seed': int(self._np_seed),
            },
        }
        with open(os.path.join(outdir, 'prc_keys.pkl'), 'wb') as f:
            pickle.dump(keys_obj, f)

        # also save separate keys (optional convenience)
        with open(os.path.join(outdir, 'encoding_key.pkl'), 'wb') as f:
            pickle.dump(self.encoding_key, f)
        with open(os.path.join(outdir, 'decoding_key.pkl'), 'wb') as f:
            pickle.dump(self.decoding_key, f)

        with open(os.path.join(outdir, 'prc_meta.json'), 'w', encoding='utf-8') as f:
            json.dump(keys_obj['meta'], f, indent=2)
# -------------------------
# Surrogate (CLIP hinge)
# -------------------------

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


class CLIPHingeSurrogate(torch.nn.Module):
    """Differentiable hinge surrogate based on CLIP image-text similarity."""

    def __init__(self, clip_model_id: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.clip = CLIPModel.from_pretrained(clip_model_id).to(self.device)
        self.proc = CLIPProcessor.from_pretrained(clip_model_id)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)

        # CLIP recommended normalization constants
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def forward(self, images_01: torch.Tensor, prompt: str, margin: float = 0.25) -> torch.Tensor:
        B = images_01.shape[0]
        img = F.interpolate(images_01, size=(224, 224), mode="bilinear", align_corners=False)
        img = (img - self.mean.to(img.device)) / self.std.to(img.device)

        with torch.no_grad():
            tok = self.proc(text=[prompt] * B, images=None, return_tensors="pt", padding=True).to(self.device)
            text_emb = self.clip.get_text_features(**{k: tok[k] for k in ["input_ids", "attention_mask"]})
            text_emb = l2_normalize(text_emb, dim=-1)

        img_emb = self.clip.get_image_features(pixel_values=img.to(self.device))
        img_emb = l2_normalize(img_emb, dim=-1)

        sim = (img_emb * text_emb).sum(dim=-1)
        return F.relu(margin - sim).mean()


# -------------------------
# Diffusion sampler wrapper
# -------------------------

class DiffusionSampler:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.device = torch.device(device)
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            safety_checker=None,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

        # components
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.scheduler = pipe.scheduler

        # -------------------------
        # DType management (important for ssp/ARR gradients)
        # -------------------------
        # Many diffusers components (e.g., timestep embedding / LoRA layers) are sensitive to mixed Float/Half graphs
        # when we do autograd on latents. To make ssp/ARR robust, we switch UNet+VAE to fp32 whenever
        # latents.requires_grad=True, and switch back to fp16 for normal generation (speed/low-mem).
        self._dtype_fp16 = torch.float16 if self.device.type == "cuda" else torch.float32
        self._dtype_fp32 = torch.float32
        self._dtype_mode = "fp16"
    @torch.no_grad()
    def encode_prompt(self, prompt: str, negative_prompt: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tok_n = self.tokenizer(
            [negative_prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tok.input_ids.to(self.pipe.device)
        input_ids_n = tok_n.input_ids.to(self.pipe.device)
        emb = self.text_encoder(input_ids)[0]
        emb_n = self.text_encoder(input_ids_n)[0]
        return emb, emb_n

    def _set_model_dtype(self, mode: str) -> None:
        """Switch UNet+VAE dtype.
        mode: 'fp16' or 'fp32'
        """
        if mode == self._dtype_mode:
            return
        if mode not in ("fp16", "fp32"):
            raise ValueError(f"Unknown dtype mode: {mode}")
        target = self._dtype_fp32 if mode == "fp32" else self._dtype_fp16
        # Keep text_encoder as-is (we cast embeddings in _denoise_step).
        self.unet.to(dtype=target)
        self.vae.to(dtype=target)
        self._dtype_mode = mode
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to image tensor in [0,1].

        - Keeps gradients when latents.requires_grad=True (ssp/ARR).
        - Fixes dtype mismatch between latents and VAE conv weights.
        - Mimics diffusers' 'upcast_vae + cast latents to post_quant_conv dtype' behavior.
        """
        grad_on = bool(getattr(latents, "requires_grad", False))

        # Some diffusers VAE configs set force_upcast=True for numerical stability
        force_upcast = bool(getattr(self.vae.config, "force_upcast", False))
        needs_upcasting = (self.vae.dtype == torch.float16) and force_upcast

        with torch.set_grad_enabled(grad_on):
            if needs_upcasting:
                self.vae.to(dtype=torch.float32)

            target_dtype = next(iter(self.vae.post_quant_conv.parameters())).dtype
            latents = latents.to(device=self.vae.device, dtype=target_dtype)

            latents = latents / self.vae.config.scaling_factor
            img = self.vae.decode(latents).sample
            img = (img / 2 + 0.5).clamp(0, 1)

            if needs_upcasting:
                self.vae.to(dtype=torch.float16)

        return img
    def _denoise_step(self, latents: torch.Tensor, t: torch.Tensor, emb: torch.Tensor, emb_n: torch.Tensor, cfg: float) -> torch.Tensor:
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        model_dtype = next(self.unet.parameters()).dtype
        grad_on = bool(getattr(latents, "requires_grad", False))        # autocast is useful in fp16/bf16 mode; in fp32 mode we keep everything in fp32 to avoid dtype-mismatch.
        use_amp = (self.pipe.device.type == "cuda") and (model_dtype in (torch.float16, torch.bfloat16))
        if use_amp:
            amp_ctx = torch.autocast(device_type="cuda", dtype=model_dtype)
        else:
            amp_ctx = contextlib.nullcontext()

        with torch.set_grad_enabled(grad_on), amp_ctx:
            latent_model_input = latent_model_input.to(dtype=model_dtype)
            emb = emb.to(dtype=model_dtype)
            emb_n = emb_n.to(dtype=model_dtype)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat([emb_n, emb])
            ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)
        noise_pred = noise_pred.to(dtype=latents.dtype)

        latents = self.scheduler.step(noise_pred, t, latents).prev_sample.to(dtype=latents.dtype)
        return latents


    def sample(self, prompt: str, negative_prompt: str, latents: torch.Tensor, cfg: DiffusionSamplerConfig) -> torch.Tensor:
        # latents: (B,C,H,W) starting noise (may require grad in ssp/ARR)
        grad_on = bool(getattr(latents, "requires_grad", False))
        # Switch model dtype depending on whether we need gradients on latents
        self._set_model_dtype("fp32" if grad_on else "fp16")
        mini = int(getattr(cfg, "mini_steps", 0) or 0)
        full = int(cfg.num_steps)
        steps = mini if (grad_on and mini > 0 and mini < full) else full

        self.scheduler.set_timesteps(steps, device=self.pipe.device)

        model_dtype = next(self.unet.parameters()).dtype
        # keep cast in graph (important when latents is float32 leaf but model is fp16)
        latents = latents.to(self.pipe.device, dtype=model_dtype)

        # DPM scheduler uses init_noise_sigma
        init_sigma = torch.as_tensor(self.scheduler.init_noise_sigma, device=latents.device, dtype=latents.dtype)
        latents = latents * init_sigma

        emb, emb_n = self.encode_prompt(prompt, negative_prompt=negative_prompt)
        for t in self.scheduler.timesteps:
            latents = self._denoise_step(latents, t, emb, emb_n, cfg=cfg.guidance_scale)

        img = self.decode_latents(latents)
        return img


# -------------------------
# Workflow: ssp + SSM + ARR
# -------------------------

class AlignPreserveNaW:
    def __init__(self, sampler: DiffusionSampler, surrogate, latent_shape: Tuple[int,int,int,int], device: str = "cuda"):
        self.sampler = sampler
        self.surrogate = surrogate
        self.latent_shape = latent_shape
        self.device = torch.device(device)
        # latent dim
        B,C,H,W = latent_shape
        self.D = int(C*H*W)
        # ssp basis
        self.B_sens: Optional[torch.Tensor] = None
        # PRC (global)
        self.prc: Optional[GlobalPRCWatermark] = None
        # mixing lambda: zT = lam1 * z_wm + (1-lam1) * z_sens
        self.lam1 = 0.6
        self.prc_posterior_boost = True
        self.prc_boost_var = 1.5
        self.prc_boost_tau = 0.25
        self.prc_boost_beta = 0.10

    def set_mix_lambda(self, lam1: float) -> None:
        lam1 = float(lam1)
        if not (0.0 <= lam1 <= 1.0):
            raise ValueError('lam1 must be in [0,1]')
        self.lam1 = lam1

    def set_prc(self, prc: GlobalPRCWatermark) -> None:
        self.prc = prc

    def run_ssp(self, cal_prompts: List[str], ssp_cfg: sspConfig) -> Dict[str, torch.Tensor]:
        """Calibrate sensitive subspace B_sens (no B_wm in this variant)."""
        B, C, H, W = self.latent_shape

        grads: List[torch.Tensor] = []
        for i, p in enumerate(cal_prompts[: int(ssp_cfg.N_cal)]):
            # Per-prompt random init (aligned with TR/T2S ssp style)
            g = torch.Generator(device=self.device.type)
            g.manual_seed(12345 + int(i))
            z0 = torch.randn((1, C, H, W), generator=g, device=self.device, dtype=torch.float32, requires_grad=True)

            cfg = DiffusionSamplerConfig(
                num_steps=int(ssp_cfg.mini_steps),
                guidance_scale=float(ssp_cfg.guidance_scale),
                eta=0.0,
                mini_steps=int(ssp_cfg.mini_steps),
            )
            img = self.sampler.sample(p, negative_prompt="", latents=z0, cfg=cfg)
            loss = self.surrogate(img, p)
            g_z = torch.autograd.grad(loss, z0, retain_graph=False, create_graph=False)[0]
            grads.append(flatten_latent(g_z.detach()).cpu())

            del z0, img, loss, g_z
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        G = torch.cat(grads, dim=0)  # (N,D)

        U, S, Vt = torch.linalg.svd(G, full_matrices=False)
        energy = (S ** 2).reshape(-1)
        cum = torch.cumsum(energy, dim=0) / (energy.sum() + 1e-12)

        ratio = float(ssp_cfg.energy_ratio)
        thr = torch.tensor(ratio, device=cum.device, dtype=cum.dtype)
        k = int(torch.searchsorted(cum, thr).item() + 1)
        k = max(1, min(k, int(energy.numel())))

        d_sens = min(int(ssp_cfg.d_sens_max), k)
        d_sens = max(1, int(d_sens))

        B_sens = Vt[:d_sens].transpose(0, 1).contiguous()  # (D,d_sens)
        B_sens = orthonormalize_cols_qr(B_sens).to(self.device)
        self.B_sens = B_sens

        return {
            'B_sens': B_sens,
            'd_sens': torch.tensor([int(B_sens.shape[1])], device=self.device),
        }

    def _proj(self, X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if B is None or B.numel() == 0:
            return torch.zeros_like(X)
        return (X @ B) @ B.transpose(0, 1)

    def SSM_sample(self, batch_size: int, seed: int, latent_chw: Tuple[int,int,int]) -> torch.Tensor:
        """SSM: build zT by mixing global PRC watermark z_wm and sensitive projection z_sens.

        New rule:
          z ~ N(0,I)
          z_sens = Proj_{span(B_sens)}(z)
          z_wm  = PRC-global pseudoGaussian sample (full space)
          zT = lam1 * z_wm + sqrt(1-lam1*lam1) * z_sens
        """
        if self.B_sens is None:
            raise ValueError('B_sens not initialized. Run ssp first.')
        if self.prc is None:
            raise ValueError('PRC not initialized. Call set_prc(...) before SSM.')

        C, H, W_ = latent_chw
        g = torch.Generator(device='cpu')
        g.manual_seed(int(seed))

        z = torch.randn((batch_size, C, H, W_), generator=g, device='cpu', dtype=torch.float32)
        z_flat = flatten_latent(z).to(self.device)

        # z_sens: projection onto sensitive subspace
        z_sens = self._proj(z_flat, self.B_sens)

        # z_wm: global PRC
        z_wm = self.prc.sample_z_wm(batch_size=batch_size, device=self.device, dtype=z_flat.dtype)

        lam1 = float(self.lam1)
        lam2 = math.sqrt(1.0 - lam1*lam1)
        zT_flat = lam1 * z_wm + lam2 * z_sens

        zT = unflatten_latent(zT_flat, (batch_size, C, H, W_)).detach().requires_grad_(True)
        z_wm_T=unflatten_latent(z_wm, (batch_size, C, H, W_)).detach().requires_grad_(True)
        return zT,z_wm_T

    def repair_prc_global(self, zT: torch.Tensor) -> torch.Tensor:
        """PRC repair on the full space."""
        if self.prc is None:
            raise ValueError('PRC not initialized.')
        B, C, H, W_ = zT.shape
        z_flat = flatten_latent(zT)                      # (B, D)
        z_new = self.prc.repair_sign_preserve_amp(z_flat)  # (B, D)

        # -------------------------
        # ADD HERE: posterior-boost (Decode-friendly)
        # -------------------------
        if not hasattr(self, "_dbg_boost_once"):
            self._dbg_boost_once = True
            print("[DBG] prc_posterior_boost =", getattr(self, "prc_posterior_boost", False))
        if bool(getattr(self, "prc_posterior_boost", False)):
            # hyperparams (给默认值，不用你额外改 CLI 也能跑)
            var  = float(getattr(self, "prc_boost_var", 1.5))   # 和你检测脚本 var 对齐的常用默认
            tau  = float(getattr(self, "prc_boost_tau", 0.25))  # “低置信”阈值
            beta = float(getattr(self, "prc_boost_beta", 0.10)) # 放大强度（建议 0.05~0.15）

            # 记录每个样本 boost 前的 std，避免整体能量漂移
            std0 = z_new.std(dim=1, keepdim=True, unbiased=False).detach()

            # 每个样本单独算 posteriors（pg.recover_posteriors 典型是 1D 输入）
            z_new_list = []
            for b in range(B):
                zb = z_new[b].detach().to(dtype=torch.float64).cpu()  # (D,)
                post = pg.recover_posteriors(zb, variances=var)        # numpy/torch 都可能
                post = torch.as_tensor(post, dtype=z_new.dtype, device=z_new.device)  # (D,)

                # 对 |post| 小的维度轻微放大 |z|（不改符号）
                # w in [1, 1+beta]
                denom = tau if tau > 1e-6 else 1e-6
                w = 1.0 + beta * torch.clamp((tau - post.abs()) / denom, min=0.0, max=1.0)  # (D,)

                zb_new = torch.sign(z_new[b]) * (z_new[b].abs() * w)
                z_new_list.append(zb_new)

            z_new = torch.stack(z_new_list, dim=0)

            # 把 std 拉回 boost 前，避免方差漂移影响采样分布
            std1 = z_new.std(dim=1, keepdim=True, unbiased=False)
            z_new = z_new * (std0 / (std1 + 1e-8))
        if getattr(self, "_dbg_latent_detect_cnt", 0) < 3:
            self._dbg_latent_detect_cnt = getattr(self, "_dbg_latent_detect_cnt", 0) + 1
            zb = z_new[0].detach().to(torch.float64).cpu()     # (D,)
            post = pg.recover_posteriors(zb, variances=float(1.5))  # var 用你检测脚本一致的
            post = torch.as_tensor(post)
            det = prc_lib.Detect(self.prc.decoding_key, post, false_positive_rate=1e-5)
            print(f"[DBG] latent-side Detect={det}  mean|post|={post.abs().mean():.4f}  frac(|post|<0.05)={(post.abs()<0.05).float().mean():.3f}")
        return unflatten_latent(z_new, (B, C, H, W_))


    def ARR_refine(self, zT: torch.Tensor, prompt: str, negative_prompt: str, ARR_cfg: ARRConfig, gen_cfg: DiffusionSamplerConfig) -> torch.Tensor:
        """ARR: gradient update + global PRC repair each step."""
        zT = zT.detach().requires_grad_(True)
            # repair (global PRC)
        zT = self.repair_prc_global(zT).detach().requires_grad_(True)
        return zT.detach()
# -------------------------
# Utils: IO
# -------------------------

def read_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines()]
    lines = [x for x in lines if x]
    return lines


def save_image_grid(images: List[Image.Image], rows: int, cols: int, out_path: str) -> None:
    assert len(images) == rows*cols
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols*w, rows*h))
    for r in range(rows):
        for c in range(cols):
            grid.paste(images[r*cols+c], box=(c*w, r*h))
    grid.save(out_path)


# -------------------------
# CLI main
# -------------------------

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip_margin", type=float, default=0.2, help="CLIP hinge margin for ssp/ARR surrogate")
    parser.add_argument("--save_zT", type=int, default=1, help="Save pre/post ARR zT tensors (.pt)")
    parser.add_argument("--outdir", type=str, required=True)
    # ---- Export-only: directly export merged zT16 (prompt0) without generating images ----
    parser.add_argument("--export_zT16_only", type=int, default=0,
                        help="1: skip image generation; export 16 refined zT (prompt0) as a single [16,4,64,64] pt")
    parser.add_argument("--export_latents_dir", type=str, default="",
                        help="Where to save merged zT16 pt (and optionally wm_meta). If empty, defaults to <outdir>/latents_experiment")
    parser.add_argument("--export_latents_name", type=str, default="",
                        help="Filename for merged zT16 pt, e.g., generate_PRC_w_att_0_85.pt. Required when export_zT16_only=1")
    parser.add_argument("--wm_meta_dir", type=str, default="",
                        help="Where to save wm_meta. If empty, defaults to <outdir>/wm_meta (normal) or <export_latents_dir>/<wm_meta_subdir> (export mode)")
    parser.add_argument("--wm_meta_subdir", type=str, default="wm_meta_prc",
                        help="Subfolder name under export_latents_dir for wm_meta when wm_meta_dir is empty in export mode.")
    parser.add_argument("--export_K", type=int, default=16,
                        help="Number of refined zT to export in export-only mode (default 16).")


    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--gen_bs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)

    # ssp params
    parser.add_argument("--ssp_N_cal", type=int, default=12)
    parser.add_argument("--ssp_energy_ratio", type=float, default=0.9)
    parser.add_argument("--ssp_mini_steps", type=int, default=12)
    parser.add_argument("--ssp_d_sens_max", type=int, default=64)
    parser.add_argument("--ssp_d_wm", type=int, default=256)
    parser.add_argument("--reuse_ssp", type=int, default=1, help="1: reuse ssp basis from outdir/wm_meta/ssp_basis.pt if present; 0: recompute")
    parser.add_argument("--lam1", type=float, default=0.6, help="Mixing weight for PRC watermark (lam1). zT = lam1*z_wm + (1-lam1)*Proj_Bsens(z)")
    parser.add_argument("--zfree_scale", type=float, default=1.0, help="Scale factor for z_sens in zT = z_wm + scale * z_sens.")

    # PRC params
    parser.add_argument("--prc_message_length", type=int, default=32)
    parser.add_argument("--prc_error_prob", type=float, default=0.01)
    parser.add_argument("--master_key", type=str, default="change_me")

    # ARR params
    parser.add_argument("--ARR_T_r", type=int, default=1)
    parser.add_argument("--ARR_eta", type=float, default=0.05)
    parser.add_argument("--ARR_mini_steps", type=int, default=2)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    outdir = args.outdir

    export_only = bool(int(getattr(args, "export_zT16_only", 0)))
    export_dir = str(getattr(args, "export_latents_dir", "")).strip()
    if export_dir == "":
        export_dir = os.path.join(outdir, "latents_experiment")
    os.makedirs(export_dir, exist_ok=True)

    # Only create image folders if we actually generate images
    if not export_only:
        sliced_dir = os.path.join(outdir, "sliced")
        os.makedirs(sliced_dir, exist_ok=True)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    sampler = DiffusionSampler(args.model_id, device=device)

    surrogate_impl = CLIPHingeSurrogate(clip_model_id=str(args.clip_model), device=str(device))

    # latent shape
    # SD uses 8x downsampling
    C = 4
    H = args.height // 8
    W = args.width // 8
    latent_shape = (args.gen_bs, C, H, W)

    workflow = AlignPreserveNaW(
        sampler=sampler,
        surrogate=lambda img, prompt: surrogate_impl(img, prompt, margin=float(args.clip_margin)),
        latent_shape=latent_shape,
        device=str(device),
    )
    # mixing hyperparameter: zT = lam1*z_wm + (1-lam1)*z_sens
    workflow.set_mix_lambda(float(args.lam1))

    prompts = read_prompts(args.prompts)
    rows, cols = int(args.rows), int(args.cols)

    wm_meta_dir = str(getattr(args, "wm_meta_dir", "")).strip()
    if wm_meta_dir != "":
        meta_dir = wm_meta_dir
    else:
        if export_only:
            meta_dir = os.path.join(export_dir, str(getattr(args, "wm_meta_subdir", "wm_meta_prc")))
        else:
            meta_dir = os.path.join(outdir, "wm_meta")
    os.makedirs(meta_dir, exist_ok=True)


    # --- Step 1: ssp (calibration prompts from the prompt file) ---
    ssp_cache_path = os.path.join(meta_dir, "ssp_Bsens.pt")
    use_ssp_cache = bool(int(getattr(args, "reuse_ssp", 1)))
    loaded_ssp = False

    if use_ssp_cache and os.path.isfile(ssp_cache_path):
        try:
            ckpt = torch.load(ssp_cache_path, map_location="cpu")
            B_sens = ckpt.get("B_sens", None)
            if B_sens is None:
                raise ValueError("Cache missing B_sens")
            if int(B_sens.shape[0]) != int(workflow.D):
                raise ValueError(f"D mismatch: cached D={int(B_sens.shape[0])} vs current D={int(workflow.D)}")
            workflow.B_sens = B_sens.to(device=workflow.device, dtype=torch.float32)
            print(f"[ssp] Reusing cached B_sens from {ssp_cache_path} (d_sens={int(workflow.B_sens.shape[1])}).")
            loaded_ssp = True
        except Exception as e:
            print(f"[ssp] Failed to load cached ssp basis: {e}. Recomputing...")

    if not loaded_ssp:
        cal = (prompts if len(prompts) <= int(args.ssp_N_cal) else random.sample(prompts, k=int(args.ssp_N_cal)))[: int(args.ssp_N_cal)]
        ssp_cfg = sspConfig(
            N_cal=len(cal),
            guidance_scale=7.5,
            d_sens_max=int(args.ssp_d_sens_max),
            d_wm=int(getattr(args, 'ssp_d_wm', 0)),  # legacy arg, unused in this PRC-global variant
            energy_ratio=float(args.ssp_energy_ratio),
            mini_steps=int(args.ssp_mini_steps),
        )
        print(f"[ssp] Running ssp with N_cal={ssp_cfg.N_cal}, d_sens_max={ssp_cfg.d_sens_max} ...")
        stats = workflow.run_ssp(cal, ssp_cfg)
        print(f"[ssp] d_sens={int(stats['d_sens'].item())}")

        torch.save(
            {
                "B_sens": workflow.B_sens.detach().cpu() if workflow.B_sens is not None else None,
                "ssp_cfg": vars(ssp_cfg),
                "latent_shape": latent_shape,
                "model_id": args.model_id,
                "scheduler": "DPMSolverMultistepScheduler",
            },
            ssp_cache_path,
        )
        with open(os.path.join(meta_dir, "cal_prompts.txt"), "w", encoding="utf-8") as f:
            for pp in cal:
                f.write(pp + "\n")

    # --- Step 2: Build GLOBAL PRC (official math) and save keys/meta to wm_meta ---
    wm_cfg = WatermarkConfig(
        family="proj",
        proj_mode="prc_global",
        prc_message_length=int(args.prc_message_length),
        prc_error_prob=float(args.prc_error_prob),
        master_key=str(args.master_key),
    )
    prc = GlobalPRCWatermark(workflow.D, wm_cfg)
    workflow.set_prc(prc)
    prc.dump_runtime_artifacts(meta_dir)

    # ARR and full sampling configs
    ARR_cfg = ARRConfig(T_r=int(args.ARR_T_r), eta=float(args.ARR_eta), normalize_grad=True, mini_steps=int(args.ARR_mini_steps))
    full_cfg = DiffusionSamplerConfig(num_steps=int(args.steps), guidance_scale=float(args.cfg), eta=0.0, mini_steps=int(args.ARR_mini_steps))

    if export_only:
        if len(prompts) == 0:
            raise ValueError("No prompts loaded from --prompts")
        prompt0 = prompts[0]

        K = int(getattr(args, "export_K", 16))
        if K <= 0:
            raise ValueError("--export_K must be > 0")

        export_name = str(getattr(args, "export_latents_name", "")).strip()
        if export_name == "":
            raise ValueError("export-only mode requires --export_latents_name, e.g., generate_PRC_w_att_0_85.pt")
        out_pt = os.path.join(export_dir, export_name)

        print(f"[EXPORT] export_only=1, using prompt0, K={K} -> {out_pt}")

        z_list = []
        idx = 0
        for k in range(K):
            seed = int(args.seed) + idx
            set_seed_everywhere(seed)

            zT, z_wm = workflow.SSM_sample(
                batch_size=1,
                seed=seed,
                latent_chw=(C, H, W),
            )

            zT_refined = workflow.ARR_refine(
                zT,
                prompt=prompt0,
                negative_prompt=args.negative_prompt,
                ARR_cfg=ARR_cfg,
                gen_cfg=full_cfg,
            )

            # 只保存 refined 后的 zT
            z_list.append(zT_refined[0].detach().cpu())
            idx += 1

        z_merged = torch.stack(z_list, dim=0)  # [K,4,64,64]
        torch.save(z_merged, out_pt)
        print(f"[OK] Saved merged zT: {out_pt} shape={tuple(z_merged.shape)}")
        print(f"[OK] wm_meta saved under: {meta_dir}")
        print("[DONE]")
        return



if __name__ == "__main__":
    cli_main()
