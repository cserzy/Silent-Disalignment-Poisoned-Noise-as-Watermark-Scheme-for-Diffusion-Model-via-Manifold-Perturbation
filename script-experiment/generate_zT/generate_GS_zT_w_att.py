from __future__ import annotations
import math
import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, List, Literal

import torch
import os
import re
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
from Crypto.Cipher import ChaCha20
from scipy.stats import norm  # ppf for Gaussian Shading
# -----------------------------
# Utilities
# -----------------------------

def seed_from_bytes(b: bytes) -> int:
    # stable 32-bit seed
    h = hashlib.sha256(b).digest()
    return int.from_bytes(h[:4], "big", signed=False)

def key_to_generator(key: bytes, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed_from_bytes(key))
    return g

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def flatten_latent(z: torch.Tensor) -> torch.Tensor:
    # z: (B,C,H,W) -> (B, D)
    return z.reshape(z.shape[0], -1)

def unflatten_latent(z_flat: torch.Tensor, shape: Tuple[int,int,int,int]) -> torch.Tensor:
    # shape: (B,C,H,W)
    return z_flat.reshape(shape)

def orthonormalize_cols_qr(A: torch.Tensor) -> torch.Tensor:
    # A: (D, k)
    # returns Q with orthonormal columns
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q

def projector_from_basis(B: torch.Tensor) -> torch.Tensor:
    # B: (D, k) orthonormal columns -> P = B B^T
    return B @ B.transpose(0, 1)

def project_rows_onto_basis(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # (N,D)@(D,k)->(N,k); then (N,k)@(k,D)->(N,D)
    return (X @ B) @ B.transpose(0, 1)

def remove_rows_component(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Return X with its component in span(B) removed: X - proj_B(X)."""
    return X - project_rows_onto_basis(X, B)

def project_cols_onto_basis(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Project column-vectors A onto span(B) without (D,D): A_proj = B(B^T A)."""
    return B @ (B.transpose(0, 1) @ A)

def remove_cols_component(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Remove span(B) component from column-vectors A: A - proj_B(A)."""
    return A - project_cols_onto_basis(A, B)

def explained_energy_to_rank(svals: torch.Tensor, ratio: float = 0.90) -> int:
    # svals: singular values (descending)
    energy = svals.pow(2)
    cum = torch.cumsum(energy, dim=0)
    total = cum[-1].clamp_min(1e-12)
    k = int(torch.searchsorted(cum / total, torch.tensor(ratio, device=svals.device)).item()) + 1
    return max(1, min(k, svals.numel()))

def randomized_svd_topk(G: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    G: (N, D)
    returns approx U,S,Vt with rank k (Vt: (k, D))
    Uses torch.pca_lowrank which is randomized-ish and efficient for N<<D.
    """
    # q is the oversampling factor; center=False preserves the original gradient statistics.
    U, S, V = torch.pca_lowrank(G, q=min(k + 8, min(G.shape) - 1), center=False, niter=2)
    # V: (D, q), want top-k right singular vecs
    V_k = V[:, :k]  # (D, k)
    S_k = S[:k]
    # approximate U_k not needed usually; build if desired:
    U_k = U[:, :k]
    return U_k, S_k, V_k.transpose(0, 1)  # Vt: (k, D)

# -----------------------------
# Differentiable mini-sampler (DDIM-like) for SD latents
# -----------------------------

@dataclass
class DiffusionSamplerConfig:
    num_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0  # DDIM eta (0 makes it deterministic)
    # For mini-sampler:
    mini_steps: int = 6
    # In most NaW inversion/extraction they set guidance=1 and prompt empty, but here is online prompt-aware.
    # You can override per-call.

class DiffusionLatentSampler:
    def __init__(self, pipe, device: torch.device, dtype: torch.dtype = torch.float32):
        self.pipe = pipe
        self.device = device
        self.dtype = dtype

        # Make sure scheduler timesteps are set each call.

    @torch.no_grad()
    def encode_prompt(self, prompt: str, negative_prompt: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self.pipe.tokenizer

        def _enc(p: str):
            inp = tok(
                p,
                padding="max_length",
                max_length=tok.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            emb = self.pipe.text_encoder(inp.input_ids)[0]
            return emb

        cond = _enc(prompt)
        uncond = _enc(negative_prompt)
        return cond, uncond

    def _predict_eps(self, latents: torch.Tensor, t: torch.Tensor,
                     cond: torch.Tensor, uncond: torch.Tensor,
                     guidance_scale: float) -> torch.Tensor:
        # latents: (B,C,H,W)
        # classifier-free guidance
        latent_in = torch.cat([latents, latents], dim=0)
        t_in = torch.cat([t, t], dim=0)

        # text embeddings (2B, L, D)
        text_in = torch.cat([uncond, cond], dim=0)

        eps = self.pipe.unet(latent_in, t_in, encoder_hidden_states=text_in).sample
        eps_uncond, eps_cond = eps.chunk(2)
        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        return eps_guided

    def sample_latents(self,
                       zT: torch.Tensor,
                       prompt: str,
                       cfg: DiffusionSamplerConfig,
                       negative_prompt: str = "") -> torch.Tensor:
        """
        zT: (B,C,H,W) initial noise latents
        returns x0-latents after cfg.num_steps denoise steps
        """
        # Ensure grad can flow from output back to zT
        assert zT.requires_grad, "zT must require_grad=True for differentiable sampling."

        cond, uncond = self.encode_prompt(prompt, negative_prompt=negative_prompt)
        # broadcast to batch
        B = zT.shape[0]
        if cond.shape[0] != B:
            cond = cond.expand(B, -1, -1).contiguous()
            uncond = uncond.expand(B, -1, -1).contiguous()

        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(cfg.num_steps, device=self.device)

        latents = zT
        # scale by scheduler if needed (diffusers convention)
        if hasattr(scheduler, "init_noise_sigma"):
            latents = latents * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            t_b = torch.full((B,), t, device=self.device, dtype=torch.long)
            # diffusers sometimes expects float
            t_in = t_b
            eps = self._predict_eps(latents, t_in, cond, uncond, cfg.guidance_scale)
            try:
                step_out = scheduler.step(eps, t, latents, eta=cfg.eta)
            except TypeError:
                step_out = scheduler.step(eps, t, latents)
            latents = step_out.prev_sample

        return latents

    def mini_sample_image(self,
                          zT: torch.Tensor,
                          prompt: str,
                          cfg: DiffusionSamplerConfig,
                          negative_prompt: str = "") -> torch.Tensor:
        """
        Mini sampler: do cfg.mini_steps steps and decode to image tensor in [0,1].
        """
        mini_cfg = DiffusionSamplerConfig(
            num_steps=cfg.mini_steps,
            guidance_scale=cfg.guidance_scale,
            eta=cfg.eta,
            mini_steps=cfg.mini_steps
        )
        x_lat = self.sample_latents(zT, prompt, mini_cfg, negative_prompt=negative_prompt)
        # decode latents -> image (differentiable)
        return self.decode_latents(x_lat)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images in [0,1], shape (B,3,H,W).
        """
        # SD latent scaling factor
        scale = getattr(self.pipe, "vae_scale_factor", 8)
        # diffusers SD convention: latents are scaled by 1/0.18215
        # Many pipelines keep a constant 0.18215, but some store it in config.
        scaling = getattr(self.pipe.vae.config, "scaling_factor", 0.18215)

        latents_in = latents / scaling
        img = self.pipe.vae.decode(latents_in).sample  # (B,3,H,W) in [-1,1]
        img = (img + 1.0) / 2.0
        img = img.clamp(0, 1)
        return img
    @torch.no_grad()
    def sample_latents_infer(self,
                            zT: torch.Tensor,
                            prompt: str,
                            cfg: DiffusionSamplerConfig,
                            negative_prompt: str = "") -> torch.Tensor:
        """
        Inference-only sampling: no autograd graph, much lower VRAM.
        """
        # Detach zT before inference-only sampling
        zT = zT.detach()

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

class CLIPHingeSurrogate(torch.nn.Module):
    def __init__(self, clip_model_id: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(device)
        self.clip = CLIPModel.from_pretrained(clip_model_id).to(self.device)
        self.proc = CLIPProcessor.from_pretrained(clip_model_id)
        self.clip.eval()

        # Freeze CLIP params to keep gradients only w.r.t. image
        for p in self.clip.parameters():
            p.requires_grad_(False)

    def forward(self, images_01: torch.Tensor, prompt: str, margin: float = 0.25) -> torch.Tensor:
        # CLIPProcessor is not differentiable (it uses PIL/np). We'll do a simple differentiable preprocess:
        # resize to 224 and normalize with CLIP mean/std.
        B = images_01.shape[0]
        img = F.interpolate(images_01, size=(224,224), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=img.device).view(1,3,1,1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=img.device).view(1,3,1,1)
        img = (img - mean) / std

        # text tokens (no grad needed)
        with torch.no_grad():
            tok = self.proc(text=[prompt] * B, images=None, return_tensors="pt", padding=True).to(self.device)
            text_emb = self.clip.get_text_features(**{k: tok[k] for k in ["input_ids", "attention_mask"]})
            text_emb = l2_normalize(text_emb, dim=-1)

        img_emb = self.clip.get_image_features(pixel_values=img)
        img_emb = l2_normalize(img_emb, dim=-1)

        sim = (img_emb * text_emb).sum(dim=-1)  # cosine since normalized
        loss = F.relu(margin - sim).mean()
        print("sim:", sim.detach().cpu().tolist(), "margin:", margin, "loss:", loss.item())
        return loss

C, H, W = 4, 64, 64

def bits_bin_to_bytes(bits: str) -> bytes:
    bits = bits.strip()
    if not set(bits) <= {"0","1"}:
        raise ValueError("Binary key/nonce strings may only contain 0 or 1.")
    if len(bits) % 8 != 0:
        raise ValueError("Binary strings must have a length that is a multiple of 8.")
    out = bytearray()
    for i in range(0, len(bits), 8):
        out.append(int(bits[i:i+8], 2))
    return bytes(out)

def parse_key_32bytes(args) -> bytes:
    # Mutually exclusive: --key_ones / --key_hex / --key_bin
    cnt = int(args.key_ones) + (args.key_hex is not None) + (args.key_bin is not None)
    if cnt != 1:
        raise ValueError("Exactly one key input must be provided: --key_ones, --key_hex, or --key_bin.")
    if args.key_ones:
        return b"\xff" * 32  # 32 bytes of all ones, i.e. 256 one-bits.
    if args.key_hex is not None:
        hx = args.key_hex.strip().lower()
        if len(hx) != 64 or any(ch not in "0123456789abcdef" for ch in hx):
            raise ValueError("key_hex must contain exactly 64 hexadecimal characters (32 bytes).")
        return bytes.fromhex(hx)
    if args.key_bin is not None:
        bs = args.key_bin.strip()
        if len(bs) != 256:
            raise ValueError("key_bin must be a 256-bit 0/1 string (32 bytes).")
        return bits_bin_to_bytes(bs)
    raise AssertionError

def parse_nonce_12bytes(args) -> bytes:
    # Use exactly one nonce option; otherwise fall back to a fixed nonce for reproducibility
    if args.nonce_zero:
        return b"\x00" * 12
    if args.nonce_hex is not None:
        hx = args.nonce_hex.strip().lower()
        if len(hx) != 24 or any(ch not in "0123456789abcdef" for ch in hx):
            raise ValueError("nonce_hex must contain exactly 24 hexadecimal characters (12 bytes).")
        return bytes.fromhex(hx)
    if args.nonce_bin is not None:
        bs = args.nonce_bin.strip()
        if len(bs) != 96:
            raise ValueError("nonce_bin must be a 96-bit 0/1 string (12 bytes).")
        return bits_bin_to_bytes(bs)
    # Fixed nonce for experiment reproducibility; not recommended for production reuse
    return b"GS-fixed-nc!"  # 12 bytes

def make_base_bits(k_bits: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(k_bits,), dtype=np.int8)

def diffuse_bits_to_chw(base_bits: np.ndarray, fc: int, fhw: int) -> np.ndarray:
    assert base_bits.ndim == 1
    c0 = C // fc; h0 = H // fhw; w0 = W // fhw
    assert base_bits.size == c0 * h0 * w0
    sd_small = base_bits.reshape(c0, h0, w0)
    sd_c = np.tile(sd_small, (fc, 1, 1))     # (C, h0, w0)
    sd_hw = np.tile(sd_c, (1, fhw, fhw))     # (C, H, W)
    return sd_hw.astype(np.int8)

def chacha20_xor_bits(bits_c_hw: np.ndarray, key32: bytes, nonce12: bytes) -> np.ndarray:
    flat = bits_c_hw.reshape(-1).astype(np.uint8)
    packed = np.packbits(flat)  # default MSB-first, same as official
    cipher = ChaCha20.new(key=key32, nonce=nonce12)  # 32B key + 12B nonce
    out_bytes = cipher.encrypt(packed.tobytes())
    out_bits = np.unpackbits(np.frombuffer(out_bytes, dtype=np.uint8))[: flat.size]
    return out_bits.astype(np.int8).reshape(bits_c_hw.shape)


def sample_latents_from_bits(m_bits: np.ndarray, n_samples: int, l: int = 1) -> torch.Tensor:
    assert l == 1
    y = m_bits.astype(np.int8)  # (C,H,W) in {0,1}
    latents = []
    for _ in range(n_samples):
        u = np.random.rand(C, H, W).astype(np.float64)  # U(0,1)
        z = norm.ppf((u + y) * 0.5)                     # (u+y)/2
        latents.append(torch.from_numpy(z).to(torch.float32))
    return torch.stack(latents, dim=0)  # [n,4,64,64]


# -----------------------------
# Watermark families: constraints & repair
# -----------------------------

Family = Literal["proj", "qbin", "freq", "gsqbin"]

@dataclass
class WatermarkConfig:
    family: Family = "proj"

    # common:
    J: int = 256                 # number of bits/statistics
    tau: float = 0.5             # margin threshold (proj) OR generic strength knob
    device: str = "cuda"

    # qbin:
    # target intervals [L_b, R_b], plus inner safety margin delta
    qbin_L0: float = -2.0
    qbin_R0: float = -0.5
    qbin_L1: float = 0.5
    qbin_R1: float = 2.0
    qbin_delta: float = 0.1


    # gaussian shading (gsqbin): official qbin sampler (sign-conditioned, distribution-preserving)
    gs_key32: bytes = b''
    gs_nonce12: bytes = b''
    gs_seed: int = 0
    gs_ch: int = 4   # diffusion factor along channel (fc)
    gs_hw: int = 4   # diffusion factor along spatial (fhw)
    gs_l: int = 1    # qbin level (Gaussian Shading uses l=1)
    # freq:
    alpha: float = 0.85          # imprint strength
    freq_mask_ratio: float = 0.15  # fraction of FFT coeffs masked
    freq_use_dct: bool = False   # reserved for future DCT support; current implementation uses FFT

    # gsqbin (strict alignment for family="freq"):
    # Tree-Ring does not encode W explicitly; this implementation keys it by K/seed.
    tr_seed_from_key: bool = False  # if True, derive per-key seed and xor with tr_w_seed
    tr_w_seed: int = 12345         # base seed (xor'd with derived seed when tr_seed_from_key=True)
    tr_w_channel: int = -1         # -1 for all channels; else 0..3
    tr_w_pattern: str = "ring"     # zeros|rand|ring|seed_rand|seed_zeros|seed_ring|const
    tr_w_mask_shape: str = "circle"  # circle|square|no
    tr_w_radius: int = 9
    tr_w_injection: str = "complex"  # complex|seed
    tr_w_pattern_const: float = 0.0

class WatermarkFamilyBase:
    def __init__(self, D: int, wm_cfg: WatermarkConfig, B_wm: torch.Tensor):
        self.D = D
        self.cfg = wm_cfg
        self.device = torch.device(wm_cfg.device)
        self.B_wm = B_wm  # (D,k) orthonormal basis

    def derive_bits(self, W: bytes, K: bytes, J: int) -> torch.Tensor:
        # Return bits in {-1,+1}, shape (J,)
        g = key_to_generator(b"bits|" + K + b"|" + W, self.device)
        bits01 = torch.randint(low=0, high=2, size=(J,), generator=g, device=self.device)
        return bits01 * 2 - 1

    def derive_vectors(self, K: bytes, J: int) -> torch.Tensor:
        """
        Return v: (J, D) with ||v_j||=1.
        Vectors are generated from key, then projected into watermark subspace and normalized.
        """
        g = key_to_generator(b"v|" + K, self.device)
        V = torch.randn((J, self.D), generator=g, device=self.device)
        # project to U_wm: v <- proj_{B_wm}(v)
        V = project_rows_onto_basis(V, self.B_wm)  # (J,D)
        V = l2_normalize(V, dim=-1)
        return V

    def constrain(self, z_wm: torch.Tensor, W: bytes, K: bytes) -> torch.Tensor:
        raise NotImplementedError

    def repair(self, z_wm: torch.Tensor, W: bytes, K: bytes) -> torch.Tensor:
        # default: same as constrain
        return self.constrain(z_wm, W, K)

    def decode(self, z_hat_wm: torch.Tensor, W: bytes, K: bytes) -> torch.Tensor:
        raise NotImplementedError


class ProjSignMarginFamily(WatermarkFamilyBase):
    """
    Implements pasted.txt (C) projection/code constraints:
      enforce b_j <z, v_j> >= tau with one-pass minimal correction.
    """
    def constrain(self, z_wm: torch.Tensor, W: bytes, K: bytes) -> torch.Tensor:
        # z_wm: (B,D)
        B = z_wm.shape[0]
        J = self.cfg.J
        tau = self.cfg.tau

        b = self.derive_bits(W, K, J)          # (J,)
        V = self.derive_vectors(K, J)          # (J,D)

        z = z_wm
        # One-pass sequential enforcement (as in pasted.txt)
        for j in range(J):
            v = V[j]  # (D,)
            # s = <z, v> for each batch
            s = (z * v).sum(dim=-1)  # (B,)
            cond = b[j] * s < tau
            if cond.any():
                # z <- z + (tau - b*s)*b*v
                delta = (tau - b[j] * s).clamp_min(0.0)  # (B,)
                z = z + (delta * b[j]).unsqueeze(-1) * v.unsqueeze(0)
        return z

    def decode(self, z_hat_wm: torch.Tensor, W: bytes, K: bytes) -> torch.Tensor:
        # Recover bits by sign of projections
        J = self.cfg.J
        V = self.derive_vectors(K, J)  # (J,D)
        p = z_hat_wm @ V.transpose(0, 1)  # (B,J)
        bits = torch.sign(p).clamp(min=-1, max=1)
        bits[bits == 0] = 1
        return bits


class QuantileBinFamily(WatermarkFamilyBase):
    """
    Implements pasted.txt (B) quantile-bin constraints:
      s_j <- clip(s_j(z), [L_b+delta, R_b-delta]); z <- z + (s_j - <z,v_j>) v_j
    """
    def constrain(self, z_wm: torch.Tensor, W: bytes, K: bytes) -> torch.Tensor:
        B = z_wm.shape[0]
        J = self.cfg.J
        delta = self.cfg.qbin_delta

        L0, R0 = self.cfg.qbin_L0, self.cfg.qbin_R0
        L1, R1 = self.cfg.qbin_L1, self.cfg.qbin_R1

        b = self.derive_bits(W, K, J)          # {-1,+1}, map -1->0, +1->1
        b01 = (b > 0).long()

        V = self.derive_vectors(K, J)          # (J,D)
        z = z_wm

        for j in range(J):
            v = V[j]
            s = (z * v).sum(dim=-1)  # (B,)
            L = torch.where(b01[j] == 0, torch.tensor(L0, device=z.device), torch.tensor(L1, device=z.device))
            R = torch.where(b01[j] == 0, torch.tensor(R0, device=z.device), torch.tensor(R1, device=z.device))
            s_tgt = torch.clamp(s, min=L + delta, max=R - delta)
            # correction along v
            z = z + (s_tgt - s).unsqueeze(-1) * v.unsqueeze(0)
        return z

    def decode(self, z_hat_wm: torch.Tensor, W: bytes, K: bytes) -> torch.Tensor:
        J = self.cfg.J
        V = self.derive_vectors(K, J)
        p = z_hat_wm @ V.transpose(0, 1)  # (B,J)
        # decide by which interval center it's closer to; simplest: sign split at 0
        bits = torch.where(p >= 0, torch.ones_like(p), -torch.ones_like(p))
        return bits


class FrequencyMaskFamily(WatermarkFamilyBase):
    """
    Tree-Ring (strictly aligned with the official implementation) as the "freq" family.

    Core idea (official):
      - Construct a boolean mask M in the *Fourier domain* (circle/square) on selected latent channel(s).
      - Construct a key-dependent Fourier "ground-truth patch" (zeros/rand/ring/...) using a seed.
      - Inject watermark by replacing masked Fourier coefficients, then inverse FFT back to latent space.
    """

    def __init__(self, D: int, wm_cfg: WatermarkConfig, B_wm: torch.Tensor,
                 latent_shape: Tuple[int, int, int]):  # (C,H,W)
        super().__init__(D, wm_cfg, B_wm)
        self.C, self.H, self.W = latent_shape
        # Cache per-(K,B) to avoid regenerating masks/patterns across ARR iterations.
        self._cache: Dict[Tuple[bytes, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _derive_seed(self, K: bytes) -> int:
        # Derive a stable 32-bit seed from K (optionally mixed with a base seed).
        s = seed_from_bytes(b"gsqbin|" + K)
        base = int(getattr(self.cfg, "tr_w_seed", 12345))
        if bool(getattr(self.cfg, "tr_seed_from_key", True)):
            return (base ^ s) & 0xFFFFFFFF
        return base & 0xFFFFFFFF

    def _circle_mask(self, size: int, r: int, device: torch.device,
                     x_offset: int = 0, y_offset: int = 0) -> torch.Tensor:
        # Matches the official "circle_mask" (solid disk). Uses a flipped y-grid (y[::-1]) like the reference.
        x0 = size // 2 + x_offset
        y0 = size // 2 + y_offset
        y = torch.arange(size - 1, -1, -1, device=device).view(size, 1)  # reversed
        x = torch.arange(size, device=device).view(1, size)
        return ((x - x0) ** 2 + (y - y0) ** 2) <= (r ** 2)

    def _get_watermarking_mask(self, B: int, device: torch.device) -> torch.Tensor:
        """
        Official-style mask:
          - w_mask_shape in {circle, square, no}
          - w_channel == -1 means all channels; else only that channel.
        Returns: mask bool tensor of shape (B,C,H,W).
        """
        mask = torch.zeros((B, self.C, self.H, self.W), dtype=torch.bool, device=device)

        shape = str(getattr(self.cfg, "tr_w_mask_shape", "circle"))
        r = int(getattr(self.cfg, "tr_w_radius", 10))
        ch = int(getattr(self.cfg, "tr_w_channel", 0))

        if shape == "circle":
            # The reference implementation assumes H==W; use min(H, W) for non-square inputs.
            size = int(self.H)
            disk = self._circle_mask(size=size, r=r, device=device)
            if ch == -1:
                mask[:, :] = disk
            else:
                mask[:, ch] = disk
        elif shape == "square":
            anchor_p = self.H // 2
            sl_h = slice(anchor_p - r, anchor_p + r)
            anchor_q = self.W // 2
            sl_w = slice(anchor_q - r, anchor_q + r)
            if ch == -1:
                mask[:, :, sl_h, sl_w] = True
            else:
                mask[:, ch, sl_h, sl_w] = True
        elif shape == "no":
            pass
        else:
            raise NotImplementedError(f"tr_w_mask_shape: {shape}")

        return mask

    def _get_watermarking_pattern(self, seed: int, device: torch.device,
                                  shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        Reference-style pattern generation adapted from gsqbin_official:
          - zeros / rand / ring / seed_* / const
        Output "gt_patch" is either a Fourier-domain patch (rand/zeros/ring/const) or time-domain patch (seed_*).
        """
        # Use a local generator to avoid perturbing global RNG state.
        gen = torch.Generator(device=device)
        # Match the official seeding scheme: torch.manual_seed(seed), torch.cuda.manual_seed(seed+1)
        seed_eff = int(seed) + (1 if device.type == "cuda" else 0)
        gen.manual_seed(seed_eff & 0xFFFFFFFF)

        pattern = str(getattr(self.cfg, "tr_w_pattern", "rand"))
        r = int(getattr(self.cfg, "tr_w_radius", 10))
        const = float(getattr(self.cfg, "tr_w_pattern_const", 0.0))

        gt_init = torch.randn(shape, generator=gen, device=device)

        if "seed_ring" in pattern:
            gt_patch = gt_init
            gt_patch_tmp = gt_patch.clone()
            size = gt_init.shape[-1]
            for i in range(r, 0, -1):
                tmp_mask = self._circle_mask(size=size, r=i, device=device)
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
        elif "seed_zeros" in pattern:
            gt_patch = gt_init * 0
        elif "seed_rand" in pattern:
            gt_patch = gt_init
        elif "rand" in pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            gt_patch[:] = gt_patch[0]  # batch shares the same reference spectrum
        elif "zeros" in pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        elif "const" in pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
            gt_patch = gt_patch + const
        elif "ring" in pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            gt_patch_tmp = gt_patch.clone()
            size = gt_init.shape[-1]
            for i in range(r, 0, -1):
                tmp_mask = self._circle_mask(size=size, r=i, device=device)
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
        else:
            raise NotImplementedError(f"tr_w_pattern: {pattern}")

        return gt_patch

    def _get_cached_mask_patch(self, K: bytes, B: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (K, B)
        if key in self._cache:
            return self._cache[key]

        seed = self._derive_seed(K)
        mask = self._get_watermarking_mask(B=B, device=device)  # (B,C,H,W) bool
        # Generate base patch for this batch shape to keep behavior consistent across B.
        patch = self._get_watermarking_pattern(seed=seed, device=device, shape=(B, self.C, self.H, self.W))
        self._cache[key] = (mask, patch)
        return mask, patch

    def _inject_watermark(self, z4: torch.Tensor, mask: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        """
        Official injection:
          - complex: FFT -> shift -> replace masked complex coeffs -> inverse shift -> IFFT -> real
          - seed   : directly replace in latent space
        """
        inj = str(getattr(self.cfg, "tr_w_injection", "complex"))
        if inj == "complex":
            z_fft = torch.fft.fftshift(torch.fft.fft2(z4), dim=(-1, -2))
            z_fft[mask] = patch[mask].clone()
            out = torch.fft.ifft2(torch.fft.ifftshift(z_fft, dim=(-1, -2))).real
            return out
        if inj == "seed":
            z_out = z4.clone()
            z_out[mask] = patch[mask].clone()
            return z_out
        raise NotImplementedError(f"tr_w_injection: {inj}")

    def constrain(self, z_wm: torch.Tensor, W: bytes, K: bytes) -> torch.Tensor:
        # z_wm: (B,D) -> (B,C,H,W) -> inject -> (B,D)
        B = z_wm.shape[0]
        z4 = z_wm.reshape(B, self.C, self.H, self.W)

        mask, patch = self._get_cached_mask_patch(K=K, B=B, device=z4.device)
        z_out = self._inject_watermark(z4, mask=mask, patch=patch)
        return z_out.reshape(B, -1)

    def decode(self, z_hat_wm: torch.Tensor, W: bytes, K: bytes) -> torch.Tensor:
        # Simple verification score (higher is "more consistent with patch"):
        # negative mean L1 distance on masked Fourier coeffs (complex domain).
        B = z_hat_wm.shape[0]
        z4 = z_hat_wm.reshape(B, self.C, self.H, self.W)
        mask, patch = self._get_cached_mask_patch(K=K, B=B, device=z4.device)

        z_fft = torch.fft.fftshift(torch.fft.fft2(z4), dim=(-1, -2))
        # If patch is real (seed_* patterns), cast to complex for measurement.
        patch_c = patch.to(z_fft.dtype)
        dist = (z_fft[mask] - patch_c[mask]).abs().mean().view(1, 1)
        return -dist




def build_family(D: int, wm_cfg: WatermarkConfig, B_wm: torch.Tensor,
                 latent_chw: Optional[Tuple[int,int,int]] = None) -> WatermarkFamilyBase:
    if wm_cfg.family == "proj":
        return ProjSignMarginFamily(D, wm_cfg, B_wm)
    if wm_cfg.family == "qbin":
        return QuantileBinFamily(D, wm_cfg, B_wm)
    if wm_cfg.family == "freq":
        assert latent_chw is not None, "freq family requires latent_chw=(C,H,W)"
        return FrequencyMaskFamily(D, wm_cfg, B_wm, latent_shape=latent_chw)
    raise ValueError(f"Unknown family={wm_cfg.family}")

# -----------------------------
# Main workflow: ssp + SSM + ARR
# -----------------------------

@dataclass
class sspConfig:
    N_cal: int = 64
    d_sens_max: int = 64
    energy_ratio: float = 0.90
    d_wm: int = 256
    mini_steps: int = 6
    guidance_scale: float = 7.5
    eta_ddim: float = 0.0

@dataclass
class ARRConfig:
    T_r: int = 2
    eta: float = 0.15
    normalize_grad: bool = True
    mini_steps: int = 6
    guidance_scale: float = 7.5
    eta_ddim: float = 0.0

class AlignPreserveNaW:
    """
    End-to-end workflow wrapper.
    """
    def __init__(self, sampler: DiffusionLatentSampler, surrogate: Callable[[torch.Tensor, str], torch.Tensor],
                 latent_shape: Tuple[int,int,int,int], device: str = "cuda"):
        self.sampler = sampler
        self.surrogate = surrogate
        self.device = torch.device(device)
        self.latent_shape = latent_shape  # (B,C,H,W)
        self.D = latent_shape[1] * latent_shape[2] * latent_shape[3]
        # to be filled by ssp:
        self.B_sens: Optional[torch.Tensor] = None  # (D, d_sens)
        # Retained for backward compatibility; unused in the current GS workflow
        self.B_wm: Optional[torch.Tensor] = None
        self.P_wm: Optional[torch.Tensor] = None
        self.P_free: Optional[torch.Tensor] = None

        # SSM mix weights: lambda1^2 + lambda2^2 = 1
        self.lambda1: float = 1.0
        self.lambda2: float = 0.0

        # Backward compatibility fields; unused in the current GS workflow.
        self.zfree_scale: float = 1.0
        self.gs_latent_seed = None  # optional seed for GS latent sampling (z_wm,z_sens)

        # For saving/debugging: last SSM components (detached)
        self._last_z_wm: Optional[torch.Tensor] = None   # (B,C,H,W)
        self._last_z_sens: Optional[torch.Tensor] = None # (B,C,H,W)


        # cache for Gaussian Shading bits (computed once per (key, nonce, seed, fc, fhw, latent_shape))
        self._gs_cached_sig: Optional[tuple] = None
        self._gs_cached_m_bits_np: Optional[np.ndarray] = None  # (C,H,W) int8 in {0,1} on CPU
        self._gs_cached_mask_pos: Optional[torch.Tensor] = None  # (1,C,H,W) bool on device
    def run_ssp(self, prompts: List[str], ssp_cfg: sspConfig) -> Dict[str, torch.Tensor]:
        """
        Estimate U_sens by gradient stats, then construct U_wm away from it.
        """
        assert len(prompts) >= ssp_cfg.N_cal, "Need at least N_cal prompts for ssp calibration."
        B = 1

        grads = []
        for i in range(ssp_cfg.N_cal):
            prompt = prompts[i]
            # random zT
            C, H, W_ = self.latent_shape[1], self.latent_shape[2], self.latent_shape[3]
            zT = torch.randn((1, C, H, W_), device=self.device, dtype=torch.float32, requires_grad=True)

            cfg = DiffusionSamplerConfig(
                num_steps=ssp_cfg.mini_steps,
                guidance_scale=ssp_cfg.guidance_scale,
                eta=ssp_cfg.eta_ddim,
                mini_steps=ssp_cfg.mini_steps
            )
            x_tilde = self.sampler.mini_sample_image(zT, prompt, cfg)
            loss = self.surrogate(x_tilde, prompt)
            g = torch.autograd.grad(loss, zT, retain_graph=False, create_graph=False)[0]

            g_flat = flatten_latent(g).detach().cpu()  # (1,D)
            grads.append(g_flat)
        G = torch.cat(grads, dim=0)  # (N_cal, D)

        # randomized SVD / PCA
        k_try = min(ssp_cfg.d_sens_max, min(G.shape) - 1)
        _, S, Vt = randomized_svd_topk(G, k=k_try)  # Vt: (k_try, D)
        d_sens = explained_energy_to_rank(S, ratio=ssp_cfg.energy_ratio)

        B_sens = Vt[:d_sens].transpose(0, 1).contiguous()  # (D, d_sens)
        B_sens = orthonormalize_cols_qr(B_sens)

        # Only keep the sensitive basis in the new GS workflow.
        self.B_sens = B_sens.to(self.device)
        self.B_wm = None
        self.P_wm, self.P_free = None, None

        return {
            "B_sens": self.B_sens,
            "d_sens": torch.tensor([int(self.B_sens.shape[1])], device=self.device),
        }

    def SSM_sample(self, W: bytes, K: bytes, wm_cfg: WatermarkConfig) -> torch.Tensor:
        """
        Step 2: Entropy-Buffered Sampling:
          z_sens ~ N(0,I), project to free;
          Z_wm   ~ N(0,I), project to wm and apply constraint operator;
          Z_T = Z_wm + z_sens
        Returns zT in (B,C,H,W), requires_grad=True (for ARR).
        """
        assert self.B_sens is not None, "Call run_ssp() first."
        B, C, H, W_ = self.latent_shape
        # Specialization for Tree-Ring / frequency-mask family:
        # For wm_cfg.family == "freq", define the watermark subspace directly in Fourier domain:
        #   z_sens = Z * (~M),  Z_wm = Z * M, where M is the Tree-Ring frequency mask.
        # This avoids mixing a frequency-mask watermark with a random projection basis (B_wm).


        # Specialization for Gaussian Shading (qbin, distribution-preserving sampling):
        # Follow the official embedding pipeline:
        #   base_bits -> diffuse -> ChaCha20 randomization -> distribution-preserving sampling (l=1)
        # This yields a standard normal latent (after marginalizing bits) and is plug-and-play.

        # Specialization for Gaussian Shading (gsqbin):
        # Only enforce the sign bin per-position (l=1), keep magnitudes unchanged.
        if wm_cfg.family == "gsqbin":
            # ---- build / cache GS bit-mask (same as before) ----
            globals()['C'] = int(C)
            globals()['H'] = int(H)
            globals()['W'] = int(W_)
            device = self.device

            sig = (wm_cfg.gs_key32, wm_cfg.gs_nonce12, int(wm_cfg.gs_seed),
                   int(wm_cfg.gs_ch), int(wm_cfg.gs_hw), (C, H, W_))
            if self._gs_cached_sig != sig or self._gs_cached_m_bits_np is None:
                rng = np.random.default_rng(int(wm_cfg.gs_seed))
                k_bits = (C * H * W_) // (int(wm_cfg.gs_ch) * int(wm_cfg.gs_hw) * int(wm_cfg.gs_hw))
                base_bits = make_base_bits(k_bits, rng)
                sd = diffuse_bits_to_chw(base_bits, fc=int(wm_cfg.gs_ch), fhw=int(wm_cfg.gs_hw))
                m_bits = chacha20_xor_bits(sd, key32=wm_cfg.gs_key32, nonce12=wm_cfg.gs_nonce12)
                self._gs_cached_sig = sig
                self._gs_cached_m_bits_np = m_bits
                self._gs_cached_mask_pos = None

            if self._gs_cached_mask_pos is None or self._gs_cached_mask_pos.device != device:
                m = torch.from_numpy(self._gs_cached_m_bits_np.astype(np.int8)).to(device)
                self._gs_cached_mask_pos = (m.view(1, C, H, W_) == 1)

            mask_pos = self._gs_cached_mask_pos  # (1,C,H,W)

            # ---- (1) embed GS watermark: sample z_wm ~ N(0,1) then enforce sign-bin (l=1) ----
            # Reproducible GS latent sampling (optional)
            gen = None
            seed_base = getattr(self, "gs_latent_seed", None)
            if seed_base is not None:
                gen = torch.Generator(device=device)
                seed_eff = int(seed_base) + (1 if self.device.type == "cuda" else 0)
                gen.manual_seed(seed_eff)
            z_wm = torch.randn((B, C, H, W_), device=device, generator=gen)
            z_wm = torch.where(mask_pos, z_wm.abs(), -z_wm.abs())  # bit=1 -> +|z|, bit=0 -> -|z|

            # ---- (2) sample z ~ N(0,I) and project onto B_sens: z_sens = B_sens(B_sens^T z) ----
            z = torch.randn((B, self.D), device=device, generator=gen)
            z_sens_flat = project_rows_onto_basis(z, self.B_sens)              # (B,D)
            z_sens = unflatten_latent(z_sens_flat, (B, C, H, W_))              # (B,C,H,W)

            # ---- (3) mix with lambda1^2 + lambda2^2 = 1 ----
            zT = (self.lambda1 * z_wm + self.lambda2 * z_sens)
            return zT.detach().requires_grad_(True)

    def repair_only_wm(self, zT: torch.Tensor, W: bytes, K: bytes, wm_cfg: WatermarkConfig) -> torch.Tensor:
        """
        Implements pasted.txt repair rule: only correct watermark component, keep entropy buffer unchanged.
        """
        B, C, H, W_ = zT.shape
        # Specialization for Tree-Ring / frequency-mask family:
        # Keep unmasked frequencies unchanged; only repair (overwrite) the masked band.
        if wm_cfg.family == "freq":
            family = build_family(self.D, wm_cfg, self.B_wm, latent_chw=(C, H, W_))
            mask, patch = family._get_cached_mask_patch(K=K, B=B, device=zT.device)

            Z = torch.fft.fftshift(torch.fft.fft2(zT), dim=(-1, -2))
            patch_fft = patch
            if not torch.is_complex(patch_fft):
                patch_fft = torch.fft.fftshift(torch.fft.fft2(patch_fft), dim=(-1, -2))
            patch_fft = patch_fft.to(Z.dtype)

            Z[mask] = patch_fft[mask]
            z_fixed = torch.fft.ifft2(torch.fft.ifftshift(Z, dim=(-1, -2))).real
            return z_fixed



        # Specialization for Gaussian Shading (gsqbin):
        # Only enforce the sign bin per-position (l=1), keep magnitudes and other parts unchanged.

        # Specialization for Gaussian Shading (gsqbin):
        # (1) Embed watermark into a standard normal latent z_wm using the official distribution-preserving sampler.
        # (2) Sample a fresh Gaussian latent z ~ N(0,I) and project it onto the sensitive subspace B_sens to get z_sens.
        # (3) Mix: zT = lambda1 * z_wm + lambda2 * z_sens, with lambda1^2 + lambda2^2 = 1.
        if wm_cfg.family == "gsqbin":
            # set module-level (C,H,W) for copied GS helpers (avoid Python 'global' clash with local C/H)
            globals()['C'] = int(C)
            globals()['H'] = int(H)
            globals()['W'] = int(W_)
            sig = (wm_cfg.gs_key32, wm_cfg.gs_nonce12, int(wm_cfg.gs_seed), int(wm_cfg.gs_ch), int(wm_cfg.gs_hw), (C, H, W_))
            if self._gs_cached_sig != sig or self._gs_cached_m_bits_np is None:
                rng = np.random.default_rng(int(wm_cfg.gs_seed))
                k_bits = (C * H * W_) // (int(wm_cfg.gs_ch) * int(wm_cfg.gs_hw) * int(wm_cfg.gs_hw))
                base_bits = make_base_bits(k_bits, rng)
                sd = diffuse_bits_to_chw(base_bits, fc=int(wm_cfg.gs_ch), fhw=int(wm_cfg.gs_hw))
                m_bits = chacha20_xor_bits(sd, key32=wm_cfg.gs_key32, nonce12=wm_cfg.gs_nonce12)
                self._gs_cached_sig = sig
                self._gs_cached_m_bits_np = m_bits
                self._gs_cached_mask_pos = None
            if self._gs_cached_mask_pos is None or self._gs_cached_mask_pos.device != zT.device:
                m = torch.from_numpy(self._gs_cached_m_bits_np.astype(np.int8)).to(zT.device)
                self._gs_cached_mask_pos = (m.view(1, C, H, W_) == 1)
            mask_pos = self._gs_cached_mask_pos  # (1,C,H,W)
            z_abs = zT.abs()
            # bit=1 -> +|z|, bit=0 -> -|z| (keep magnitude, only fix sign bins)
            z_fixed = torch.where(mask_pos, z_abs, -z_abs)
            return z_fixed

        # Fallback: no-op repair for unsupported watermark families in this script.
        return zT
    def ARR_refine(self, zT: torch.Tensor, prompt: str, W: bytes, K: bytes,
                   wm_cfg: WatermarkConfig, ARR_cfg: ARRConfig) -> torch.Tensor:
        """ARR loop (REPAIR-ONLY mode).

        We remove the surrogate gradient update to make this stage a *pure watermark repair*.
        Concretely, for T_r iterations we only apply the repair operator (fix GS sign-bin) on zT.

        This configuration treats ARR as a repair-only step rather than an alignment or attack step.
        but a deterministic watermark-recovery step.
        """
        if int(ARR_cfg.T_r) <= 0:
            return zT
        # No gradients needed in repair-only mode
        with torch.no_grad():
            for _ in range(int(ARR_cfg.T_r)):
                zT = self.repair_only_wm(zT, W=W, K=K, wm_cfg=wm_cfg)
        return zT

# -----------------------------
# Example runnable main
# -----------------------------



def _read_prompts_txt(path: str):
    prompts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('#'):
                continue
            prompts.append(s)
    if not prompts:
        raise ValueError(f"No prompts found in: {path}")
    return prompts


def _sanitize_filename(s: str, max_len: int = 120) -> str:
    # keep it filesystem-safe
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff\-_,.;()\[\]{} ]+", "_", s)
    s = s.replace(' ', '_')
    return s[:max_len] if len(s) > max_len else s


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _pil_from_tensor(img_01: torch.Tensor):
    # img_01: (3,H,W) in [0,1]
    img_01 = img_01.detach().clamp(0, 1).cpu()
    to_pil = transforms.ToPILImage()
    return to_pil(img_01)


def _make_grid_pil(imgs, rows: int, cols: int, pad: int = 8, bg=(255,255,255)):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    W = cols * w + (cols - 1) * pad
    H = rows * h + (rows - 1) * pad
    canvas = Image.new('RGB', (W, H), color=bg)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            x = c * (w + pad)
            y = r * (h + pad)
            canvas.paste(imgs[idx], (x, y))
    return canvas


def main():
    """
    Batch generator (Gaussian Shading (official embed) + B_sens projection mix + ARR + DPM scheduler).

    What it does (aligned with eval/vis_generate-slice_multi-12.05.py):
      - read prompts from a txt file
      - for each prompt, generate N images (default N=rows*cols)
      - save a 4x4 grid per prompt
      - save all single images under outdir/sliced/
      - write a manifest.csv for later NSFW scoring

    Notes:
      - We only generate *watermarked* images (no NoWM branch).
      - We keep changes minimal: reuse the existing workflow (ssp + freqSSM + ARR + full sampling).
      - Scheduler is DPM-Solver Multistep, consistent with the aligned generator configuration.
    """
    import argparse
    import csv
    import gc
    from datetime import datetime
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    parser = argparse.ArgumentParser(description='Batch generate Tree-Ring watermarked images (freqSSM + DPM).')

    # IO
    parser.add_argument('--prompts', type=str, required=True, help='Path to a .txt file, one prompt per line.')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory.')
    parser.add_argument('--prompt_set', type=str, default='', help='Optional prompt set name in manifest.')
    parser.add_argument('--group', type=str, default='TRWM_freqSSM_DPM', help='Group name in manifest.')
    parser.add_argument('--label', type=str, default='TRWM', help='Label name in manifest.')

    # SD
    parser.add_argument('--model_id', type=str, default='/home/yancy/work/dm_backdoor_latent_space/checkpoints/sd21')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--cfg', type=float, default=7.5)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)

    # Grid / batch
    parser.add_argument('--rows', type=int, default=4)
    parser.add_argument('--cols', type=int, default=4)
    parser.add_argument('--num_per_prompt', type=int, default=0, help='If 0, use rows*cols.')
    parser.add_argument('--gen_bs', type=int, default=1, help='Micro-batch size per prompt to control GPU memory. Generate in chunks and stitch into a grid. If <=0, use full num_per_prompt.')
    parser.add_argument('--lambda1', type=float, default=0.8, help='SSM mixing weight for the GS watermarked latent: zT = lambda1*z_wm + lambda2*z_sens, with lambda1^2+lambda2^2=1. lambda2 is derived from lambda1.')

    # Save zT for decoding (bypass inversion)
    parser.add_argument('--save_zt', action='store_true', help='Save initial/refined zT latents (.pt) under outdir/latents for direct GS decoding (no inversion).')
    parser.add_argument('--save_zt_mode', type=str, default='both', choices=['SSM','refined','both'], help="Which zT to save: 'SSM' (after SSM), 'refined' (after ARR+repair), or 'both'.")
    parser.add_argument('--save_zt_fp16', action='store_true', help='Save zT tensors in fp16 to reduce disk (default fp32). Not recommended if you want exact reproducibility debugging.')

    # Export-only: generate ONE repaired zT tensor (shape [n_zt,4,64,64]) and exit (no image decoding).
    parser.add_argument("--export_zt_only", action="store_true", help="Only export repaired zT (.pt) and exit early (skip image sampling/decoding).")
    parser.add_argument("--export_latents_dir", type=str, default="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment", help="Where to save the exported zT tensor.")
    parser.add_argument("--export_latents_name", type=str, default="generate_GS_w_att.pt", help="Filename of the exported zT tensor.")
    parser.add_argument("--n_zt", type=int, default=16, help="Number of zT samples to export when --export_zt_only is set (default 16).")
    parser.add_argument("--zt_seed", type=int, default=12345, help="Seed for sampling z_wm and z_sens in GS SSM (controls diversity across the n_zt samples).")

    # Surrogate + ARR
    parser.add_argument('--margin', type=float, default=0.2, help='CLIP hinge margin.')
    parser.add_argument('--ARR_T_r', type=int, default=2)
    parser.add_argument('--ARR_eta', type=float, default=0.15)
    parser.add_argument('--ARR_mini_steps', type=int, default=6)

    # ssp
    parser.add_argument('--ssp_N_cal', type=int, default=12)
    parser.add_argument('--ssp_mini_steps', type=int, default=6)
    parser.add_argument('--ssp_energy_ratio', type=float, default=0.90)
    parser.add_argument('--ssp_d_sens_max', type=int, default=64)
    parser.add_argument('--ssp_d_wm', type=int, default=256)
    parser.add_argument('--ssp_cache', type=str, default='', help='Path to cached ssp bases (.pt). If exists, load and reuse; otherwise compute once and save.')

    # Gaussian Shading params (official-integrated)
    parser.add_argument('--gs_key_ones', action='store_true', help='Use 32B all-ones key (default if no other key provided).')
    parser.add_argument('--gs_key_hex', type=str, default=None, help='64 hex chars (=32B). Mutually exclusive with --gs_key_ones/--gs_key_bin.')
    parser.add_argument('--gs_key_bin', type=str, default=None, help='256-bit 01 string (=32B). Mutually exclusive with --gs_key_ones/--gs_key_hex.')

    parser.add_argument('--gs_nonce_zero', action='store_true', help='Use 12B all-zero nonce. If none provided, use a fixed nonce for reproducibility (NOT recommended for production).')
    parser.add_argument('--gs_nonce_hex', type=str, default=None, help='24 hex chars (=12B).')
    parser.add_argument('--gs_nonce_bin', type=str, default=None, help='96-bit 01 string (=12B).')

    parser.add_argument('--gs_seed', type=int, default=12345, help='Seed for base watermark bits (before diffusion/randomization).')
    parser.add_argument('--gs_ch', type=int, default=4, help='Channel diffusion factor fc (default 4 -> capacity 256 bits for SD2.1 64x64).')
    parser.add_argument('--gs_hw', type=int, default=4, help='Spatial diffusion factor fhw (default 4 -> capacity 256 bits for SD2.1 64x64).')
    # (Tree-Ring args removed in GS script)

    args = parser.parse_args()
    if (not args.gs_key_ones) and (args.gs_key_hex is None) and (args.gs_key_bin is None):
        args.gs_key_ones = True


    device = args.device
    torch.set_grad_enabled(True)

    prompts = _read_prompts_txt(args.prompts)

    outdir = args.outdir
    sliced_dir = os.path.join(outdir, 'sliced')
    _ensure_dir(outdir)
    _ensure_dir(sliced_dir)

    latents_dir = os.path.join(outdir, 'latents')
    if bool(args.save_zt):
        _ensure_dir(latents_dir)

    prompt_set = args.prompt_set.strip() or Path(args.prompts).stem

    # Use a fixed latent shape inferred from height/width
    # SD latent resolution = (H/8, W/8)
    H_lat = args.height // 8
    W_lat = args.width // 8

    # number of samples per prompt (grid) or export-only zT count
    if bool(getattr(args, "export_zt_only", False)):
        n = int(getattr(args, "n_zt", 16))
    else:
        n = args.num_per_prompt if args.num_per_prompt > 0 else (args.rows * args.cols)
        if n != args.rows * args.cols:
            raise ValueError("For now, please keep num_per_prompt == rows*cols (to simplify grid logic).")

    latent_shape = (n, 4, H_lat, W_lat)

    print('[Init] Loading Stable Diffusion pipeline...')
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,  # keep fp32 for gradient stability
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    sampler = DiffusionLatentSampler(pipe, device=torch.device(device), dtype=torch.float32)
    surrogate_impl = CLIPHingeSurrogate(device=device)

    workflow = AlignPreserveNaW(
        sampler=sampler,
        surrogate=lambda img, prompt: surrogate_impl(img, prompt, margin=float(args.margin)),
        latent_shape=latent_shape,
        device=device,
    )
    # SSM mixing weights (lambda1^2 + lambda2^2 = 1)
    workflow.lambda1 = float(args.lambda1)
    if workflow.lambda1 < 0:
        raise ValueError('--lambda1 must be >= 0')
    if workflow.lambda1 > 1:
        print(f'[WARN] lambda1={workflow.lambda1} > 1, clamping to 1')
        workflow.lambda1 = 1.0
    workflow.lambda2 = float((max(0.0, 1.0 - workflow.lambda1 * workflow.lambda1)) ** 0.5)
    workflow.gs_latent_seed = int(getattr(args, "zt_seed", 12345))

    cal = (prompts * ((args.ssp_N_cal + len(prompts) - 1) // len(prompts)))[: args.ssp_N_cal]
    ssp_cfg = sspConfig(
        N_cal=args.ssp_N_cal,
        d_sens_max=args.ssp_d_sens_max,
        energy_ratio=float(args.ssp_energy_ratio),
        d_wm=args.ssp_d_wm,
        mini_steps=args.ssp_mini_steps,
        guidance_scale=7.5,
        eta_ddim=0.0,
    )

    # default cache path: under outdir (per-run). For persistent reuse, set --ssp_cache to a fixed path.
    ssp_cache_path = args.ssp_cache.strip() if hasattr(args, "ssp_cache") and args.ssp_cache else os.path.join(
        outdir, f"ssp_cache_D{workflow.D}_Ns{ssp_cfg.N_cal}_er{float(ssp_cfg.energy_ratio):.3f}_bsensOnly.pt"
    )

    loaded = False
    if ssp_cache_path and os.path.exists(ssp_cache_path):
        try:
            ckpt = torch.load(ssp_cache_path, map_location="cpu")
            if (ckpt.get("D", None) == workflow.D) and (ckpt.get("bwm_mode", "") in ["bsens_only", "bsensOnly", "none"]):
                workflow.B_sens = ckpt["B_sens"].to(workflow.device)
                workflow.B_wm = None
                workflow.P_wm, workflow.P_free = None, None
                d_sens_val = int(ckpt.get("d_sens", workflow.B_sens.shape[1]))
                stats = {
                    "B_sens": workflow.B_sens,
                    "d_sens": torch.tensor([int(d_sens_val)], device=workflow.device),
                }
                loaded = True
                print(f"[ssp] loaded cache: {ssp_cache_path}")
            else:
                print(f"[ssp] cache mismatch, recompute. cache(D={ckpt.get('D')}, mode={ckpt.get('bwm_mode')}) != current(D={workflow.D}, mode=bsens_only)")
        except Exception as e:
            print(f"[ssp] failed to load cache ({ssp_cache_path}): {e}. Will recompute.")

    if not loaded:
        stats = workflow.run_ssp(cal, ssp_cfg)
        try:
            Path(ssp_cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "D": workflow.D,
                    "bwm_mode": "bsens_only",
                    "d_sens": int(stats["d_sens"].item()),
                    "B_sens": stats["B_sens"].detach().cpu(),
                    "ssp_cfg": {
                        "N_cal": int(ssp_cfg.N_cal),
                        "d_sens_max": int(ssp_cfg.d_sens_max),
                        "energy_ratio": float(ssp_cfg.energy_ratio),
                        "mini_steps": int(ssp_cfg.mini_steps),
                    },
                },
                ssp_cache_path,
            )
            print(f"[ssp] saved cache: {ssp_cache_path}")
        except Exception as e:
            print(f"[ssp] failed to save cache ({ssp_cache_path}): {e}")

    print(f"[ssp] d_sens={int(stats['d_sens'].item())}")

    W = b''
    # Gaussian Shading key/nonce parsing (official-integrated)
    from types import SimpleNamespace
    _gs = SimpleNamespace(
        key_ones=bool(args.gs_key_ones),
        key_hex=args.gs_key_hex,
        key_bin=args.gs_key_bin,
        nonce_zero=bool(args.gs_nonce_zero),
        nonce_hex=args.gs_nonce_hex,
        nonce_bin=args.gs_nonce_bin,
    )
    gs_key32 = parse_key_32bytes(_gs)
    gs_nonce12 = parse_nonce_12bytes(_gs)
    K = gs_key32


    # --- Watermark config (Gaussian Shading as gsqbin family) ---
    wm_cfg = WatermarkConfig(
        family='gsqbin',
        J=args.ssp_d_wm,
        tau=0.5,
        device=device,

        # gaussian shading
        gs_key32=gs_key32,
        gs_nonce12=gs_nonce12,
        gs_seed=int(args.gs_seed),
        gs_ch=int(args.gs_ch),
        gs_hw=int(args.gs_hw),
        gs_l=1,
    )

    # ARR and full sampling configs
    ARR_cfg = ARRConfig(
        T_r=int(args.ARR_T_r),
        eta=float(args.ARR_eta),
        normalize_grad=True,
        mini_steps=int(args.ARR_mini_steps),
        guidance_scale=7.5,
        eta_ddim=0.0,
    )
    full_cfg = DiffusionSamplerConfig(num_steps=int(args.steps), guidance_scale=float(args.cfg), eta=0.0, mini_steps=int(args.ARR_mini_steps))


    # -----------------------------
    # Export-only: generate ONE ARR-repaired zT tensor and exit
    # -----------------------------
    if bool(getattr(args, "export_zt_only", False)):
        export_dir = Path(str(args.export_latents_dir))
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = export_dir / str(args.export_latents_name)

        # Any prompt works here (ARR is repair-only now).
        prompt0 = prompts[0] if len(prompts) > 0 else ""

        # Step 2: SSM (GS watermark + B_sens projected free component)
        zT_all = workflow.SSM_sample(W=W, K=K, wm_cfg=wm_cfg).detach()  # (n,4,64,64)

        # Step 3: ARR (repair-only loop)
        gen_bs = int(args.gen_bs) if int(args.gen_bs) > 0 else n
        gen_bs = min(gen_bs, n)
        zT_ref_cpu = torch.empty_like(zT_all.detach().to("cpu"))
        for start in range(0, n, gen_bs):
            mb = min(gen_bs, n - start)
            zT = zT_all[start:start + mb].detach()
            zT_ref = workflow.ARR_refine(zT, prompt=prompt0, W=W, K=K, wm_cfg=wm_cfg, ARR_cfg=ARR_cfg)
            zT_ref_cpu[start:start + mb] = zT_ref.detach().to("cpu")

        save_obj = {
            "latents": zT_ref_cpu,
            "meta": {
                "method": "GaussianShading",
                "mode": "w_att_ARR_repair_only",
                "shape": list(zT_ref_cpu.shape),
                "gs": {
                    "seed": int(args.gs_seed),
                    "ch": int(args.gs_ch),
                    "hw": int(args.gs_hw),
                    "l": 1,
                    "key_hex": gs_key32.hex(),
                    "nonce_hex": gs_nonce12.hex(),
                    "pack_order": "np.packbits(bitorder=big)",
                },
                "SSM": {
                    "lambda1": float(workflow.lambda1),
                    "lambda2": float(workflow.lambda2),
                    "zt_seed": int(getattr(args, "zt_seed", 12345)),
                },
                "ssp": {
                    "d_sens": int(stats["d_sens"].item()),
                    "ssp_cache": str(ssp_cache_path),
                },
                "ARR": {
                    "T_r": int(ARR_cfg.T_r),
                    "repair_only": True,
                },
            },
        }
        torch.save(save_obj, str(export_path))
        print(f"[OK] Exported repaired zT: {zT_ref_cpu.shape} -> {export_path}")
        return
    manifest_path = os.path.join(sliced_dir, 'manifest.csv')
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['file', 'orig', 'prompt_set', 'group', 'row', 'col', 'label', 'label_idx', 'prompt'])

        for p_i, prompt in enumerate(prompts):
            print(f"\n[{p_i+1}/{len(prompts)}] prompt: {prompt}")

            # Output naming
            base = f"p{p_i:04d}_{_sanitize_filename(prompt)}"
            grid_name = f"grid_{base}.png"
            grid_path = os.path.join(outdir, grid_name)

            # Step 2: SSM (generate all zT once so we don't repeat the same chunk when micro-batching)
            zT_all = workflow.SSM_sample(W=W, K=K, wm_cfg=wm_cfg).detach()  # (n,4,H,W)

            # Optionally save SSM zT (pre-ARR) for direct decoding
            zT_all_cpu = None
            zT_ref_cpu = None
            if bool(args.save_zt) and args.save_zt_mode in ['SSM', 'both', 'refined']:
                # Keep a CPU copy for saving/decoding; does not affect generation.
                zT_all_cpu = zT_all.detach().to('cpu')
                if bool(args.save_zt_fp16):
                    zT_all_cpu = zT_all_cpu.to(torch.float16)
            if bool(args.save_zt) and args.save_zt_mode in ['refined', 'both']:
                # We'll fill this during micro-batching (refined zT after ARR+repair).
                # Store on CPU to avoid extra VRAM.
                zT_ref_cpu = torch.empty_like(zT_all_cpu if zT_all_cpu is not None else zT_all.detach().to('cpu'))
                if bool(args.save_zt_fp16):
                    zT_ref_cpu = zT_ref_cpu.to(torch.float16)

            # Micro-batch to avoid OOM during ARR (backprop through UNet is the big memory hog)
            gen_bs = int(args.gen_bs) if int(args.gen_bs) > 0 else n
            gen_bs = min(gen_bs, n)

            pil_imgs = [None] * n

            for start in range(0, n, gen_bs):
                mb = min(gen_bs, n - start)
                zT = zT_all[start:start + mb].detach().requires_grad_(True)

                # Step 3: ARR refine (optional)
                if ARR_cfg.T_r > 0:
                    zT_refined = workflow.ARR_refine(zT, prompt=prompt, W=W, K=K, wm_cfg=wm_cfg, ARR_cfg=ARR_cfg)
                else:
                    zT_refined = zT


                # Save refined zT (after ARR+repair) for direct decoding
                if zT_ref_cpu is not None:
                    zT_ref_cpu[start:start + mb] = zT_refined.detach().to('cpu').to(zT_ref_cpu.dtype)

                # Full sampling (no grad)
                zT_refined = zT_refined.detach()
                latents_final = workflow.sampler.sample_latents_infer(zT_refined, prompt, full_cfg)
                imgs = workflow.sampler.decode_latents(latents_final)  # (mb,3,H,W) in [0,1]

                # Save singles + fill grid buffer
                for j in range(mb):
                    idx = start + j
                    r = idx // args.cols
                    c = idx % args.cols
                    pil = _pil_from_tensor(imgs[j])
                    pil_imgs[idx] = pil

                    fn = f"{base}_r{r}_c{c}.png"
                    fp = os.path.join(sliced_dir, fn)
                    pil.save(fp)
                    w.writerow([fp, grid_path, prompt_set, args.group, r, c, args.label, idx, prompt])

                # Cleanup chunk
                del zT, zT_refined, latents_final, imgs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Save grid
            grid = _make_grid_pil(pil_imgs, rows=args.rows, cols=args.cols, pad=8)
            grid.save(grid_path)

            # Save zT tensors for decoding (per-prompt)
            if bool(args.save_zt):
                save_obj = {
                    'prompt': prompt,
                    'prompt_index': int(p_i),
                    'base': base,
                    'n': int(n),
                    'latent_shape': (int(n), int(4), int(H_lat), int(W_lat)),
                    'lambda1': float(workflow.lambda1),
                    'lambda2': float(workflow.lambda2),
                    'ARR': {'T_r': int(ARR_cfg.T_r), 'eta': float(ARR_cfg.eta), 'mini_steps': int(ARR_cfg.mini_steps)},
                    'gen': {'steps': int(args.steps), 'cfg': float(args.cfg), 'height': int(args.height), 'width': int(args.width)},
                    'gs': {
                        'seed': int(args.gs_seed),
                        'ch': int(args.gs_ch),
                        'hw': int(args.gs_hw),
                        'l': 1,
                        'key_hex': (gs_key32.hex()),
                        'nonce_hex': (gs_nonce12.hex()),
                        'pack_order': 'np.packbits_msb',                },
                }
                # Optional: also save the SSM components for debugging/analysis
                if args.save_zt_mode in ['SSM', 'both'] and getattr(workflow, '_last_z_wm', None) is not None:
                    save_obj['z_wm'] = workflow._last_z_wm.detach().to('cpu').to(zT_all_cpu.dtype if zT_all_cpu is not None else torch.float32)
                if args.save_zt_mode in ['SSM', 'both'] and getattr(workflow, '_last_z_sens', None) is not None:
                    save_obj['z_sens'] = workflow._last_z_sens.detach().to('cpu').to(zT_all_cpu.dtype if zT_all_cpu is not None else torch.float32)

                if args.save_zt_mode in ['SSM', 'both'] and zT_all_cpu is not None:
                    save_obj['zT_SSM'] = zT_all_cpu
                if args.save_zt_mode in ['refined', 'both'] and zT_ref_cpu is not None:
                    save_obj['zT_refined'] = zT_ref_cpu
                zt_name = f"zT_{base}.pt"
                torch.save(save_obj, os.path.join(latents_dir, zt_name))
            # Cleanup (prompt-level)
            del zT_all, zT_all_cpu, zT_ref_cpu, pil_imgs, grid
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nDone. Saved grids under: {outdir}")
    print(f"Singles + manifest: {sliced_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
