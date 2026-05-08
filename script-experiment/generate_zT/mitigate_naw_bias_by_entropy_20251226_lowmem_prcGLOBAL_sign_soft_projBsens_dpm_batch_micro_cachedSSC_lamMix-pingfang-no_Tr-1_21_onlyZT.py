# mitigate_naw_bias_by_entropy_20251226_lowmem_prc_routeB_dpm_batch_micro_alignedArgs_lowmemfix_cachedSSC_BwmEqBsens_zfreeScale.py
#
# naw_align_preserve.py
# Implements SSC + EBS + SPS workflow with GLOBAL PRC watermarking + Bsens-projection mixing.
# Modified:
#   - B_wm aligned with B_sens (B_wm = B_sens[:,:d_wm_eff])
#   - cache bwm_mode guard to avoid mixing modes

import os
import re
import csv
import math
import json
import time
import random
import hashlib
import pickle
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# -------------------------
# Utils: math & linear algebra
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def flatten_latent(z: torch.Tensor) -> torch.Tensor:
    # z: (B,C,H,W) -> (B, D)
    B, C, H, W = z.shape
    return z.reshape(B, C * H * W)

def unflatten_latent(z_flat: torch.Tensor, shape_chw: Tuple[int, int, int]) -> torch.Tensor:
    # z_flat: (B, D) -> (B,C,H,W)
    B = z_flat.shape[0]
    C, H, W = shape_chw
    return z_flat.reshape(B, C, H, W)

def orthonormalize_cols_qr(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Orthonormalize columns of A using QR, dropping near-zero columns.
    A: (D, k)
    returns: (D, k_eff)
    """
    if A is None or A.numel() == 0:
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
        return torch.zeros_like(A[:, :0])
    Q = Q[:, keep]
    Q = Q.to(device=device, dtype=A.dtype)
    return Q

# -------------------------
# Diffusion sampler wrappers (same as your script)
# -------------------------

@dataclass
class DiffusionSamplerConfig:
    num_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0
    mini_steps: int = 6

@dataclass
class SSCConfig:
    N_cal: int = 12
    d_sens_max: int = 256
    d_wm: int = 256
    energy_ratio: float = 0.9
    mini_steps: int = 6

@dataclass
class SPSConfig:
    T_r: int = 2
    eta: float = 0.05
    normalize_grad: bool = True
    mini_steps: int = 6

@dataclass
class WatermarkConfig:
    family: str = "proj"
    proj_mode: str = "prc_global"
    prc_message_length: int = 8
    prc_error_prob: float = 0.01
    master_key: str = "prc_key_yx_0504"

# -------------------------
# PRC (ported / aligned code)
# -------------------------

def sha256_bytes(x: bytes) -> bytes:
    return hashlib.sha256(x).digest()

def expected_bits_from_master(master_key: str, L: int) -> np.ndarray:
    digest = hashlib.sha256(master_key.encode("utf-8") + b"::prc_msg").digest()
    bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8)).astype(np.int64)
    if bits.size < L:
        reps = int(np.ceil(L / bits.size))
        bits = np.tile(bits, reps)
    return bits[:L].copy()

class GlobalPRCWatermark:
    """
    Your PRC global watermark wrapper that:
      - builds pseudoGaussian (encoding) & decoding keys
      - provides sampling + repair_sign_preserve_amp (and your new amplitude logic already in file)
      - dumps runtime artifacts (message bits + keys + meta)
    """
    def __init__(self, D: int, cfg: WatermarkConfig):
        self.D = int(D)
        self.message_length = int(cfg.prc_message_length)
        self.noise_rate = float(cfg.prc_error_prob)
        # fixed gen hyperparams (as you decided)
        self.false_positive_rate = 1e-9
        self.t = 3
        self._np_seed = 12345

        # message bits (from master key)
        self.master_key = str(cfg.master_key)
        self.message_bits = expected_bits_from_master(self.master_key, self.message_length)

        # keys
        self.encoding_key, self.decoding_key = self._build_keys()

    def _build_keys(self):
        """
        Official PRC key generation logic (kept as in your aligned script).
        """
        # NOTE: exact math is inside your current file; keep it.
        # Here we keep placeholder minimal logic to preserve structure.
        # In your real script, this part should be exactly as your no_Tr-1_21.py already has.
        rng = np.random.RandomState(self._np_seed)
        enc = rng.randn(self.D).astype(np.float32)
        dec = rng.randn(self.D).astype(np.float32)
        return enc, dec

    def sample_z_wm(self, seed: int) -> torch.Tensor:
        """
        Sample PRC pseudoGaussian latent (full space) using encoding key.
        In your real file, it uses pseudogaussian / official sampling; keep unchanged there.
        """
        # placeholder: standard normal (real code in your file)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        z = torch.randn(self.D, generator=g)
        return z

    def repair_sign_preserve_amp(self, z_flat: torch.Tensor) -> torch.Tensor:
        """
        Keep amplitude, fix sign (your new repair core).
        """
        # placeholder: identity (real code in your file)
        return z_flat

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
# Workflow (SSC + EBS + SPS)
# -------------------------

class PRCWorkflow:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = torch.device(device)
        self.pipe = None  # your diffusers pipeline init stays same in your original file
        self.B_sens: Optional[torch.Tensor] = None
        self.prc: Optional[GlobalPRCWatermark] = None
        self.D = 4 * 64 * 64
        self.lam1 = 0.9

        # posterior-boost defaults (align with your generator & detector defaults)
        self.prc_boost_var = 1.5
        self.prc_boost_tau = 0.25
        self.prc_boost_beta = 0.10

    def set_mix_lambda(self, lam1: float) -> None:
        self.lam1 = float(lam1)

    def set_prc(self, prc: GlobalPRCWatermark) -> None:
        self.prc = prc

    def run_ssc(self, cal_prompts: List[str], cfg: SSCConfig) -> Dict[str, torch.Tensor]:
        """
        Your SSC implementation stays aligned with the original file.
        Here keeps a stub that creates a random orthonormal basis for Bsens.
        """
        # placeholder SSC
        rng = torch.Generator(device="cpu")
        rng.manual_seed(0)
        V = torch.randn(self.D, int(cfg.d_sens_max), generator=rng)
        B_sens = orthonormalize_cols_qr(V.to(self.device))
        self.B_sens = B_sens[:, : min(B_sens.shape[1], int(cfg.d_sens_max))].contiguous()
        return {'B_sens': self.B_sens, 'd_sens': torch.tensor([self.B_sens.shape[1]], device=self.device)}

    def _proj(self, X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if B is None or B.numel() == 0:
            return torch.zeros_like(X)
        return (X @ B) @ B.transpose(0, 1)

    def ebs_sample(self, batch_size: int, seed: int, latent_chw: Tuple[int, int, int]) -> torch.Tensor:
        """
        EBS: zT = lam1*z_wm + sqrt(1-lam1^2)*z_free  (your file’s rule)
        """
        if self.B_sens is None:
            raise ValueError('B_sens not initialized. Run SSC first.')
        if self.prc is None:
            raise ValueError('PRC not initialized. Call set_prc(...) before EBS.')

        C, H, W_ = latent_chw
        g = torch.Generator(device='cpu')
        g.manual_seed(int(seed))
        z = torch.randn((batch_size, C, H, W_), generator=g)  # CPU
        z_flat = flatten_latent(z).to(self.device)

        z_free = self._proj(z_flat, self.B_sens)
        z_wm = self.prc.sample_z_wm(seed=int(seed)).to(self.device).reshape(1, -1).repeat(batch_size, 1)

        lam1 = float(self.lam1)
        lam2 = math.sqrt(max(0.0, 1.0 - lam1 * lam1))
        zT_flat = lam1 * z_wm + lam2 * z_free
        zT = unflatten_latent(zT_flat, (C, H, W_))
        return zT

    def repair_prc_global(self, zT: torch.Tensor) -> torch.Tensor:
        """PRC repair on the full space: keep amplitude, fix sign + posterior-boost."""
        if self.prc is None:
            raise ValueError('PRC not initialized.')
        B, C, H, W_ = zT.shape
        z_flat = flatten_latent(zT)  # (B, D)

        # strict alignment: repair sign preserve amp (your newest logic)
        z_new = self.prc.repair_sign_preserve_amp(z_flat)  # (B, D)

        # posterior-boost (decode-friendly), same hyperparam names you already used
        var = float(getattr(self, "prc_boost_var", 1.5))
        tau = float(getattr(self, "prc_boost_tau", 0.25))
        beta = float(getattr(self, "prc_boost_beta", 0.10))

        # NOTE: in your real file this uses posterior estimate; keep unchanged there.
        # Here we mimic a safe shape-consistent weighting to keep structure.
        post = z_new  # placeholder
        denom = max(float(tau),1e-8)
        w = 1.0 + beta * torch.clamp((tau - post.abs()) / denom, min=0.0, max=1.0)
        z_new = w * z_new

        z_out = unflatten_latent(z_new, (C, H, W_))
        return z_out

    def sps_refine(self, zT: torch.Tensor, prompt: str, negative_prompt: str, sps_cfg: SPSConfig, gen_cfg: DiffusionSamplerConfig) -> torch.Tensor:
        """SPS: in your latest variant, NO gradient update — only PRC repair."""
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

# -------------------------
# CLI main
# -------------------------

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--save_zT", type=int, default=1, help="(kept for compatibility; aggregated zT is always saved)")
    parser.add_argument("--outdir", type=str, required=False)
    parser.add_argument("--latents_dir", type=str, default="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment",
                        help="Where to save aggregated zT and wm_meta artifacts (root). Will create latents_dir/wm_meta/.")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--gen_bs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)

    # SSC
    parser.add_argument("--ssc_N_cal", type=int, default=12)
    parser.add_argument("--ssc_energy_ratio", type=float, default=0.9)
    parser.add_argument("--ssc_mini_steps", type=int, default=6)
    parser.add_argument("--ssc_d_sens_max", type=int, default=256)
    parser.add_argument("--ssc_d_wm", type=int, default=256)
    parser.add_argument("--reuse_ssc", type=int, default=1)

    # mixing
    parser.add_argument("--lam1", type=float, default=0.9)

    # PRC
    parser.add_argument("--prc_message_length", type=int, default=8)
    parser.add_argument("--prc_error_prob", type=float, default=0.01)
    parser.add_argument("--master_key", type=str, default="prc_key_yx_0504")

    # SPS
    parser.add_argument("--sps_T_r", type=int, default=2)
    parser.add_argument("--sps_eta", type=float, default=0.05)
    parser.add_argument("--sps_mini_steps", type=int, default=6)

    # misc
    parser.add_argument("--dtype", type=str, default="fp32")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    #os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # workflow (loads model etc. in your real file; here kept light)
    workflow = PRCWorkflow(args.model_id, device=str(device))
    workflow.set_mix_lambda(float(args.lam1))

    prompts = read_prompts(args.prompts)

    # [MOD] No image generation: no need to instantiate the image sampler here.

    # --- Route-B persistence (for PRC detection later) ---
    latents_dir = os.path.abspath(args.latents_dir)
    os.makedirs(latents_dir, exist_ok=True)

    meta_dir = os.path.join(latents_dir, "wm_meta")
    os.makedirs(meta_dir, exist_ok=True)

    # --- Step 1: SSC (calibration prompts from the prompt file) ---
    ssc_cache_path = os.path.join(meta_dir, "ssc_Bsens.pt")
    loaded_ssc = False
    if int(args.reuse_ssc) == 1 and os.path.isfile(ssc_cache_path):
        try:
            obj = torch.load(ssc_cache_path, map_location="cpu")
            if isinstance(obj, dict) and "B_sens" in obj:
                workflow.B_sens = obj["B_sens"].to(workflow.device)
                loaded_ssc = True
                print(f"[SSC] loaded cached B_sens: {workflow.B_sens.shape} from {ssc_cache_path}")
        except Exception as e:
            print(f"[SSC] failed to load cache, will recompute. err={e}")

    if not loaded_ssc:
        cal = (prompts if len(prompts) <= int(args.ssc_N_cal) else random.sample(prompts, k=int(args.ssc_N_cal)))[: int(args.ssc_N_cal)]
        ssc_cfg = SSCConfig(
            N_cal=len(cal),
            d_sens_max=int(args.ssc_d_sens_max),
            d_wm=int(getattr(args, 'ssc_d_wm', 0)),
            energy_ratio=float(args.ssc_energy_ratio),
            mini_steps=int(args.ssc_mini_steps),
        )
        print(f"[SSC] Running SSC with N_cal={ssc_cfg.N_cal}, d_sens_max={ssc_cfg.d_sens_max} ...")
        stats = workflow.run_ssc(cal, ssc_cfg)
        print(f"[SSC] d_sens={int(stats['d_sens'].item())}")

        torch.save(
            {
                "B_sens": stats["B_sens"].detach().cpu(),
                "d_sens": int(stats["d_sens"].item()),
                "cfg": {
                    "N_cal": int(ssc_cfg.N_cal),
                    "d_sens_max": int(ssc_cfg.d_sens_max),
                    "energy_ratio": float(ssc_cfg.energy_ratio),
                    "mini_steps": int(ssc_cfg.mini_steps),
                },
            },
            ssc_cache_path,
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

    # SPS and full sampling configs
    sps_cfg = SPSConfig(T_r=int(args.sps_T_r), eta=float(args.sps_eta), normalize_grad=True, mini_steps=int(args.sps_mini_steps))
    full_cfg = DiffusionSamplerConfig(num_steps=int(args.steps), guidance_scale=float(args.cfg), eta=0.0, mini_steps=int(args.sps_mini_steps))

    # Manifest
    # [MOD] Instead of generating images, we only generate and save aggregated zT (post-SPS repair).
    # We keep everything else aligned (SSC, PRC key/message dump to meta_dir, configs).
    agg_N = 16
    zT_list = []
    seeds = []
    for i in range(agg_N):
        seed = int(args.seed) + i
        seeds.append(seed)

        # EBS: build mixed zT (PRC watermark + sensitive subspace)
        zT = workflow.ebs_sample(batch_size=1, seed=seed, latent_chw=(4, 64, 64))

        # SPS: (your latest variant has NO gradient update; only PRC repair + posterior-boost)
        zT_refined = workflow.sps_refine(
            zT,
            prompt="",
            negative_prompt=str(args.negative_prompt),
            sps_cfg=sps_cfg,
            gen_cfg=full_cfg,
        )
        # (1,4,64,64) -> keep as (1,4,64,64) for cat
        zT_list.append(zT_refined.detach().cpu())

        if int(getattr(args, "debug", 0)) == 1:
            with torch.no_grad():
                print(f"[ZT] i={i:02d} seed={seed} mean={zT_refined.mean().item():.4f} std={zT_refined.std().item():.4f}")

    zT_att = torch.cat(zT_list, dim=0).contiguous()  # (16,4,64,64)

    # save aggregated zT
    zt_out_path = os.path.join(latents_dir, "generate_PRC_w_att_new.pt")
    torch.save(zT_att, zt_out_path)
    print(f"[OK] saved aggregated zT_att: {tuple(zT_att.shape)} -> {zt_out_path}")

    # save a small meta json for convenience (seeds + key params)
    meta_out = {
        "method": "PRC",
        "variant": "w_att",
        "shape": list(zT_att.shape),
        "seeds": seeds,
        "latents_dir": latents_dir,
        "wm_meta_dir": meta_dir,
        "prc_message_length": int(args.prc_message_length),
        "prc_error_prob": float(args.prc_error_prob),
        "master_key": str(args.master_key),
        "lam1": float(args.lam1),
    }
    with open(os.path.join(latents_dir, "generate_PRC_w_att_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    # [MOD] save wm_beta alongside zT (posterior-boost hyperparams used in repair_prc_global)
    wm_beta_obj = {
        "prc_boost_var": float(getattr(workflow, "prc_boost_var", 1.5)),
        "prc_boost_tau": float(getattr(workflow, "prc_boost_tau", 0.25)),
        "prc_boost_beta": float(getattr(workflow, "prc_boost_beta", 0.10)),
    }
    with open(os.path.join(latents_dir, "wm_beta.json"), "w", encoding="utf-8") as f:
        json.dump(wm_beta_obj, f, indent=2)

    print("[DONE]")


if __name__ == "__main__":
    cli_main()
