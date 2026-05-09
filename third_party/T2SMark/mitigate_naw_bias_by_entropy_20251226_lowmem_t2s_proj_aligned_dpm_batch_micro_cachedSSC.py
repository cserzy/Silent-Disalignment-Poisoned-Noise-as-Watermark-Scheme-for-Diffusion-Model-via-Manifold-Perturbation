#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2SMark aligned pipeline (DPM + cached SSC + prompt-batch generation).

This script is intentionally written as a *thin* wrapper on top of your
already-working PRC Route-B scaffold, to maximize consistency:

  - Same sampler/surrogate/SSC/SPS implementations and CLI style
  - DPM scheduler
  - SSC computed once and cached to outdir/wm_meta/ssc_basis.pt
  - Prompt loading + (sliced singles + 4x4 grid + manifest.csv) consistent with Treering/PRC

The only "new" part is the *T2SMark* watermark injection + a watermark-preserving
repair operator during SPS. For watermark injection we **reuse the official T2SMark
implementation** (the same class/methods your key_to_latent_cluster_official_t2s.py
uses), instead of re-implementing the encoding logic.

Notes
-----
1) This file assumes your environment already has the official T2SMark repo code
   importable (e.g., modules `src.t2s` and `src.utils`). If not, it will raise a
   clear error with a hint.
2) "Proj" here refers to your workflow (optimize z_T with CLIP score and then
   project/repair to keep watermark intact). The repair is tailored to T2SMark's
   Tail-Truncated Sampling constraints: keep tail positions (|z|>=tau) sign-stable
   and clamp central positions to [-tau, tau].
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch
from PIL import Image

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


# -----------------------------
# Utilities (aligned with your Treering script)
# -----------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_prompts_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def _sanitize_filename(s: str, max_len: int = 80) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-\.]+", "", s)
    return s[:max_len] if len(s) > max_len else s


def _pil_from_tensor(img_01: torch.Tensor) -> Image.Image:
    # img_01: (3,H,W) in [0,1]
    img = (img_01.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8)
    img = img.permute(1, 2, 0).numpy()
    return Image.fromarray(img)


def _make_grid_pil(pil_imgs: List[Image.Image], rows: int, cols: int, pad: int = 8) -> Image.Image:
    assert len(pil_imgs) == rows * cols
    w, h = pil_imgs[0].size
    grid_w = cols * w + (cols - 1) * pad
    grid_h = rows * h + (rows - 1) * pad
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    for idx, im in enumerate(pil_imgs):
        r = idx // cols
        c = idx % cols
        x = c * (w + pad)
        y = r * (h + pad)
        grid.paste(im, (x, y))
    return grid


# -----------------------------
# Import your proven PRC scaffold (sampler/surrogate/SSC/SPS)
# -----------------------------

def _import_base():
    import importlib

    base_mod_name = "mitigate_naw_bias_by_entropy_20251226_lowmem_prc_routeB_dpm_batch_micro_alignedArgs_lowmemfix_cachedSSC"
    try:
        return importlib.import_module(base_mod_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to import base module '{base_mod_name}'. "
            "Please run this script in the same directory as the PRC scaffold file, "
            "or add that directory to PYTHONPATH.\n"
            f"Original error: {e}"
        )


BASE = _import_base()


# -----------------------------
# T2SMark: official injection + watermark-preserving repair
# -----------------------------

class T2SOfficialWrapper:
    """Wrap official T2SMark implementation (same as key_to_latent_cluster_official_t2s.py).

    We intentionally do **not** re-implement encoding.
    """

    def __init__(
        self,
        latent_hw: Tuple[int, int],
        key_length: int,
        msg_length: int,
        tau: float,
        key_channel_idx: int = 0,
        device: torch.device = torch.device("cuda"),
        seed: int = 12345,
        fix_key: bool = True,
    ):
        self.H_lat, self.W_lat = int(latent_hw[0]), int(latent_hw[1])
        self.key_length = int(key_length)
        self.msg_length = int(msg_length)
        self.tau = float(tau)
        self.key_channel_idx = int(key_channel_idx)
        self.device = device
        self.seed = int(seed)
        self.fix_key = bool(fix_key)

        # Import official modules
        try:
            from src.t2s import T2SMark  # type: ignore
            import src.utils as t2s_utils  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Cannot import official T2SMark modules (expected `src.t2s` and `src.utils`).\n"
                "Make sure the official T2SMark repo is in PYTHONPATH (e.g., `export PYTHONPATH=/path/to/T2SMark:$PYTHONPATH`).\n"
                f"Original error: {e}"
            )

        self.T2SMark = T2SMark
        self.t2s_utils = t2s_utils

        # Two-stage instances (key: 1 channel; msg: 3 channels)
        self.t2s_key = self.T2SMark(latent_shape=(1, self.H_lat, self.W_lat), m=self.key_length, tau=self.tau)
        self.t2s_msg = self.T2SMark(latent_shape=(3, self.H_lat, self.W_lat), m=self.msg_length, tau=self.tau)

        # Fixed master key & message for reproducibility / blind verify (recommended)
        self.master_key: Optional[torch.Tensor] = None
        self.msg: Optional[torch.Tensor] = None
        if self.fix_key:
            self._init_fixed_payload()

    def _init_fixed_payload(self) -> None:
        self.t2s_utils.set_random_seed(self.seed)
        # bits in {0,1}
        self.master_key = torch.randint(0, 2, (self.key_length,), device=self.device)
        self.msg = torch.randint(0, 2, (self.msg_length,), device=self.device)

    @torch.no_grad()
    def sample_zT(self, batch: int, prompt_id: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Return zT of shape (B,4,H_lat,W_lat) following the official two-stage encoding."""
        B = int(batch)
        # determinism: each prompt has its own seed base
        base_seed = self.seed + int(prompt_id) * 100_000

        z_out = torch.empty((B, 4, self.H_lat, self.W_lat), device=self.device, dtype=torch.float32)
        # store keys if you want; for blind detection you only need master_key + msg (if fixed)
        keys_int: List[int] = []

        for i in range(B):
            self.t2s_utils.set_random_seed(base_seed + i)

            # session key per-image
            key = torch.randint(0, 2, (self.key_length,), device=self.device)
            master_key = self.master_key
            msg = self.msg
            if not self.fix_key:
                # fully random payload per-image
                master_key = torch.randint(0, 2, (self.key_length,), device=self.device)
                msg = torch.randint(0, 2, (self.msg_length,), device=self.device)

            assert master_key is not None and msg is not None

            # encode key-channel and msg-channels
            z_k = self.t2s_key.encode(key, master_key)  # (1,H,W)
            z_b = self.t2s_msg.encode(msg, key)         # (3,H,W)

            # insert into 4 channels
            if self.key_channel_idx == 0:
                z = torch.cat([z_k, z_b], dim=0)
            elif self.key_channel_idx == 3:
                z = torch.cat([z_b, z_k], dim=0)
            else:
                # general insert (rarely used): key channel at idx, msg channels fill others in order
                parts: List[torch.Tensor] = []
                msg_ptr = 0
                for c in range(4):
                    if c == self.key_channel_idx:
                        parts.append(z_k)
                    else:
                        parts.append(z_b[msg_ptr:msg_ptr + 1])
                        msg_ptr += 1
                z = torch.cat(parts, dim=0)

            z_out[i] = z.to(torch.float32)
            # store as int (optional)
            keys_int.append(int("".join(["1" if b.item() > 0 else "0" for b in key]), 2))

        meta: Dict[str, Any] = {
            "seed": self.seed,
            "prompt_id": int(prompt_id),
            "fix_key": self.fix_key,
            "key_length": self.key_length,
            "msg_length": self.msg_length,
            "tau": self.tau,
            "key_channel_idx": self.key_channel_idx,
            "keys_int": keys_int,
        }
        if self.fix_key:
            assert self.master_key is not None and self.msg is not None
            meta["master_key_int"] = int("".join(["1" if b.item() > 0 else "0" for b in self.master_key]), 2)
            meta["msg_int"] = int("".join(["1" if b.item() > 0 else "0" for b in self.msg]), 2)
        return z_out, meta


def t2s_repair_with_base(z: torch.Tensor, z_base: torch.Tensor, tau: float) -> torch.Tensor:
    """Watermark-preserving repair for T2SMark.

    - Tail region (|z_base| >= tau): keep sign consistent with base, and ensure |z| >= tau.
    - Central region (|z_base| <  tau): clamp to [-tau, tau].

    This keeps the tail-truncated structure stable while still allowing SPS to optimize
    the entropy buffer (central region).
    """
    assert z.shape == z_base.shape
    tau = float(tau)

    base = z_base
    mask_tail = base.abs() >= tau
    base_sign = torch.sign(base)
    base_sign[base_sign == 0] = 1

    z_abs = z.abs()
    z_tail = base_sign * torch.clamp(z_abs, min=tau)
    z_center = torch.clamp(z, min=-tau, max=tau)
    return torch.where(mask_tail, z_tail, z_center)


def sps_refine_t2s(
    sampler: "BASE.DiffusionLatentSampler",
    surrogate_fn,
    zT: torch.Tensor,
    z_base: torch.Tensor,
    prompt: str,
    sps_cfg: "BASE.SPSConfig",
    tau: float,
    negative_prompt: str = "",
) -> torch.Tensor:
    """SPS loop that preserves T2SMark via `t2s_repair_with_base` after each update."""

    z = zT
    for _ in range(int(sps_cfg.T_r)):
        cfg = BASE.DiffusionSamplerConfig(
            num_steps=int(sps_cfg.mini_steps),
            guidance_scale=float(sps_cfg.guidance_scale),
            eta=float(getattr(sps_cfg, "eta_ddim", 0.0)),
            mini_steps=int(sps_cfg.mini_steps),
        )

        img = sampler.mini_sample_image(z, prompt, cfg, negative_prompt=negative_prompt)
        loss = surrogate_fn(img, prompt)
        g = torch.autograd.grad(loss, z, retain_graph=False, create_graph=False)[0]

        if bool(getattr(sps_cfg, "normalize_grad", True)):
            g = g / (g.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, 1, 1, 1))

        z = z - float(sps_cfg.eta) * g
        # Repair + detach to avoid graph explosion
        z = t2s_repair_with_base(z, z_base, tau=tau).detach().requires_grad_(True)

    return z.detach()


# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("T2SMark aligned NaW mitigation (DPM + cached SSC)")

    # IO / prompts
    p.add_argument("--prompts", type=str, required=True, help="Path to a txt file, one prompt per line")
    p.add_argument("--outdir", type=str, required=True, help="Output directory")
    p.add_argument("--prompt_set", type=str, default="", help="Optional prompt set name in manifest")
    p.add_argument("--group", type=str, default="T2SWM_proj_DPM", help="Group name in manifest")
    p.add_argument("--label", type=str, default="T2S", help="Label name in manifest")

    # model / sampler
    p.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-1", help="HF model id")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--negative", type=str, default="")

    # generation layout
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--num_per_prompt", type=int, default=16, help="Must equal rows*cols for grid")
    p.add_argument("--gen_bs", type=int, default=2, help="Micro-batch size for SPS + sampling")

    # surrogate / SPS
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--sps_T_r", type=int, default=2)
    p.add_argument("--sps_eta", type=float, default=0.15)
    p.add_argument("--sps_mini_steps", type=int, default=6)

    # SSC (cached)
    p.add_argument("--reuse_ssc", type=int, default=1, help="1: load cached SSC if exists")
    p.add_argument("--ssc_N_cal", type=int, default=64)
    p.add_argument("--ssc_d_sens_max", type=int, default=64)
    p.add_argument("--ssc_energy_ratio", type=float, default=0.90)
    p.add_argument("--ssc_d_wm", type=int, default=256)
    p.add_argument("--ssc_mini_steps", type=int, default=6)

    # T2SMark (official)
    p.add_argument("--t2s_key_length", type=int, default=16)
    p.add_argument("--t2s_msg_length", type=int, default=256)
    p.add_argument("--t2s_tau", type=float, default=0.674)  # normal quartile ~ 0.674
    p.add_argument("--t2s_key_channel_idx", type=int, default=0)
    p.add_argument("--t2s_seed", type=int, default=12345)
    p.add_argument("--t2s_fix_key", type=int, default=1)

    return p


def main():
    args = build_argparser().parse_args()

    device = torch.device(args.device)
    torch.set_grad_enabled(True)

    prompts = _read_prompts_txt(args.prompts)
    if not prompts:
        raise ValueError("No prompts found.")

    outdir = args.outdir
    sliced_dir = os.path.join(outdir, "sliced")
    meta_dir = os.path.join(outdir, "wm_meta")
    _ensure_dir(outdir)
    _ensure_dir(sliced_dir)
    _ensure_dir(meta_dir)

    prompt_set = args.prompt_set.strip() or Path(args.prompts).stem

    n = int(args.num_per_prompt)
    if n != int(args.rows) * int(args.cols):
        raise ValueError("For now keep num_per_prompt == rows*cols (grid logic aligned).")

    H_lat, W_lat = int(args.height) // 8, int(args.width) // 8
    latent_shape = (n, 4, H_lat, W_lat)

    print("[Init] Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    # DPM scheduler (aligned)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    sampler = BASE.DiffusionLatentSampler(pipe, device=device, dtype=torch.float32)
    surrogate = BASE.CLIPHingeSurrogate(device=str(device))

    # Keep the workflow object only for SSC caching (aligned with your PRC/Treering pipeline)
    workflow = BASE.AlignPreserveNaW(
        sampler=sampler,
        surrogate=lambda img, prompt: surrogate(img, prompt, margin=float(args.margin)),
        latent_shape=latent_shape,
        device=str(device),
    )

    # -----------------
    # Step 1: SSC (cached)
    # -----------------
    ssc_cache_path = os.path.join(meta_dir, "ssc_basis.pt")
    loaded_ssc = False
    if bool(int(args.reuse_ssc)) and os.path.isfile(ssc_cache_path):
        try:
            ckpt = torch.load(ssc_cache_path, map_location="cpu")
            B_wm = ckpt.get("B_wm", None)
            B_sens = ckpt.get("B_sens", None)
            if B_wm is None:
                raise ValueError("B_wm is None in cached ssc_basis.pt")
            if B_wm.shape[0] != workflow.D:
                raise ValueError(f"D mismatch: cached D={B_wm.shape[0]} vs current D={workflow.D}")
            workflow.B_wm = B_wm.to(device=workflow.device, dtype=torch.float32)
            workflow.B_sens = B_sens.to(device=workflow.device, dtype=torch.float32) if B_sens is not None else None
            d_sens = int(workflow.B_sens.shape[1]) if workflow.B_sens is not None else 0
            print(f"[SSC] Reusing cached SSC basis from {ssc_cache_path} (d_wm={int(workflow.B_wm.shape[1])}, d_sens={d_sens}).")
            loaded_ssc = True
        except Exception as e:
            print(f"[SSC] Failed to load cached SSC basis: {e}. Recomputing...")

    if not loaded_ssc:
        # deterministic: use first N_cal prompts, repeat if needed
        cal = (prompts * ((int(args.ssc_N_cal) + len(prompts) - 1) // len(prompts)))[: int(args.ssc_N_cal)]
        ssc_cfg = BASE.SSCConfig(
            N_cal=len(cal),
            d_sens_max=int(args.ssc_d_sens_max),
            d_wm=int(args.ssc_d_wm),
            energy_ratio=float(args.ssc_energy_ratio),
            mini_steps=int(args.ssc_mini_steps),
        )
        print(f"[SSC] Running SSC with N_cal={ssc_cfg.N_cal}, d_wm={ssc_cfg.d_wm}, d_sens_max={ssc_cfg.d_sens_max} ...")
        stats = workflow.run_ssc(cal, ssc_cfg)
        print(f"[SSC] d_sens={int(stats['d_sens'].item())}, d_wm={ssc_cfg.d_wm}")

        torch.save(
            {
                "B_wm": workflow.B_wm.detach().cpu() if workflow.B_wm is not None else None,
                "B_sens": workflow.B_sens.detach().cpu() if workflow.B_sens is not None else None,
                "ssc_cfg": vars(ssc_cfg),
                "latent_shape": latent_shape,
                "model_id": args.model_id,
                "scheduler": "DPMSolverMultistepScheduler",
            },
            ssc_cache_path,
        )
        with open(os.path.join(meta_dir, "cal_prompts.txt"), "w", encoding="utf-8") as f:
            for p in cal:
                f.write(p + "\n")

    # -----------------
    # Step 2: T2SMark official zT sampling
    # -----------------
    t2s = T2SOfficialWrapper(
        latent_hw=(H_lat, W_lat),
        key_length=int(args.t2s_key_length),
        msg_length=int(args.t2s_msg_length),
        tau=float(args.t2s_tau),
        key_channel_idx=int(args.t2s_key_channel_idx),
        device=device,
        seed=int(args.t2s_seed),
        fix_key=bool(int(args.t2s_fix_key)),
    )

    # Persist fixed payload (so later detection is reproducible)
    t2s_meta_out = {
        "t2s_key_length": int(args.t2s_key_length),
        "t2s_msg_length": int(args.t2s_msg_length),
        "t2s_tau": float(args.t2s_tau),
        "t2s_key_channel_idx": int(args.t2s_key_channel_idx),
        "t2s_seed": int(args.t2s_seed),
        "t2s_fix_key": int(args.t2s_fix_key),
    }
    if t2s.fix_key and t2s.master_key is not None and t2s.msg is not None:
        t2s_meta_out["master_key_bits"] = [int(x.item()) for x in t2s.master_key.detach().cpu()]
        t2s_meta_out["msg_bits"] = [int(x.item()) for x in t2s.msg.detach().cpu()]
    with open(os.path.join(meta_dir, "t2s_fixed_payload.json"), "w", encoding="utf-8") as f:
        json.dump(t2s_meta_out, f, indent=2, ensure_ascii=False)

    # SPS & full sampling configs
    sps_cfg = BASE.SPSConfig(
        T_r=int(args.sps_T_r),
        eta=float(args.sps_eta),
        normalize_grad=True,
        mini_steps=int(args.sps_mini_steps),
        guidance_scale=float(args.cfg),
        eta_ddim=0.0,
    )
    full_cfg = BASE.DiffusionSamplerConfig(
        num_steps=int(args.steps),
        guidance_scale=float(args.cfg),
        eta=0.0,
        mini_steps=int(args.sps_mini_steps),
    )

    manifest_path = os.path.join(sliced_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "orig", "prompt_set", "group", "row", "col", "label", "label_idx", "prompt"])

        for p_i, prompt in enumerate(prompts):
            print(f"\n[{p_i+1}/{len(prompts)}] prompt: {prompt}")
            base = f"p{p_i:04d}_{_sanitize_filename(prompt)}"
            grid_name = f"grid_{base}.png"
            grid_path = os.path.join(outdir, grid_name)

            # T2S official zT for this prompt
            zT_all, z_meta = t2s.sample_zT(batch=n, prompt_id=p_i)
            # store prompt-specific key list (optional)
            with open(os.path.join(meta_dir, f"t2s_keys_prompt{p_i:04d}.json"), "w", encoding="utf-8") as jf:
                json.dump(z_meta, jf, indent=2, ensure_ascii=False)

            gen_bs = int(args.gen_bs) if int(args.gen_bs) > 0 else n
            gen_bs = min(gen_bs, n)
            pil_imgs: List[Optional[Image.Image]] = [None] * n

            for start in range(0, n, gen_bs):
                mb = min(gen_bs, n - start)
                z_base = zT_all[start:start + mb].detach()
                zT = z_base.detach().requires_grad_(True)

                # Step 3: SPS refine (optional)
                if int(sps_cfg.T_r) > 0:
                    zT_refined = sps_refine_t2s(
                        sampler=sampler,
                        surrogate_fn=workflow.surrogate,
                        zT=zT,
                        z_base=z_base,
                        prompt=prompt,
                        sps_cfg=sps_cfg,
                        tau=float(args.t2s_tau),
                        negative_prompt=str(args.negative),
                    )
                else:
                    zT_refined = zT.detach()

                # Full sampling (no grad)
                latents_final = sampler.sample_latents_infer(zT_refined, prompt, full_cfg, negative_prompt=str(args.negative))
                imgs = sampler.decode_latents(latents_final)  # (mb,3,H,W) in [0,1]

                for j in range(mb):
                    idx = start + j
                    r = idx // int(args.cols)
                    c = idx % int(args.cols)
                    pil = _pil_from_tensor(imgs[j])
                    pil_imgs[idx] = pil

                    fn = f"{base}_r{r}_c{c}.png"
                    fp = os.path.join(sliced_dir, fn)
                    pil.save(fp)
                    w.writerow([fp, grid_path, prompt_set, args.group, r, c, args.label, idx, prompt])

                # cleanup chunk
                del z_base, zT, zT_refined, latents_final, imgs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Save grid
            assert all(x is not None for x in pil_imgs)
            grid = _make_grid_pil([x for x in pil_imgs if x is not None], rows=int(args.rows), cols=int(args.cols), pad=8)
            grid.save(grid_path)
            del zT_all, pil_imgs, grid
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nDone. Saved grids under: {outdir}")
    print(f"Singles + manifest: {sliced_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
