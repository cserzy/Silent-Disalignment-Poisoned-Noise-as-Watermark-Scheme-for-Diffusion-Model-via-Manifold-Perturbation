#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""detect_T2S_oms.py

OMS-aware T2S detector:
  image -> inversion(zT_oms) -> OMS inverse(zT_restored) -> original T2S decode

Built from detect_T2S.py with minimal changes:
  - keep original T2S meta loading + decode logic
  - only insert OMS inverse between inversion and decode
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import torch
from tqdm import tqdm
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

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

# Official imports (auto-try local T2SMark paths before requiring PYTHONPATH)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PRE_AP = argparse.ArgumentParser(add_help=False)
_PRE_AP.add_argument("--t2s_root", type=str, default="")
_PRE_ARGS, _ = _PRE_AP.parse_known_args()
_T2S_ROOT_CLI = str(_PRE_ARGS.t2s_root).strip()

# Keep original candidates; add project-level third_party fallbacks.
_T2S_CANDS_WITH_SRC: List[Tuple[Path, str]] = []
if _T2S_ROOT_CLI:
    _T2S_CANDS_WITH_SRC.append((Path(_T2S_ROOT_CLI).expanduser(), "--t2s_root"))
_T2S_CANDS_WITH_SRC.extend([
    (_SCRIPT_DIR, "auto candidate (legacy)"),
    (_SCRIPT_DIR.parent / "T2SMark", "auto candidate (legacy)"),
    (_SCRIPT_DIR.parent.parent / "T2SMark", "auto candidate (legacy)"),
    (_SCRIPT_DIR.parent.parent.parent / "third_party" / "T2SMark", "auto candidate (third_party fallback)"),
    (_SCRIPT_DIR.parent.parent.parent.parent / "third_party" / "T2SMark", "auto candidate (third_party fallback)"),
])

_T2S_USED_ROOT: Optional[Path] = None
_T2S_USED_SOURCE: str = ""
_T2S_TRIED: List[str] = []
_T2S_CANDS_SEEN = set()

for _cand, _src_type in _T2S_CANDS_WITH_SRC:
    _cand = _cand.resolve()
    if str(_cand) in _T2S_CANDS_SEEN:
        continue
    _T2S_CANDS_SEEN.add(str(_cand))
    _T2S_TRIED.append(f"{_cand}  (source={_src_type})")
    if (_cand / "src").is_dir():
        _cp = str(_cand)
        _src = str(_cand / "src")
        if _cp not in sys.path:
            sys.path.insert(0, _cp)
        if _src not in sys.path:
            sys.path.insert(0, _src)
        _T2S_USED_ROOT = _cand
        _T2S_USED_SOURCE = _src_type
        break

if _T2S_USED_ROOT is not None:
    print(f"[INFO] T2SMark root resolved: {_T2S_USED_ROOT} (source={_T2S_USED_SOURCE})", flush=True)
else:
    _tried_msg = "\n  - " + "\n  - ".join(_T2S_TRIED)
    raise RuntimeError(
        "Failed to locate T2SMark root (expected a directory containing `src`).\n"
        "Please pass --t2s_root /path/to/T2SMark, or place it in a known auto-candidate location.\n"
        f"Tried candidates:{_tried_msg}"
    )

_LIGHTWEIGHT_T2S_MODE = True
_INVERSION_BACKEND_NAME = "lightweight/manual inversion"
_LEGACY_INVPIPE_DISABLED = True
_OFFICIAL_HELPER_STATUS: Dict[str, Any] = {
    "src.utils_imported": False,  # intentionally bypassed to avoid datasets/aiohttp chain
    "src.utils_reason": "intentionally_skipped_for_dependency_minimized_mode",
    "src.t2s_imported": False,
    "src.t2s_error": "",
    "src.inversion_imported": False,  # lazy import when image path is used
    "src.inversion_error": "legacy InvPipe intentionally disabled for diffusers compatibility",
}

try:
    from src.t2s import T2SMark
    _OFFICIAL_HELPER_STATUS["src.t2s_imported"] = True
except Exception as e:
    _OFFICIAL_HELPER_STATUS["src.t2s_error"] = repr(e)
    _tried_msg = "\n  - " + "\n  - ".join(_T2S_TRIED)
    raise RuntimeError(
        "Failed to import required module `src.t2s` after resolving T2SMark root.\n"
        f"Chosen root: {_T2S_USED_ROOT} (source={_T2S_USED_SOURCE})\n"
        f"Module error: {e}\n"
        f"Tried candidates:{_tried_msg}"
    )

def backward_ddim(x_t, alpha_t, alpha_tm1, eps_xt):
    """Same DDIM update used by the legacy T2S inversion pipeline."""
    return (
        alpha_tm1 ** 0.5
        * (
            (alpha_t ** -0.5 - alpha_tm1 ** -0.5) * x_t
            + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
        )
        + x_t
    )


class LightweightStableInversionBackend:
    """
    Minimal Stable Diffusion inversion backend used to avoid the legacy
    T2SMark InversableStableDiffusionPipeline.from_pretrained() path, which is
    incompatible with the current diffusers version in Hijacking.
    """

    def __init__(
        self,
        *,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: PNDMScheduler,
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        self.device = torch.device("cpu")
        self._progress_bar_disable = True
        self.vae_scale_factor = getattr(self.vae.config, "scaling_factor", 0.18215)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        torch_dtype: torch.dtype,
    ) -> "LightweightStableInversionBackend":
        model_path = Path(model_id)
        print(f"[load] tokenizer <- {model_path / 'tokenizer'}", flush=True)
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

        print(f"[load] text_encoder <- {model_path / 'text_encoder'}", flush=True)
        text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=torch_dtype,
        )

        print(f"[load] vae <- {model_path / 'vae'}", flush=True)
        vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch_dtype,
        )

        print(f"[load] unet <- {model_path / 'unet'}", flush=True)
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=torch_dtype,
        )

        print(f"[load] scheduler <- {model_path / 'scheduler'}", flush=True)
        scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")

        return cls(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )

    def to(self, device: torch.device) -> "LightweightStableInversionBackend":
        self.device = torch.device(device)
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.to(self.device)
        return self

    def set_progress_bar_config(self, disable: bool = True, **_: Any) -> None:
        self._progress_bar_disable = bool(disable)

    @torch.inference_mode()
    def get_image_latents(self, image: torch.Tensor, sample: bool = True) -> torch.Tensor:
        image = image.to(device=self.device, dtype=self.vae.dtype)
        encoding_dist = self.vae.encode(image).latent_dist
        encoding = encoding_dist.sample() if sample else encoding_dist.mode()
        return encoding * float(self.vae_scale_factor)

    @torch.inference_mode()
    def encode_prompt(
        self,
        prompt: Any,
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = torch.device(device)
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        attention_mask = text_inputs.attention_mask.to(device) if hasattr(text_inputs, "attention_mask") else None
        prompt_embeds = self.text_encoder(
            text_inputs.input_ids.to(device),
            attention_mask=attention_mask,
        )[0]
        prompt_embeds = prompt_embeds.to(device=device, dtype=self.text_encoder.dtype)
        prompt_embeds = prompt_embeds.repeat_interleave(int(num_images_per_prompt), dim=0)

        negative_prompt_embeds: Optional[torch.Tensor] = None
        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            negative_inputs = self.tokenizer(
                negative_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            negative_attention_mask = (
                negative_inputs.attention_mask.to(device)
                if hasattr(negative_inputs, "attention_mask")
                else None
            )
            negative_prompt_embeds = self.text_encoder(
                negative_inputs.input_ids.to(device),
                attention_mask=negative_attention_mask,
            )[0]
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=self.text_encoder.dtype)
            negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(
                int(num_images_per_prompt), dim=0
            )
        return prompt_embeds, negative_prompt_embeds

    @torch.inference_mode()
    def naive_forward_diffusion(
        self,
        *,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        do_classifier_free_guidance = float(guidance_scale) > 1.0
        try:
            self.scheduler.set_timesteps(int(num_inference_steps), device=self.device)
        except TypeError:
            self.scheduler.set_timesteps(int(num_inference_steps))
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        latents = latents.to(device=self.device, dtype=self.unet.dtype)
        latents = latents * self.scheduler.init_noise_sigma

        if do_classifier_free_guidance and text_embeddings.shape[0] == latents.shape[0]:
            null_embeds, _ = self.encode_prompt("", self.device, latents.shape[0], False, None)
            text_embeddings = torch.cat([null_embeds, text_embeddings], dim=0)
        else:
            text_embeddings = text_embeddings.to(device=self.device, dtype=self.text_encoder.dtype)

        time_iter = reversed(timesteps_tensor)
        if self._progress_bar_disable:
            iterator = time_iter
        else:
            iterator = tqdm(time_iter, total=len(timesteps_tensor), desc="inv_steps", leave=False)

        for t in iterator:
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + float(guidance_scale) * (noise_pred_text - noise_pred_uncond)

            t_idx = int(t.item()) if torch.is_tensor(t) else int(t)
            prev_timestep = (
                t_idx - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            )
            alpha_prod_t = self.scheduler.alphas_cumprod[t_idx]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
            latents = backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )
        return latents


InversableStableDiffusionPipeline = None  # kept for compatibility diagnostics only


def _get_inversion_pipeline_cls():
    global InversableStableDiffusionPipeline
    if InversableStableDiffusionPipeline is not None:
        return InversableStableDiffusionPipeline
    try:
        from src.inversion.inverse_stable_diffusion import InversableStableDiffusionPipeline as _InvPipe
        InversableStableDiffusionPipeline = _InvPipe
        _OFFICIAL_HELPER_STATUS["src.inversion_imported"] = True
        return InversableStableDiffusionPipeline
    except Exception as e:
        _OFFICIAL_HELPER_STATUS["src.inversion_error"] = repr(e)
        raise RuntimeError(
            "Failed to import required module `src.inversion.inverse_stable_diffusion`.\n"
            f"Chosen T2S root: {_T2S_USED_ROOT} (source={_T2S_USED_SOURCE})\n"
            "This error is unrelated to `src.utils` (which is intentionally bypassed).\n"
            f"Module error: {e}"
        )

# Reuse OMS inverse logic from script-experiment/oms_repair_pt.py
_OMS_ROOT = _SCRIPT_DIR.parent
if str(_OMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_OMS_ROOT))
try:
    from oms_repair_pt import (  # type: ignore
        flatten_latent_4d,
        load_q_pt,
        resolve_forward_aux_from_q_and_meta,
        run_apply_with_fallback,
        run_blended_inverse_with_fallback,
        unflatten_latent_2d,
    )
except Exception as e:
    raise RuntimeError(
        "Failed to import OMS helpers from script-experiment/oms_repair_pt.py. "
        f"Details: {e}"
    )


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


def collect_images_from_run_dir(run_dir: str) -> List[Path]:
    """Prefer run_dir/sliced/*.png, then run_dir/*.png, then recursive *.png."""
    rd = Path(run_dir)
    if not rd.is_dir():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    sliced = sorted((rd / "sliced").glob("*.png"))
    if sliced:
        return sliced
    direct = sorted(rd.glob("*.png"))
    if direct:
        return direct
    return sorted(rd.rglob("*.png"))


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
    Returns normalized:
      master_keys: [16, key_len] int32
      session_keys: [16, key_len] int32
      msgs: [16, msg_len] int32
      settings: dict
      tau: float
      key_channel_idx: int
      key_length, msg_length
    """
    pack = torch.load(cluster_meta_pt, map_location="cpu")
    if not isinstance(pack, dict):
        raise TypeError(f"cluster_meta_pt must be a dict .pt, got: {type(pack)}")

    settings = pack.get("settings", {}) if isinstance(pack.get("settings", {}), dict) else {}

    # Try common keys
    if "master_keys" in pack and "keys" in pack and "msgs" in pack:
        master_keys = _to_bits_tensor(pack["master_keys"], device)
        session_keys = _to_bits_tensor(pack["keys"], device)
        msgs = _to_bits_tensor(pack["msgs"], device)
    else:
        # tolerate alternate naming
        mk = pack.get("master_key_bits", None) or pack.get("master_keys_bits", None) or pack.get("master_key", None)
        sk = pack.get("session_key_bits", None) or pack.get("session_keys", None) or pack.get("keys", None)
        mg = pack.get("msg_bits", None) or pack.get("msgs_bits", None) or pack.get("msgs", None)
        if mk is None or sk is None or mg is None:
            raise KeyError(f"cluster_meta_pt missing keys. got={list(pack.keys())}")
        master_keys = _to_bits_tensor(mk, device)
        session_keys = _to_bits_tensor(sk, device)
        msgs = _to_bits_tensor(mg, device)

    # Ensure 2D [K,L]
    if master_keys.ndim == 1:
        master_keys = master_keys.unsqueeze(0)
    if session_keys.ndim == 1:
        session_keys = session_keys.unsqueeze(0)
    if msgs.ndim == 1:
        msgs = msgs.unsqueeze(0)

    tau = float(settings.get("tau", pack.get("t2s_tau", 0.674)))
    key_channel_idx = int(settings.get("key_channel_idx", pack.get("key_channel_idx", 0)))

    key_length = int(settings.get("key_length", master_keys.shape[-1]))
    msg_length = int(settings.get("msg_length", msgs.shape[-1]))

    # Sanity
    if master_keys.shape[0] < 1 or session_keys.shape[0] < 1 or msgs.shape[0] < 1:
        raise ValueError("Empty keys/msg in cluster_meta_pt")
    if master_keys.shape[0] != session_keys.shape[0]:
        # allow master_keys repeated but still should match K
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


_DECODE_DEVICE_LOGGED = False


def ensure_t2s_decode_device(t: torch.Tensor, decode_device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(t):
        raise TypeError(f"Expected tensor for T2S decode, got {type(t)}")
    return t.to(device=decode_device)


def decode_latents(post_reversed_latents: torch.Tensor,
                   key_channel_idx: int,
                   t2s_key: T2SMark,
                   t2s_msg: T2SMark,
                   master_key: torch.Tensor,
                   session_key: torch.Tensor,
                   msg: torch.Tensor,
                   decode_device: torch.device,
                   verbose: bool = False) -> Dict[str, float]:
    """
    Match official decode(): use master key to detect key channel, recover session key,
    then decode msg with recovered session key (and also oracle session key for debug).
    """
    global _DECODE_DEVICE_LOGGED

    if decode_device.type != "cuda":
        raise RuntimeError(
            "T2SMark decode currently requires CUDA because upstream decode hardcodes .cuda()."
        )

    msg_channel_idx = [i for i in range(4) if i != key_channel_idx]
    reversed_key_channel = ensure_t2s_decode_device(
        post_reversed_latents[0, key_channel_idx, :, :], decode_device
    )
    reversed_msg_channel = ensure_t2s_decode_device(
        post_reversed_latents[0, msg_channel_idx, :, :], decode_device
    )
    master_key = ensure_t2s_decode_device(master_key, decode_device)
    session_key = ensure_t2s_decode_device(session_key, decode_device)
    msg = ensure_t2s_decode_device(msg, decode_device)

    if verbose and (not _DECODE_DEVICE_LOGGED):
        print(
            f"[INFO] T2S decode device: {decode_device}, "
            f"reversed_key_channel.device={reversed_key_channel.device}, "
            f"reversed_msg_channels.device={reversed_msg_channel.device}",
            flush=True,
        )
        _DECODE_DEVICE_LOGGED = True

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


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def apply_oms_inverse_to_latent(
    zt_oms: torch.Tensor,
    q_obj: Dict[str, Any],
    *,
    oms_q_pt: str,
    oms_meta_json: str,
    device: torch.device,
    verbose: bool,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Restore latent from OMS domain to original T2S domain.
    Reuses the same inverse chain as oms_repair_pt.py.
    """
    if not torch.is_tensor(zt_oms):
        raise TypeError(f"Expected tensor zt_oms, got {type(zt_oms)}")
    if zt_oms.ndim != 4:
        raise ValueError(f"Expected 4D latent [N,C,H,W], got shape={tuple(zt_oms.shape)}")

    x_all = flatten_latent_4d(zt_oms.to(torch.float32))
    perm = q_obj["perm"]
    inv_perm = q_obj["inv_perm"]
    q_blocks = q_obj["q_blocks"]

    if int(perm.numel()) != int(x_all.shape[1]):
        raise ValueError(f"OMS q dimension mismatch: q_D={perm.numel()}, latent_D={x_all.shape[1]}")

    aux = resolve_forward_aux_from_q_and_meta(
        q_obj=q_obj,
        q_pt_path=oms_q_pt,
        q_meta_json=oms_meta_json,
    )
    alpha_eff = float(aux.get("blend_alpha", 1.0))
    match_eff = _to_bool(aux.get("match_target_std", False))
    rescale_factor = float(aux.get("rescale_factor", 1.0))

    if not (0.0 <= alpha_eff <= 1.0):
        raise ValueError(f"Invalid OMS blend_alpha={alpha_eff}")
    if match_eff and abs(rescale_factor) < 1e-12:
        raise ValueError(f"Invalid OMS rescale_factor={rescale_factor}")

    # Undo forward std scaling first.
    x_unscaled = x_all / rescale_factor if match_eff else x_all

    solve_summary: Optional[Dict[str, Any]] = None
    if alpha_eff == 1.0:
        x_restored, used_device = run_apply_with_fallback(
            x_2d_cpu=x_unscaled,
            perm=perm,
            inv_perm=inv_perm,
            q_blocks=q_blocks,
            inverse=True,
            device=device,
            verbose=verbose,
        )
        inverse_mode = "pure_Q_inverse" if not match_eff else "blended_plus_rescale_inverse"
    else:
        x_restored, solve_summary, used_device = run_blended_inverse_with_fallback(
            y_2d_cpu=x_unscaled,
            perm=perm,
            inv_perm=inv_perm,
            q_blocks=q_blocks,
            alpha=alpha_eff,
            device=device,
            verbose=verbose,
        )
        inverse_mode = "blended_inverse" if not match_eff else "blended_plus_rescale_inverse"

    zt_restored = unflatten_latent_2d(x_restored, tuple(zt_oms.shape))
    info = {
        "inverse_mode": inverse_mode,
        "blend_alpha": float(alpha_eff),
        "match_target_std": bool(match_eff),
        "rescale_factor": float(rescale_factor),
        "solve_summary": solve_summary,
        "used_device": used_device,
        "param_source": aux.get("source", {}),
    }
    return zt_restored, info


def maybe_resize_pil(img, target: Optional[int]):
    if target is None:
        return img
    if img.size[0] == target and img.size[1] == target:
        return img
    return img.resize((target, target))


def pil_to_tensor(img) -> torch.Tensor:
    """
    Lightweight replacement for src.utils.to_tensor:
      PIL RGB -> torch tensor [1,3,H,W] in [-1, 1], float32 on CPU.
    """
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected image array shape: {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.shape[2] != 3:
        raise ValueError(f"Expected 3 channels after conversion, got shape: {arr.shape}")
    arr = arr / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    t = t * 2.0 - 1.0
    return t


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
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "name", "mode", "image", "image_path", "run_dir", "model_id",
        "zT_idx", "inv_steps",
        "zT_oms_path", "zT_restored_path",
        "cluster_meta_pt", "cluster_meta_json", "oms_q_pt",
        "oms_blend_alpha", "oms_match_target_std", "oms_rescale_factor", "oms_inverse_mode",
        "tau", "key_channel_idx", "nbits",
        "bit_acc", "ber",
        "detected", "norm1_no_w", "norm1_w",
        "acc_key", "acc_msg", "acc_msg_oracle",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            rob = r.get("robustness", {}) if isinstance(r.get("robustness", {}), dict) else {}
            mt = r.get("meta", {}) if isinstance(r.get("meta", {}), dict) else {}
            bit_acc = float(rob.get("acc_msg", 0.0))
            flat = {
                "name": r.get("name", ""),
                "mode": r.get("mode", ""),
                "image": r.get("name", ""),
                "image_path": r.get("image", ""),
                "run_dir": mt.get("run_dir", ""),
                "model_id": mt.get("model_id", ""),
                "zT_idx": r.get("zT_idx", -1),
                "inv_steps": mt.get("num_inversion_steps", mt.get("inv_steps", "")),
                "zT_oms_path": mt.get("zT_oms_path", ""),
                "zT_restored_path": mt.get("zT_restored_path", ""),
                "cluster_meta_pt": mt.get("cluster_meta_pt", ""),
                "cluster_meta_json": mt.get("cluster_meta_json", ""),
                "oms_q_pt": mt.get("oms_q_pt", ""),
                "oms_blend_alpha": mt.get("oms_blend_alpha", ""),
                "oms_match_target_std": mt.get("oms_match_target_std", ""),
                "oms_rescale_factor": mt.get("oms_rescale_factor", ""),
                "oms_inverse_mode": mt.get("oms_inverse_mode", ""),
                "tau": mt.get("tau", ""),
                "key_channel_idx": mt.get("key_channel_idx", ""),
                "nbits": mt.get("nbits", ""),
                "bit_acc": bit_acc,
                "ber": float(1.0 - bit_acc),
                "detected": rob.get("detected", ""),
                "norm1_no_w": rob.get("norm1_no_w", ""),
                "norm1_w": rob.get("norm1_w", ""),
                "acc_key": rob.get("acc_key", ""),
                "acc_msg": rob.get("acc_msg", ""),
                "acc_msg_oracle": rob.get("acc_msg_oracle", ""),
            }
            w.writerow(flat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t2s_root", type=str, default="",
                    help="Optional explicit T2SMark root. If provided, it is prioritized before auto candidates.")

    # ---- inputs ----
    ap.add_argument("--cluster_meta_pt", type=str,
                    default="/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_T2S_w_att_meta.pt",
                    help="Meta .pt dict containing master_keys/session_keys/msgs/settings (default points to your experiment folder).")
    ap.add_argument("--cluster_meta_json", type=str, default=None,
                    help="Optional T2S meta json (can override tau/key_channel_idx).")

    ap.add_argument("--zT16_pt", type=str, default=None,
                    help="Optional: [16,4,64,64] zT tensor .pt, for direct decode path.")

    ap.add_argument("--images_glob", type=str, default=None,
                    help="Optional: glob for images (script will glob itself), e.g. /path/to/sliced/*.png")
    ap.add_argument("--run_dir", type=str, default=None,
                    help="Optional run dir. If images_glob is empty, scan run_dir/sliced/*.png then run_dir/*.png.")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Optional output dir for json/csv and optional zT saves.")

    ap.add_argument("--model_id", type=str, default=None,
                    help="Stable Diffusion diffusers path or HF id (required if image inversion is used).")
    ap.add_argument("--oms_q_pt", type=str, required=True,
                    help="OMS q pt path.")
    ap.add_argument("--oms_meta_json", type=str, default="",
                    help="Optional OMS meta json fallback.")

    # ---- inversion options ----
    ap.add_argument("--num_inversion_steps", type=int, default=10,
                    help="Inversion steps (default 10, aligns with official option defaults).")
    ap.add_argument("--inv_steps", type=int, default=-1,
                    help="Alias of --num_inversion_steps.")
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
    ap.add_argument("--dtype", type=str, default="", choices=["", "fp16", "fp32"],
                    help="Optional alias to override fp16/fp32.")
    ap.add_argument("--compute_auc", action="store_true",
                    help="Compute AUC-like summary using norm1_no_w vs norm1_w.")
    ap.add_argument("--print_each", action="store_true", default=True)
    ap.add_argument("--save_zt_oms", action="store_true",
                    help="Save inversion zT before OMS inverse.")
    ap.add_argument("--save_zt_restored", action="store_true",
                    help="Save zT after OMS inverse (used for T2S decode).")
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    if int(args.inv_steps) > 0:
        args.num_inversion_steps = int(args.inv_steps)
    if str(args.dtype).strip():
        args.fp16 = str(args.dtype).lower() == "fp16"

    need_image_path = bool(args.images_glob or args.run_dir)
    if (not need_image_path) and (not args.zT16_pt):
        raise SystemExit("Need at least one of (--images_glob/--run_dir) or --zT16_pt")

    decode_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if decode_device.type != "cuda":
        raise RuntimeError(
            "T2SMark decode currently requires CUDA because upstream decode hardcodes .cuda()."
        )
    device = decode_device

    torch_dtype = torch.float16 if args.fp16 else torch.float32

    # output paths
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.run_dir:
        out_dir = Path(args.run_dir)
    elif args.images_glob:
        out_dir = Path(os.path.dirname(args.images_glob) or ".")
    elif args.zT16_pt:
        out_dir = Path(args.zT16_pt).resolve().parent
    else:
        out_dir = Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = Path(args.out_json) if args.out_json else (out_dir / "detect_t2s_oms.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_csv if args.out_csv else str(out_dir / "detect_t2s_oms.csv")
    out_lat_dir = out_dir / "latents"
    if args.save_zt_oms or args.save_zt_restored:
        out_lat_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] model_id={args.model_id}", flush=True)
    print(f"[INFO] run_dir={args.run_dir}", flush=True)
    print(f"[INFO] images_glob={args.images_glob}", flush=True)
    print(f"[INFO] out_dir={out_dir}", flush=True)
    print(f"[INFO] cluster_meta_pt={args.cluster_meta_pt}", flush=True)
    print(f"[INFO] cluster_meta_json={args.cluster_meta_json}", flush=True)
    print(f"[INFO] oms_q_pt={args.oms_q_pt}", flush=True)
    print(f"[INFO] oms_meta_json_provided={bool(str(args.oms_meta_json).strip())}", flush=True)
    print(
        f"[INFO] T2S dependency mode: {'lightweight/dependency-minimized' if _LIGHTWEIGHT_T2S_MODE else 'default'}",
        flush=True,
    )
    print(
        f"[INFO] official helper import status: src.utils={_OFFICIAL_HELPER_STATUS['src.utils_imported']} "
        f"(reason={_OFFICIAL_HELPER_STATUS['src.utils_reason']}), "
        f"src.t2s={_OFFICIAL_HELPER_STATUS['src.t2s_imported']}, "
        f"src.inversion(lazy)={_OFFICIAL_HELPER_STATUS['src.inversion_imported']}",
        flush=True,
    )
    print(
        f"[INFO] inversion backend={_INVERSION_BACKEND_NAME}; "
        f"legacy_t2smark_invpipe_disabled={_LEGACY_INVPIPE_DISABLED}",
        flush=True,
    )
    print(f"[INFO] t2s_decode_device={decode_device}", flush=True)

    # ---- load meta (true source of keys/msg/settings) ----
    meta_pack = load_cluster_meta(args.cluster_meta_pt, device=device)
    K = int(meta_pack["K"])
    tau = float(meta_pack["tau"])
    key_channel_idx = int(meta_pack["key_channel_idx"])
    key_length = int(meta_pack["key_length"])
    msg_length = int(meta_pack["msg_length"])

    # Optional json override for tau/key_channel_idx
    if args.cluster_meta_json and Path(args.cluster_meta_json).is_file():
        try:
            with open(args.cluster_meta_json, "r", encoding="utf-8") as f:
                meta_j = json.load(f)
            if isinstance(meta_j, dict):
                if "tau" in meta_j:
                    tau = float(meta_j["tau"])
                if "key_channel_idx" in meta_j:
                    key_channel_idx = int(meta_j["key_channel_idx"])
                settings = meta_j.get("settings", {})
                if isinstance(settings, dict):
                    if "tau" in settings:
                        tau = float(settings["tau"])
                    if "key_channel_idx" in settings:
                        key_channel_idx = int(settings["key_channel_idx"])
        except Exception as e:
            print(f"[WARN] failed to read cluster_meta_json={args.cluster_meta_json}: {e}", flush=True)

    master_keys = meta_pack["master_keys"]         # [K, key_len]
    session_keys = meta_pack["session_keys"]       # [K, key_len]
    msgs = meta_pack["msgs"]                       # [K, msg_len]

    # ---- build T2S objects ----
    t2s_cache: Dict[Tuple[str, int, float], T2SMark] = {}
    t2s_key = build_t2s_objects(t2s_cache, "key", key_length, tau, (1, 64, 64))
    t2s_msg = build_t2s_objects(t2s_cache, "msg", msg_length, tau, (3, 64, 64))

    # ---- load OMS q ----
    q_obj = load_q_pt(args.oms_q_pt, block_mode="flat_chunk")
    oms_aux = resolve_forward_aux_from_q_and_meta(
        q_obj=q_obj,
        q_pt_path=args.oms_q_pt,
        q_meta_json=args.oms_meta_json,
    )
    oms_alpha = float(oms_aux.get("blend_alpha", 1.0))
    oms_match_std = _to_bool(oms_aux.get("match_target_std", False))
    oms_rescale = float(oms_aux.get("rescale_factor", 1.0))
    if oms_alpha == 1.0:
        inverse_hint = "pure_Q_inverse" if not oms_match_std else "blended_plus_rescale_inverse"
    else:
        inverse_hint = "blended_inverse" if not oms_match_std else "blended_plus_rescale_inverse"
    print(
        f"[INFO] OMS inverse defaults: mode={inverse_hint}, "
        f"blend_alpha={oms_alpha:.6f}, match_target_std={oms_match_std}, rescale_factor={oms_rescale:.8f}",
        flush=True,
    )

    # ---- (optional) inversion pipe ----
    pipe = None
    null_text_embeddings = None
    if need_image_path:
        if not args.model_id:
            raise SystemExit("--model_id is required when using image inversion path")
        print(
            "[INFO] building inversion backend via manual Stable Diffusion component loading; "
            "legacy T2SMark InvPipe.from_pretrained is intentionally bypassed",
            flush=True,
        )
        pipe = LightweightStableInversionBackend.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        null_text_embeddings = pipe.encode_prompt("", device, 1, False, None)[0]
        print(
            f"[INFO] inversion backend ready: class={pipe.__class__.__name__} "
            f"vae_scale_factor={getattr(pipe, 'vae_scale_factor', 'NA')}",
            flush=True,
        )

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

            zt_oms = zT16[i:i+1]  # [1,4,64,64]
            zt_oms_path = ""
            if args.save_zt_oms:
                zt_oms_path_obj = out_lat_dir / f"zT16_{i:02d}_inv_zT_oms.pt"
                torch.save({"zT": zt_oms.detach().float().cpu(), "index": i, "domain": "oms"}, str(zt_oms_path_obj))
                zt_oms_path = str(zt_oms_path_obj)

            zt, inv_info = apply_oms_inverse_to_latent(
                zt_oms,
                q_obj=q_obj,
                oms_q_pt=args.oms_q_pt,
                oms_meta_json=args.oms_meta_json,
                device=device,
                verbose=args.verbose,
            )
            zt_restored_path = ""
            if args.save_zt_restored:
                zt_restored_path_obj = out_lat_dir / f"zT16_{i:02d}_inv_zT_restored.pt"
                torch.save({"zT": zt.detach().float().cpu(), "index": i, "domain": "restored_t2s", "oms_inverse_info": inv_info}, str(zt_restored_path_obj))
                zt_restored_path = str(zt_restored_path_obj)

            rob = decode_latents(
                zt, key_channel_idx=key_channel_idx,
                t2s_key=t2s_key, t2s_msg=t2s_msg,
                master_key=mk, session_key=sk, msg=mg,
                decode_device=decode_device,
                verbose=args.verbose,
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
                    "run_dir": str(args.run_dir or ""),
                    "model_id": str(args.model_id or ""),
                    "cluster_meta_pt": str(args.cluster_meta_pt),
                    "cluster_meta_json": str(args.cluster_meta_json or ""),
                    "oms_q_pt": str(args.oms_q_pt),
                    "oms_blend_alpha": float(inv_info.get("blend_alpha", oms_alpha)),
                    "oms_match_target_std": int(bool(inv_info.get("match_target_std", oms_match_std))),
                    "oms_rescale_factor": float(inv_info.get("rescale_factor", oms_rescale)),
                    "oms_inverse_mode": str(inv_info.get("inverse_mode", "")),
                    "zT_oms_path": zt_oms_path,
                    "zT_restored_path": zt_restored_path,
                    "tau": tau,
                    "key_channel_idx": key_channel_idx,
                    "key_length": key_length,
                    "msg_length": msg_length,
                    "nbits": int(msg_length),
                    "num_inversion_steps": int(args.num_inversion_steps),
                },
            }
            results[name] = item
            rows.append(item)
            if args.print_each:
                print(f"[ZT] {name} oms_mode={inv_info['inverse_mode']} detected={rob['detected']:.0f} "
                      f"norm1_no_w={rob['norm1_no_w']:.3f} norm1_w={rob['norm1_w']:.3f} "
                      f"acc_key={rob['acc_key']:.3f} acc_msg={rob['acc_msg']:.3f} "
                      f"acc_msg_oracle={rob['acc_msg_oracle']:.3f}", flush=True)

    # ---- Path B: image inversion decode with OMS inverse ----
    if need_image_path:
        from PIL import Image

        if args.images_glob:
            images_glob = args.images_glob
            if os.path.isdir(images_glob):
                images_glob = os.path.join(images_glob, "*.png")
            img_paths = sorted([Path(p) for p in glob.glob(images_glob)])
        else:
            images_glob = f"{args.run_dir}/(sliced|direct|recursive)"
            img_paths = collect_images_from_run_dir(args.run_dir)

        if args.max_images and args.max_images > 0:
            img_paths = img_paths[:args.max_images]
        if not img_paths:
            raise FileNotFoundError(f"No images matched: {images_glob}")

        for img_path in tqdm(img_paths, desc="detect(images_oms)"):
            combo_id = parse_combo_id_from_name(img_path.name)
            if combo_id is None:
                m = re.search(r"_(\d{2})\b", img_path.stem)
                combo_id = int(m.group(1)) if m else 0

            if combo_id >= K:
                if args.print_each:
                    print(f"[SKIP] {img_path.name}: combo_id={combo_id} >= K={K}", flush=True)
                continue

            mk = master_keys[combo_id]
            sk = session_keys[combo_id]
            mg = msgs[combo_id]

            pil = Image.open(img_path).convert("RGB")
            pil = maybe_resize_pil(pil, args.resize)
            image_tensor = pil_to_tensor(pil).to(device).to(torch_dtype)
            latents = pipe.get_image_latents(image_tensor, sample=False)

            if args.use_prompt:
                prompt = ""
                text_embeddings = pipe.encode_prompt(prompt, device, 1, False, None)[0]
            else:
                text_embeddings = null_text_embeddings

            reversed_latents_oms = pipe.naive_forward_diffusion(
                latents=latents.to(torch_dtype),
                text_embeddings=text_embeddings.to(torch_dtype),
                num_inference_steps=int(args.num_inversion_steps),
                guidance_scale=float(args.inv_guidance),
            )
            print(f"[IMG] inversion done: {img_path.name} shape={tuple(reversed_latents_oms.shape)}", flush=True)

            zt_oms_path = ""
            if args.save_zt_oms:
                zt_oms_path_obj = out_lat_dir / f"{img_path.stem}_inv_zT_oms.pt"
                torch.save(
                    {
                        "zT": reversed_latents_oms.detach().float().cpu(),
                        "image": img_path.name,
                        "image_path": str(img_path),
                        "run_dir": str(args.run_dir or ""),
                        "domain": "oms",
                        "oms_q_pt": str(args.oms_q_pt),
                    },
                    str(zt_oms_path_obj),
                )
                zt_oms_path = str(zt_oms_path_obj)

            reversed_latents, inv_info = apply_oms_inverse_to_latent(
                reversed_latents_oms,
                q_obj=q_obj,
                oms_q_pt=args.oms_q_pt,
                oms_meta_json=args.oms_meta_json,
                device=device,
                verbose=args.verbose,
            )
            print(
                f"[IMG] OMS inverse: mode={inv_info['inverse_mode']} "
                f"alpha={inv_info['blend_alpha']:.6f} "
                f"match_std={inv_info['match_target_std']} "
                f"rescale={inv_info['rescale_factor']:.8f}",
                flush=True,
            )
            if args.verbose:
                m = float(reversed_latents.mean().item())
                s = float(reversed_latents.std(unbiased=False).item())
                solved = None
                if isinstance(inv_info.get("solve_summary"), dict):
                    solved = inv_info["solve_summary"].get("all_blocks_solved")
                print(f"[IMG][verbose] restored mean={m:.8f} std={s:.8f} all_blocks_solved={solved}", flush=True)

            zt_restored_path = ""
            if args.save_zt_restored:
                zt_restored_path_obj = out_lat_dir / f"{img_path.stem}_inv_zT_restored.pt"
                torch.save(
                    {
                        "zT": reversed_latents.detach().float().cpu(),
                        "image": img_path.name,
                        "image_path": str(img_path),
                        "run_dir": str(args.run_dir or ""),
                        "domain": "restored_t2s",
                        "oms_q_pt": str(args.oms_q_pt),
                        "oms_inverse_info": inv_info,
                    },
                    str(zt_restored_path_obj),
                )
                zt_restored_path = str(zt_restored_path_obj)

            rob = decode_latents(
                reversed_latents, key_channel_idx=key_channel_idx,
                t2s_key=t2s_key, t2s_msg=t2s_msg,
                master_key=mk, session_key=sk, msg=mg,
                decode_device=decode_device,
                verbose=args.verbose,
            )

            name = img_path.name
            item = {
                "name": name,
                "mode": "image_inversion_oms",
                "zT_idx": combo_id,
                "image": str(img_path),
                "robustness": rob,
                "meta": {
                    "combo_id": int(combo_id),
                    "run_dir": str(args.run_dir or ""),
                    "model_id": str(args.model_id or ""),
                    "cluster_meta_pt": str(args.cluster_meta_pt),
                    "cluster_meta_json": str(args.cluster_meta_json or ""),
                    "oms_q_pt": str(args.oms_q_pt),
                    "oms_blend_alpha": float(inv_info.get("blend_alpha", oms_alpha)),
                    "oms_match_target_std": int(bool(inv_info.get("match_target_std", oms_match_std))),
                    "oms_rescale_factor": float(inv_info.get("rescale_factor", oms_rescale)),
                    "oms_inverse_mode": str(inv_info.get("inverse_mode", "")),
                    "zT_oms_path": zt_oms_path,
                    "zT_restored_path": zt_restored_path,
                    "tau": float(tau),
                    "key_channel_idx": int(key_channel_idx),
                    "num_inversion_steps": int(args.num_inversion_steps),
                    "inv_guidance": float(args.inv_guidance),
                    "use_prompt": bool(args.use_prompt),
                    "nbits": int(msg_length),
                },
            }
            results[name] = item
            rows.append(item)
            if args.print_each:
                bit_acc = float(rob["acc_msg"])
                print(f"[IMG] {name} combo={combo_id:02d} mode={inv_info['inverse_mode']} "
                      f"bit_acc={bit_acc:.3f} detected={rob['detected']:.0f} "
                      f"norm1_no_w={rob['norm1_no_w']:.3f} norm1_w={rob['norm1_w']:.3f}", flush=True)

    # ---- summary ----
    if args.compute_auc and rows:
        results["summary_all"] = compute_auc_block(rows)
        zt_rows = [r for r in rows if str(r.get("mode", "")).startswith("zT16")]
        img_rows = [r for r in rows if str(r.get("mode", "")).startswith("image_inversion")]
        if zt_rows:
            results["summary_zT16"] = compute_auc_block(zt_rows)
        if img_rows:
            results["summary_image_inversion"] = compute_auc_block(img_rows)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    write_csv(rows, out_csv)

    print(f"[OK] wrote json: {out_json}", flush=True)
    print(f"[OK] wrote csv:  {out_csv}", flush=True)
    if args.save_zt_oms or args.save_zt_restored:
        print(f"[OK] saved zT latents: {out_lat_dir}", flush=True)


if __name__ == "__main__":
    main()
